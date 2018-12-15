import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from base import BaseModel
from model.layers.vae_conv_layer import ConvVAE2d, VAE_SMALL

class HVAE_2L(BaseModel):
    """Hierarchical VAE with 2 latent spaces
        
    Sandard encoder with 4 layers
    Decoder is 3 layers, then the local conv_vae's decoder
    Can feed in either a pre-trained local decoder or untrained one:
    Train the thing end to end with loss = KLD1+KLD2+reconstruction error

    2d vae layer is fixed to be ConvVAE2d, with variable kernel size (8? 16, 32) (l2_kernel_size)
    
    l1_size: first latent variable size
    l1_width: number of convolution filters in first convolution in first layer
    l1_input_size: size of the output of l1 
    
    l2_size: second latent variable size, will be 2d grid of this
    l2_width: number of convolution filters in first layer
    l2_input_size: image size
    l2_kernel_size: the size of the kernel for the local vae layer
    l2_stride: Determined by how much the input is downsampled. (l1_input_size/l2_input_size)
    
    ??? In l1, use two fully connected layers to get mu,var.
    ??? In l2, this is just the conv output. Use 1x1 conv instead?
    """
    
    def __init__(self, img_size, in_channels=1, l2_latent_size=16, l2_width=64, l2_input_size=32, l2_kernel_size=16, l2_stride=2,
                                    l1_latent_size=32, l1_width=64, l1_input_size=8):
        super(HVAE_2L, self).__init__()
        
        self.img_size = img_size
        # Should this be automatically loaded instead of fixed vae with args?
        # would need to change config/train format or make loading fuction
        # should the small vae input and kernel be defined in small vae or in 2d layer?
        self.l2_conv_vae = ConvVAE2d(VAE_SMALL(latent_size=l2_latent_size, img_size=l2_kernel_size, in_channels=in_channels), 
                                     stride=l2_stride, padding=0, same=True, img_size=64)

        self.l2_latent_size = l2_latent_size
        self.l2_width = l2_width
        self.l2_input_size = l2_input_size
        self.l2_kernel_size = l2_kernel_size
        
        self.l1_latent_size = l1_latent_size
        self.l1_width = l1_width        
        self.l1_input_size = l1_input_size
        self.l1_last_conv_size = int(self.img_size/16) # 4 /2 downsamples
        
        self.elu = nn.ELU()
        self.enc_l1_conv1 = nn.Conv2d(in_channels, int(self.l1_width/8), kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_l1_conv2 = nn.Conv2d(int(self.l1_width/8), int(self.l1_width/4), kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_l1_conv3 = nn.Conv2d(int(self.l1_width/4), int(self.l1_width/2), kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_l1_conv4 = nn.Conv2d(int(self.l1_width/2), self.l1_width, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_l1_conv5 = nn.Conv2d(self.l1_width, self.l1_latent_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc_l1_bn1 = nn.BatchNorm2d(int(self.l1_width/8))
        self.enc_l1_bn2 = nn.BatchNorm2d(int(self.l1_width/4))
        self.enc_l1_bn3 = nn.BatchNorm2d(int(self.l1_width/2))
        self.enc_l1_bn4 = nn.BatchNorm2d(self.l1_width)

        self.dec_l1_conv1 = nn.ConvTranspose2d(self.l1_latent_size, self.l1_width, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=False)
        self.dec_l1_conv2 = nn.ConvTranspose2d(self.l1_width, int(self.l1_width/2), kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_l1_conv3 = nn.ConvTranspose2d(int(self.l1_width/2), int(2*self.l2_latent_size), kernel_size=5, stride=2, padding=2,  output_padding=1, bias=True)
        self.dec_l1_bn1 = nn.BatchNorm2d(self.l1_width)
        self.dec_l1_bn2 = nn.BatchNorm2d(int(self.l1_width/2))

        # L1 FC
        self.fc_mu = nn.Linear(int(self.l1_latent_size*self.l1_last_conv_size**2), l1_latent_size)
        self.fc_logvar= nn.Linear(int(self.l1_last_conv_size**2*self.l1_latent_size), l1_latent_size)
        self.fc_dec = nn.Linear(self.l1_latent_size, int(self.l1_last_conv_size**2*self.l1_latent_size))


    def forward(self, x, deterministic=False):
        l1_mu_logvar = self.encode(x)
        x = self.reparameterize_l1(l1_mu_logvar, deterministic=deterministic)    
        l2_mu_logvar = self.decode_l1(x)

        x = self.l2_conv_vae.reparameterize_vae_layer(l2_mu_logvar, deterministic=deterministic)
        recon_x = self.l2_conv_vae.decode_vae_layer(x)
        return recon_x, l1_mu_logvar, l2_mu_logvar

    def encode(self, x):
        """Returns tensor sized [bs, 2*latent_size]. First half is mu, 2nd is logvar"""
        x = self.elu(self.enc_l1_bn1(self.enc_l1_conv1(x)))
        x = self.elu(self.enc_l1_bn2(self.enc_l1_conv2(x)))
        x = self.elu(self.enc_l1_bn3(self.enc_l1_conv3(x)))
        x = self.elu(self.enc_l1_bn4(self.enc_l1_conv4(x)))
        x = self.enc_l1_conv5(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode_l1(self, x, deterministic=False):
        x = self.fc_dec(x)
        x = x.view((-1, self.l1_latent_size, self.l1_last_conv_size, self.l1_last_conv_size))
        x = self.elu(self.dec_l1_bn1(self.dec_l1_conv1(x)))
        x = self.elu(self.dec_l1_bn2(self.dec_l1_conv2(x)))
        l2_mu_logvar = self.dec_l1_conv3(x) # no bn because the next layer KLD will be computed
        return l2_mu_logvar
    
    def reparameterize_l1(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic:
            return mu
        else:
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)


class VAE(nn.Module):
    def __init__(self, latent_size=32, img_size=32, layer_sizes=[1, 32, 64, 128]):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.layer_sizes = layer_sizes
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.final_im_size = int(self.img_size/(2**(len(self.layer_sizes)-1)))
        self.linear_size = int(self.final_im_size**2*self.layer_sizes[-1])

        # Encoder
        self.encoder = nn.ModuleList()
        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.encoder.append(nn.Conv2d(layer_size, layer_sizes[i+1], kernel_size=4, stride=2, padding=1, bias=True))
            self.encoder.append(nn.BatchNorm2d(layer_sizes[i+1]))
            self.encoder.append(self.leakyrelu)
        self.encoder = nn.Sequential(*self.encoder)

        # FC
        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc1 = nn.Linear(self.latent_size, self.linear_size)

        # Decoder
        self.decoder = nn.ModuleList()
        for i, layer_size in enumerate(self.layer_sizes[::-1][:-1]):
            self.decoder.append(Interpolate(scale_factor=2, mode='nearest'))
            self.decoder.append(nn.ReplicationPad2d(1))
            self.decoder.append(nn.Conv2d(layer_size, self.layer_sizes[::-1][i+1], kernel_size=3, stride=1, bias=True))
            self.decoder.append(nn.BatchNorm2d(self.layer_sizes[::-1][i+1]))
            self.decoder.append(self.leakyrelu)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x, label=None, deterministic=False):  
        # label actually doesn't do anything, just for consistency with other cvae???
        mu_logvar = self.encode(x)
        z = self.reparameterize(mu_logvar, deterministic=deterministic)
        recon_x = self.decode(z)
        return recon_x, mu_logvar

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar
    
    def decode(self, x): # this is messy, really should put fc1 and decoder together into one thing.
        x = self.fc1(x)
        x = x.view((-1, self.layer_sizes[-1], self.final_im_size, self.final_im_size))
        x = self.decoder(x)
        return torch.sigmoid(x)

#     def reparameterize(self, mu, logvar=None, deterministic=False):
#         if deterministic:
#             return mu
#         else:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
    def reparameterize(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic:
            return mu
        else:
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=None)
        return x

#     def loss(self, output, x, KLD_weight=1, info=False):
#         recon_x, mu, logvar = output
#         BCE = F.mse_loss(recon_x, x, reduction='sum')
#         # see Appendix B from VAE paper:
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
#         loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
#         if info:
#             return loss, BCE, KLD
#         return loss



class VAE_ABS(nn.Module):
    """VAE based off https://arxiv.org/pdf/1805.09190v3.pdf
    ??? SHould we use global avg pooling and a 1x1 conv to get mu, sigma? Or even no 1x1, just normal conv.

    should the first fc in deconv be making the output batch*8*7*7???
    """
    def __init__(self, input_dim=1, output_dim=1, latent_size=8, img_size=32):
        super(VAE_ABS, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_size = int(16*(img_size/4)**2)

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv4 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(16, self.output_dim, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)

        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_bn3 = nn.BatchNorm2d(16)

        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)
        self.has_grad = False

    def forward(self, x, deterministic=False):
        mu_logvar = self.encode(x)
        z = self.reparameterize(mu_logvar, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu_logvar

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        if self.input_dim==1: return torch.sigmoid(x)
        else: return x

    def reparameterize(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic: # return mu 
            return mu
        else: # return mu + random
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)


class HVAE_B(BaseModel):
    """Hierarchical VAE with 2 latent spaces
    Assume input size 256- first vae downsamples to 16x16
    
    Make l1_vae, l2_vae so that the input size/stride and latentsize/input dim match
    
    SHOULD we add in L2 KLD twice for encoder and decoder?
    """
    
    def __init__(self, img_size, l1_vae, l2_vae, in_channels=1, l2_latent_size=16, l2_width=64, l2_input_size=32, l2_kernel_size=16, l2_stride=2,
                                    l1_latent_size=32, l1_width=64, l1_input_size=8):
        super(HVAE_2L, self).__init__()
        
        self.img_size = img_size
        self.l1_vae = l1_vae
        self.l2_conv_vae = ConvVAE2d(l2_vae, stride=l2_stride, padding=0, same=True, img_size=img_size)
        self.l1_last_conv_size = 1

    def forward(self, x, deterministic=False):
        l1_mu_logvar, l2_mu_logvar_enc = self.encode(x, deterministic=deterministic)
        x = self.reparameterize_l1(l1_mu_logvar, deterministic=deterministic)
        recon_x, l2_mu_logvar_dec = self.decode(x, deterministic=deterministic)
        return recon_x, l1_mu_logvar, l2_mu_logvar_enc, l2_mu_logvar_dec

    def encode(self, x, deterministic=True):
        """Returns tensor sized [bs, 2*latent_size]. First half is mu, 2nd is logvar"""
        l2_mu_logvar = self.l2_conv_vae.encode_vae_layer(x)
        x = self.l2_conv_vae.reparameterize_vae_layer(l2_mu_logvar, deterministic=deterministic)
        x = self.l1_vae.encode(x)
        x = x.view(x.size(0), -1)
        l1_mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return l1_mu_logvar, l2_mu_logvar

    def decode(self, x, deterministic=False):
        x = self.l1_vae.fc_dec(x)
        x = x.view((-1, self.l1_latent_size, self.l1_last_conv_size, self.l1_last_conv_size))
        l2_mu_logvar = self.l1_vae.decode(x)
        x = self.l2_conv_vae.reparameterize_vae_layer(l2_mu_logvar, deterministic=deterministic)
        x = self.l2_conv_vae.decode_vae_layer(x)
        return x, l2_mu_logvar
    
    def reparameterize_l1(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic:
            return mu
        else:
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

        
class NotHVAE(BaseModel):
    """HVAE_B but without any latent constraint.
       could manually construct this, buteasier to remove KLD+sampling from HVAE_B
       Here the l2 reparatrimize just extracts the means
       Hardcode the small VAE for 32x32 imgs, and the global VAE for 16x16
    """
    
    def __init__(self, img_size, in_channels=1, l2_latent_size=32, l1_latent_size=32, l2_stride=32):
        super(NotHVAE, self).__init__()
        
        self.img_size = img_size
        self.l1_vae = VAE_ABS(input_dim=l2_latent_size, output_dim=int(2*l2_latent_size), latent_size=l1_latent_size, img_size=8)
        self.l2_vae = VAE_ABS(input_dim=in_channels, output_dim=in_channels, latent_size=l2_latent_size, img_size=32)
        self.l2_conv_vae = ConvVAE2d(self.l2_vae, stride=l2_stride, padding=0, same=True, img_size=img_size)
        self.l1_latent_size = l1_latent_size

    def forward(self, x, deterministic=False):
        l1_mu_logvar = self.encode(x)
        x = self.reparameterize_l1(l1_mu_logvar, deterministic=deterministic)
        recon_x = self.decode(x)
        return recon_x, l1_mu_logvar

    def encode(self, x):
        """Returns tensor sized [bs, 2*latent_size]. First half is mu, 2nd is logvar"""
        l2_mu_logvar = self.l2_conv_vae.encode_vae_layer(x)
        x = self.l2_conv_vae.reparameterize_vae_layer(l2_mu_logvar, deterministic=True)
        l1_mu_logvar = self.l1_vae.encode(x)
        return l1_mu_logvar

    def decode(self, x):
        l2_mu_logvar = self.l1_vae.decode(x)
        x = self.l2_conv_vae.reparameterize_vae_layer(l2_mu_logvar, deterministic=True)
        x = self.l2_conv_vae.decode_vae_layer(x)
        return x
    
    def reparameterize_l1(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic:
            return mu
        else:
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)