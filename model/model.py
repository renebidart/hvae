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
