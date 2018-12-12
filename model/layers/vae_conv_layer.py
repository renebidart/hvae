import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _quadruple


class ConvVAE2d(nn.Module):
    """  ??? Fails with stride = 1
    Args:
         vae_small: vae with methods:
                     forward(x, deter) = encode(x)->reparameterize(mu, logvar, deter)-> decode(z)
                    and attribute:
                     vae_small.latent_size
                     vae_small.img_size

         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    
    def __init__(self, vae_small, stride=1, padding=0, same=True, img_size=64):
        super(ConvVAE2d, self).__init__()
        self.vae_small = vae_small
        self.vae_input_sz = self.vae_small.img_size

        self.k = _pair(self.vae_input_sz)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=self.k, stride=self.stride)
        self.padding = self._padding(size=(img_size, img_size))


    def _padding(self, x=None, size=None):
     # thanks https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598#file-median_pool-py-L5
        if self.same:
            if size:
                ih, iw = size
            else:
                ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding


    def encode_vae_layer(self, x):
        """ Encode an entire image into a 2d grid of latent variables
        Pad, reshape the input, apply the encoder convolutionally, reshape back
        x: img
        """
        (bs, ch, h, w) = x.size()
        x = F.pad(x, self._padding(x=x), mode='constant', value=0)

        # Select kernel_sz x kernel_sz chunks from the input spaced by stride
        # [bs, channels, h, w] -> [bs, channels, num_blocks1, num_blocks2, kernel_size, kernel_size]
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])

        # reshape to: [???, bs, ch kernel_sz, kernel_sz].
        # is this handlin the batches properly ???
        x = x.contiguous().view((-1,) + x.size()[:2] + x.size()[4:])

        # apply encoder convolutionally:
        x_encoded = list(map(self.vae_small.encode, torch.unbind(x, 0)))

        # reshape it back to correct way 
        x_encoded = torch.stack(x_encoded, 0)

        x_encoded = x_encoded.contiguous().view(bs, int(self.cvae_small.latent_size*2), int(h/self.stride[0]), int(w/self.stride[1]))
        return x_encoded


    def decode_vae_layer(self, x):
        """ Decode a 2d grid of latent variables
        Reshape the input, apply the decoder convolutionally, reshape back
        samp_2d: 2d layer of latent vars/samp from Normal: [bs, latent_size, h/stride, h/stride]
        """
        (bs, latent, h, w) = x.size()
        
        # reshape to: [???, bs, latent_size]
        x = x.contiguous().view((-1, bs, self.vae_small.latent_size))

        # apply decoder convolutionally:
        x_decoded = list(map(self.vae_small.decode, torch.unbind(x, 0)))        
        x_decoded = torch.stack(x_decoded, 0)

        # Convert to format for Fold: [bs, C×∏(kernel_size)-reshaped block, num_blocks]
        # padding returns 4 nums for top, bottom, .. so might be messed up here
        x_decoded = x_decoded.contiguous().view((bs, ) + (x_decoded.size()[2]*x_decoded.size()[3]*x_decoded.size()[4], ) + (-1,))
        
        # convert the block format back to 2d image:
        x_decoded = F.fold(x_decoded, output_size=(int(h*self.stride[0]), int(w*self.stride[1])), 
                           kernel_size=self.k, stride=self.stride, dilation=(1, 1), padding=self.padding[:2])
        return x_decoded


    def reparameterize_vae_layer(self, mu_logvar_2d, deterministic=False):
        """Add the randomness"""
        mu_2d = mu_logvar_2d[:, 0:int(mu_logvar_2d.size()[1]/2), :, :]

        if deterministic: # return mu 
            return mu_2d
        else: # return mu + random
            logvar_2d = mu_logvar_2d[:, int(mu_logvar_2d.size()[1]/2):]

            std = torch.exp(0.5*logvar_2d)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu_2d)
        

    def forward(self, x, deterministic=False):
        """Forward pass of vae layer. 
        Generally not useful because we want to decode or encode
        """
        mu_logvar = self.encode_vae_layer(x, c)
        z = self.reparameterize_vae_layer(mu_logvar, deterministic=deterministic)
        recon_x = self.decode_vae_layer(z, c)
        return recon_x, mu_logvar


    def loss(self, output, x, KLD_weight=1):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        recon_x, mu_logvar = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]

        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())      
        loss = Variable(BCE+KLD_weight*KLD, requires_grad=True) # seems need to do this to keep gradient
        return loss
#         return BCE + KLD_weight*KLD


class VAE_SMALL(nn.Module):
    """ use kernel of 16 or 32
    """
    def __init__(self, latent_size=10, img_size=16, in_channels=3):
        super(VAE_SMALL, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.last_conv_size =int((self.img_size/4)**2)*self.latent_size

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv3 = nn.Conv2d(64, self.latent_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(self.latent_size, 64, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(32, self.in_channels, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=True)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_bn2 = nn.BatchNorm2d(32)

        # FC
        self.fc_mu = nn.Linear(self.last_conv_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.last_conv_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.last_conv_size)


    def forward(self, x, deterministic=False):
        mu_logvar = self.encode(x)
        z = self.reparameterize(mu_logvar, deterministic=deterministic)
        recon_x = self.decode(z)
        return recon_x, mu_logvar

    def encode(self, x):
        """Returns tensor sized [bs, 2*latent_size]. First half is mu, 2nd is logvar"""
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_conv3(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, self.latent_size, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.dec_conv3(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]

        if deterministic: # return mu 
            return mu
        else: # return mu + random
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]

            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)