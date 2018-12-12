import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _quadruple


class ConvVAE2d(nn.Module):
    """  ??? Fails with stride = 1
    Args:
         cvae_small: cvae with methods:
                     forward(x, c, deter) = encode(x, c)->reparameterize(mu, logvar, deter)-> decode(z, c)
                     ???loss???

         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, cvae_small, cvae_input_sz=16, stride=1, padding=0, same=True, img_size=64):
        super(ConvVAE2d, self).__init__()
        self.cvae_small = cvae_small

        self.k = _pair(cvae_input_sz)
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


    def encode_vae_layer(self, x, c):
        """ Reshape the input, apply the encoder convolutionally, reshape back
        x: img
        c: class for the cvae layer: [bs, class]
        """
        (bs, ch, h, w) = x.size()
        x = F.pad(x, self._padding(x=x), mode='constant', value=0)

        # Select kernel_sz x kernel_sz chunks from the input spaced by stride
        # [bs, channels, h. w] -> [bs, channels, num_blocks1, num_blocks2, kernel_size, kernel_size]
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])

        # reshape to: [???, bs, ch kernel_sz, kernel_sz].
        # how to properly handle batches with convolutions?
        x = x.contiguous().view((-1,) + x.size()[:2] + x.size()[4:])

        # make the same numer of c as is sections to apply vae to
        c = c.repeat(x.size()[0], 1)

        # apply encoder convolutionally:
        x_encoded = list(map(self.cvae_small.encode, torch.unbind(x, 0), torch.unbind(c, 0)))

        # reshape it back to correct way 
        x_encoded = torch.stack(x_encoded, 0)
        x_encoded = x_encoded.contiguous().view(bs, int(self.cvae_small.latent_size*2), int(h/self.stride[0]), int(w/self.stride[1]))
        return x_encoded


    def decode_vae_layer(self, x, c):
        """ Reshape the input, apply the decoder convolutionally, reshape back
        samp_2d: 2d layer of latent vars/samp from Normal: [bs, latent_size, h/stride, h/stride]
        c: class for the cvae layer: [bs, class]
        """
        (bs, latent, h, w) = x.size()

        # reshape to: [???, bs, latent_size]
        x = x.contiguous().view((-1, bs, self.cvae_small.latent_size))

        # make the same numer of c as is sections to apply vae to
        c = c.repeat(x.size()[0], 1)

        # apply decoder convolutionally:
        x_decoded = list(map(self.cvae_small.decode, torch.unbind(x, 0), torch.unbind(c, 0)))
        x_decoded = torch.stack(x_decoded, 0)

        # Convert to format for Fold: [bs, C×∏(kernel_size)-reshaped block, num_blocks]
        # padding returns 4 nums for top, bottom, .. so might be messed up here
        x_decoded = x_decoded.contiguous().view((bs, ) + (x_decoded.size()[2]*x_decoded.size()[3]*x_decoded.size()[4], ) + (-1,))
        
        # convert the block format back to 2d image:
        x_decoded = F.fold(x_decoded, output_size=(int(h*self.stride[0]), int(w*self.stride[1])), 
                           kernel_size=self.k, stride=self.stride, dilation=(1, 1), padding=self.padding[:2])
        return x_decoded


    def reparameterize_vae_layer(self, mu_logvar_2d, deterministic=False):
        mu_2d = mu_logvar_2d[:, 0:int(mu_logvar_2d.size()[1]/2), :, :]

        if deterministic: # return mu 
            return mu_2d
        else: # return mu + random
            logvar_2d = mu_logvar_2d[:, int(mu_logvar_2d.size()[1]/2):]

            std = torch.exp(0.5*logvar_2d)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu_2d)
        

    def forward(self, x, c, deterministic=False):
        mu_logvar = self.encode_vae_layer(x, c)
        z = self.reparameterize_vae_layer(mu_logvar, deterministic=deterministic)
        recon_x = self.decode_vae_layer(z, c)
        return recon_x, mu_logvar


    def loss(self, output, x, KLD_weight=1, info=False):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        recon_x, mu_logvar = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]

        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        # loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
        # if info:
        #     return loss, BCE, KLD
        # return loss
        # print('BCE', BCE)
        # print('KLD', KLD)
        return BCE + KLD_weight*KLD


class CVAE_SMALL(nn.Module):
    """ use kernel of 16
    """
    def __init__(self, latent_size=10, img_size=16, num_labels=11):
        super(CVAE_SMALL, self).__init__()
        self.num_labels = num_labels
        self.latent_size = latent_size
        self.img_size = img_size
        self.last_conv_size =int((self.img_size/4)**2)*self.latent_size

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv3 = nn.Conv2d(64, self.latent_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(self.latent_size, 64, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(64)
        self.dec_bn2 = nn.BatchNorm2d(32)

        # FC
        self.fc_mu = nn.Linear(self.last_conv_size+self.num_labels, self.latent_size)
        self.fc_logvar= nn.Linear(self.last_conv_size+self.num_labels, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size+self.num_labels, self.last_conv_size)


    def forward(self, x, c, deterministic=False):
        mu_logvar = self.encode(x, c)
        z = self.reparameterize(mu_logvar, deterministic=deterministic)
        recon_x = self.decode(z, c)
        return recon_x, mu_logvar

    def encode(self, x, c):
        """Returns tensor sized [bs, 2*latent_size]. First half is mu, 2nd is logvar"""
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode(self, x, c):
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
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

    def loss(self, output, x, KLD_weight=1, info=False):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        recon_x, mu_logvar = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]

        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        # loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
        # if info:
        #     return loss, BCE, KLD
        # return loss
        return BCE + KLD

    def to_one_hot(self, y):
        y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size()[0], self.num_labels).type(y.type())
        y_onehot.scatter_(1, y, 1)
        return y_onehot
