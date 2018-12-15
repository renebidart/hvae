import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    """VAE
    Add one-hot encoded class label as input to the last FC layer in encoder
    """
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, x): # this is messy, really should put fc1 and decoder together into one thing.
        x = self.fc1(x)
        x = x.view((-1, self.layer_sizes[-1], self.final_im_size, self.final_im_size))
        x = self.decoder(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar=None, deterministic=False):
        if deterministic:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    # def loss(self, output, x, KLD_weight=1, info=False):
    #     """Compute the loss between the encoded features, and the generated features, not the image"""
    #     recon_x, mu, logvar = output
    #     print('------------------------------------------')
    #     print('torch.mean(x)', torch.mean(x))
    #     print('torch.max(x)', torch.max(x))
    #     print('torch.min(x)', torch.min(x))

    #     print('torch.mean(recon_x)', torch.mean(recon_x))
    #     print('torch.max(recon_x)', torch.max(recon_x))
    #     print('torch.min(recon_x)', torch.min(recon_x))

    #     BCE = F.mse_loss(recon_x, x, reduction='sum')
    #     # see Appendix B from VAE paper:
    #     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #     # https://arxiv.org/abs/1312.6114
    #     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
    #     loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
    #     print('KLD:', KLD)
    #     print('BCE:', BCE)

    #     if info:
    #         return loss, BCE, KLD
    #     return loss

    def loss(self, output, x, KLD_weight=1, info=False):
        recon_x, mu, logvar = output
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
        if info:
            return loss, BCE, KLD
        return loss

# what the fuck
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=None)
        return x