import torch
import torch.nn.functional as F
from torch.autograd import Variable

"""
Both loss and metrics should take arguments in form (output, target)
The only difference between the two is loss is used for backprop, 
metrics is something we care about to look at.

Arguments must be in form (model_forward_output, targets(label, img, etc))
"""

def make_hvae_loss(L1_KLD_weight=1, L2_KLD_weight=1):
    def hvae_loss(output, target, L1_KLD_weight=L2_KLD_weight, L2_KLD_weight=L2_KLD_weight):
        """loss is BCE + L1_KLD + L2_KLD. target is original x"""
        recon_x, l1_mu_logvar, l2_mu_logvar = output
        l1_mu = l1_mu_logvar[:, 0:int(l1_mu_logvar.size()[1]/2)]
        l1_logvar = l1_mu_logvar[:, int(l1_mu_logvar.size()[1]/2):]
        L1_KLD = -0.5 * torch.sum(1 + 2 * l1_logvar - l1_mu.pow(2) - (2 * l1_logvar).exp())
        L2_KLD = loss_KLD_2d(l2_mu_logvar)
        BCE = F.mse_loss(recon_x, target, reduction='sum')
        loss = BCE + L1_KLD_weight*L1_KLD + L2_KLD_weight*L2_KLD
        return loss
    return hvae_loss


def make_hvae_enc_dec_loss(L1_KLD_weight=1, L2_KLD_weight=1):
    def hvae_enc_dec_loss(output, target, L1_KLD_weight=L2_KLD_weight, L2_KLD_weight=L2_KLD_weight):
        """loss is BCE + L1_KLD + L2_KLD. target is original x"""
        recon_x, l1_mu_logvar, l2_mu_logvar_enc, l2_mu_logvar_dec = output
        l1_mu = l1_mu_logvar[:, 0:int(l1_mu_logvar.size()[1]/2)]
        l1_logvar = l1_mu_logvar[:, int(l1_mu_logvar.size()[1]/2):]
        L1_KLD = -0.5 * torch.sum(1 + 2 * l1_logvar - l1_mu.pow(2) - (2 * l1_logvar).exp())
        L2_KLD_enc = loss_KLD_2d(l2_mu_logvar_enc)
        L2_KLD_dec = loss_KLD_2d(l2_mu_logvar_dec)
        BCE = F.mse_loss(recon_x, target, reduction='sum')
        loss = BCE + L1_KLD_weight*L1_KLD + L2_KLD_weight*L2_KLD_enc + L2_KLD_weight*L2_KLD_dec
        return loss
    return hvae_enc_dec_loss


def loss_KLD_2d(mu_logvar):
    mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
    logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
    KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
#     KLD = Variable(KLD, requires_grad=True) # seems need to do this to keep gradient
    return KLD


def make_vae_loss(KLD_weight=1):
    def vae_loss(output, target, KLD_weight=KLD_weight):
        """loss is BCE + KLD. target is original x"""
        recon_x, mu_logvar  = output
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        BCE = F.mse_loss(recon_x, target, reduction='sum')
#         loss = Variable(BCE + KLD_weight*KLD, requires_grad=True)
        loss = BCE + KLD_weight*KLD
        return loss
    return vae_loss


