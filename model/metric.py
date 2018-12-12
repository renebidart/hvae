import torch
import torch.nn.functional as F

"""
Return scalars of the metrics we care about to log
So much duplicated computation with loss, but don't want to change interface

??? How to add variable inputs to the loss and metrics?
"""


def BCE_L2(output, target):
    recon_x, l1_mu_logvar, l2_mu_logvar = output
    BCE = F.mse_loss(recon_x, target, reduction='sum')
    return BCE.item()

def KLD_1d_L1(output, target):
    recon_x, l1_mu_logvar, l2_mu_logvar = output
    l1_mu = l1_mu_logvar[:, 0:int(l1_mu_logvar.size()[1]/2)]
    l1_logvar = l1_mu_logvar[:, int(l1_mu_logvar.size()[1]/2):]
    L1_KLD = -0.5 * torch.sum(1 + 2 * l1_logvar - l1_mu.pow(2) - (2 * l1_logvar).exp())
    return L1_KLD.item()

def KLD_2d_L2(output, target):
    recon_x, l1_mu_logvar, l2_mu_logvar = output
    l2_mu = l2_mu_logvar[:, 0:int(l2_mu_logvar.size()[1]/2)]
    l2_logvar = l2_mu_logvar[:, int(l2_mu_logvar.size()[1]/2):]
    L2_KLD = -0.5 * torch.sum(1 + 2 * l2_logvar - l2_mu.pow(2) - (2 * l2_logvar).exp())
    return L2_KLD.item()

def full_hvae_loss(output, target, L1_KLD_weight=1, L2_KLD_weight=1):
    """loss is BCE + L1_KLD + L2_KLD. target is original x"""
    recon_x, l1_mu_logvar, l2_mu_logvar = output

    L1_KLD = KLD_1d_L1(output, target)
    L2_KLD = KLD_2d_L2(output, target)
    BCE = F.mse_loss(recon_x, target, reduction='sum').item()

    loss = BCE + L1_KLD_weight*L1_KLD + L2_KLD_weight*L2_KLD
    return loss



### Noramal VAE
def BCE(output, target):
    recon_x, mu_logvar = output
    BCE = F.mse_loss(recon_x, target, reduction='sum')
    return BCE.item()

def KLD(output, target):
    recon_x, mu_logvar = output
    mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
    logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
    KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
    return KLD.item()

# def vae_loss(output, target, KLD_weight=1):
#     """loss is BCE + KLD. target is original x"""
#     recon_x, mu_logvar = output

#     KLD = KLD(output, target)
#     BCE = F.mse_loss(recon_x, target, reduction='sum').item()

#     loss = BCE + KLD_weight*KLD
#     return loss

