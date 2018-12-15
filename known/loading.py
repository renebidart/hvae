"""
???
Delete all the loading for models that are never used (all except preact)
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from vae import VAE
# from models.vae_general import VAE_ABS
# from models.cvae import CVAE_ART
# from models.vae_conv_layer import ConvVAE2d, CVAE_SMALL


def vae_from_args(args):
    if (args.net_type == 'cvae'):
        net = CVAE(num_labels=args.num_labels, latent_size=args.latent_size, img_size=args.IM_SIZE,
                    layer_sizes=args.layer_sizes)
        sizes_str =  "_".join(str(x) for x in args.layer_sizes)
        file_name = 'CVAE-'+str(sizes_str)+'-'+str(args.latent_size)+'-'+str(args.dataset)+'-'+str(args.num_labels)
    elif (args.net_type == 'vae'):
        net = VAE(latent_size=args.latent_size, img_size=args.IM_SIZE, layer_sizes=args.layer_sizes)
        sizes_str =  "_".join(str(x) for x in args.layer_sizes)
        file_name = 'VAE-'+str(sizes_str)+'-'+str(args.latent_size)+'-'+str(args.dataset)
    elif (args.net_type == 'VAE_ABS'):
        net = VAE_ABS(latent_size=args.latent_size, img_size=args.IM_SIZE)
        file_name = 'VAE_ABS-'+str(args.latent_size)+'-'+str(args.dataset)

    elif (args.net_type == 'FEAT_VAE_MNIST'):
        net = FEAT_VAE_MNIST(classifier_model=load_net(args.encoding_model_loc).to(args.device),
                             num_features=args.num_features,
                             latent_size=args.latent_size)
        file_name = 'FEAT_VAE_MNIST-'+str(args.latent_size)+'-'+str(args.num_features)+'-'+str(args.dataset)

    elif (args.net_type == 'CVAE_ABS'):
        net = CVAE_ABS(latent_size=args.latent_size, 
                       img_size=args.IM_SIZE,
                       num_labels=args.num_labels
                       )
        file_name = 'CVAE_ABS-'+str(args.latent_size)+'-'+str(args.dataset)

    elif (args.net_type == 'CVAE_ART'):
        net = CVAE_ART(latent_size=args.latent_size, 
                       img_size=args.IM_SIZE,
                       num_labels=args.num_labels)
        file_name = 'CVAE_ART-'+str(args.latent_size)+'-'+str(args.IM_SIZE)

    elif (args.net_type == 'ConvVAE2d'):
        if (args.small_net_type == 'CVAE_SMALL'):
            small_net = CVAE_SMALL(latent_size=args.latent_size_small_vae, 
                                   img_size=args.cvae_input_sz,
                                   num_labels=args.num_labels)

        net = ConvVAE2d(cvae_small=small_net, 
                        cvae_input_sz=args.cvae_input_sz,
                        stride = args.stride,
                        img_size=args.IM_SIZE)
        file_name = 'ConvVAE2d-'+str(args.IM_SIZE)+'-'+str(args.stride)+'-'+str(args.cvae_input_sz)+'-'+str(args.latent_size_small_vae)


    else:
        print('Error : Wrong net type')
        sys.exit(0)
    return net, file_name


def load_net(model_loc, args=None):
    model_file = Path(model_loc).name
    model_name = model_file.split('-')[0]

    if (model_name == 'CVAE'):
        model = CVAE(num_labels=int(model_file.split('-')[4].split('_')[0]),
                     latent_size=int(model_file.split('-')[2]), 
                     img_size=32,
                     layer_sizes=[int(i) for i in model_file.split('-')[1].split('_')])
    elif (model_name == 'VAE'):
        model = VAE(latent_size=int(model_file.split('-')[2]),
                     img_size=32,
                     layer_sizes=[int(i) for i in model_file.split('-')[1].split('_')])
    elif (model_name == 'VAE_ABS'):
        model = VAE_ABS(latent_size=8, img_size=28)

    elif (model_name == 'CVAE_ABS'):
        model = CVAE_ABS(latent_size=8, img_size=28)

    elif (model_name == 'SimpleNetMNIST'):
        model = SimpleNetMNIST(num_filters=int(model_file.split('-')[1].split('_')[0]))

    elif (model_name == 'TopkNetMNIST'):
        model = TopkNetMNIST(num_filters=int(model_file.split('-')[1].split('_')[0]), 
                        topk_num=int(model_file.split('-')[2].split('_')[0]))

    elif (model_name == 'FEAT_VAE_MNIST'):
        model = FEAT_VAE_MNIST(classifier_model=load_net(args.encoding_model_loc).to(args.device),
                             num_features=int(model_file.split('-')[2].split('_')[0]),
                             latent_size=int(model_file.split('-')[1].split('_')[0]))

    elif (model_name == 'ConvVAE2d'):
        latent_size_small_vae = int(model_file.split('-')[4].split('_')[0])
        cvae_input_sz = int(model_file.split('-')[3])
        stride = int(model_file.split('-')[2])
        IM_SIZE = int(model_file.split('-')[1])

        small_net = CVAE_SMALL(latent_size=latent_size_small_vae, 
                       img_size=cvae_input_sz,
                       num_labels=11)

        model = ConvVAE2d(cvae_small=small_net, 
                        cvae_input_sz=cvae_input_sz,
                        stride = stride,
                        img_size=IM_SIZE)

    else:
        print(f'Error : {model_file} not found')
        sys.exit(0)
    model.load_state_dict(torch.load(model_loc)['state_dict'])
    return model


# # Return network & a unique file name
# def net_from_args(args, num_classes, IM_SIZE):
#     if (args.net_type == 'vggnet'):
#         net = VGG(args.depth, num_classes, IM_SIZE)
#         file_name = 'vgg-'+str(args.depth)
#     elif (args.net_type == 'resnet'):
#         net = ResNet(args.depth, num_classes, IM_SIZE)
#         file_name = 'resnet-'+str(args.depth)
#     elif (args.net_type == 'preact_resnet'):
#         if args.frac != 1:
#             net = PResNetReg(args.depth, args.frac, args.groups, num_classes)
#             file_name = 'preact_resnet-'+str(args.depth)+'-'+str(args.frac)+'-'+str(args.groups)
#         else:
#             net = PreActResNet(args.depth, num_classes)
#             file_name = 'preact_resnet-'+str(args.depth)
#     elif (args.net_type == 'wide-resnet'):
#         net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, IM_SIZE)
#         file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)

#     elif (args.net_type == 'PResNetRegNoRelU'):
#         net = PResNetRegNoRelU(args.depth, args.frac, args.groups, num_classes)
#         file_name = 'PResNetRegNoRelU-'+str(args.depth)+'-'+str(args.frac)+'-'+str(args.groups)

#     elif (args.net_type == 'TestNetNotResNet'):
#         net = TestNetNotResNet()
#         file_name = 'TestNetNotResNet'
#     elif (args.net_type == 'TestNetMostlyResNet'):
#         net = TestNetMostlyResNet()
#         file_name = 'TestNetMostlyResNet'
#     elif (args.net_type == 'TestNetResnetTopK'):
#         net = TestNetResnetTopK()
#         file_name = 'TestNetResnetTopK'
#     elif (args.net_type == 'TestNetResnetTopKEverywhere'):
#         net = TestNetResnetTopKEverywhere()
#         file_name = 'TestNetResnetTopKEverywhere'
#     elif (args.net_type == 'TestNetResnetTopK_act'):
#         net = TestNetResnetTopK_act()
#         file_name = 'TestNetResnetTopK_act'
# ##### Nets for features for Generative classifiers
#     elif (args.net_type == 'SimpleNetMNIST'):
#         net = SimpleNetMNIST(args.num_filters)
#         file_name = 'SimpleNetMNIST-'+str(args.num_filters)
#     elif (args.net_type == 'TopkNetMNIST'):
#         net = TopkNetMNIST(num_filters=args.num_filters, topk_num=args.topk_num)
#         file_name = 'TopkNetMNIST-'+str(args.num_filters)+'-'+str(args.topk_num)

#     else:
#         # print('Error : Wrong net type')
#         sys.exit(0)
#     return net, file_name

