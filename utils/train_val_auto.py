# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# from pathlib import Path

# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients
# import torchvision.transforms as T
# from torch.optim import lr_scheduler
# import torch.optim as optim
# import time
# import shutil


# def train_epoch_auto(epoch, lr, loader, model, optimizer, criterion, device):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()

#     model.train()
#     end = time.time()
    
#     for i, (inputs, label) in enumerate(loader):
#         data_time.update(time.time() - end)
#         inputs, label = inputs.to(device), label.type(torch.LongTensor).to(device)

#         output = model(inputs, label)
#         loss = criterion(output, inputs)

#         losses.update(loss, inputs.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#     print(f'{epoch}:    Loss ({losses.avg:.4f})\t'
#            f'Time ({batch_time.avg:.3f})\t Data Time ({data_time.avg:.3f})')
#     return losses.avg

# def validate_epoch_auto(loader, model, criterion, device):
#     batch_time = AverageMeter()
#     losses = AverageMeter()

#     model.eval()

#     with torch.no_grad():
#         end = time.time()
#         for i, (inputs, label) in enumerate(loader):
#             inputs, label = inputs.to(device), label.type(torch.LongTensor).to(device)
#             output = model(inputs, label)
#             loss = criterion(output, inputs)

#             losses.update(loss.item(), inputs.size(0))
#             batch_time.update(time.time() - end)
#             end = time.time()
            
#     print(f'VALID: Loss ({losses.avg:.4f})\t Time ({batch_time.avg:.3f})\t')
#     return losses.avg

# def save_checkpoint(state, is_best, model_name, PATH):
#     save_path = str(PATH)+'/'+str(model_name)+'_ckpnt.pth.tar'
#     torch.save(state, save_path)
#     if is_best:
#         best_path = str(PATH)+'/'+str(model_name)+'_model_best.pth.tar'
#         shutil.copyfile(save_path, best_path)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count