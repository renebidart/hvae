import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


import pickle
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FilesDFImageDataset(Dataset):
    def __init__(self, files_df, base_path=None, transforms=None,
                 path_colname='path', label_colname='label', select_label=None, return_loc=False, bw=False):
        """ 
        Dataset based pandas dataframe of locations & labels 
        Optionally filter by labels or return locations
        
        files_df: Pandas Dataframe containing the class and path of an image
        base_path: the path to append to the filename
        transforms: result of transforms.Compose()
        select_label: if you only want one label returned
        return_loc: return location as well as the image and class
        path_colname: Name of column containing locations or filenames
        label_colname: Name of column containing labels
        """
        
        self.files = files_df
        self.base_path = base_path
        self.transforms = transforms
        self.path_colname = path_colname
        self.label_colname = label_colname
        self.return_loc = return_loc
        self.bw = bw
        if isinstance(select_label, int):
            self.files = self.files.loc[self.files[self.label_colname] == int(select_label)]
            print(f'Creating dataloader with only label {select_label}')

    def __getitem__(self, index):
        if self.base_path:
            loc = str(self.base_path) +'/'+ self.files[self.path_colname].iloc[index]
        else:
            loc = self.files[self.path_colname].iloc[index]
        if self.bw:
            img = Image.open(loc)
        else:
            try:
                img = Image.open(loc).convert('RGB')
            except Exception as e:
                print(e)
                print('loc', loc)
        if self.transforms is not None:
            img = self.transforms(img)
            
        label = self.files[self.label_colname].iloc[index]

        # return the right stuff:
        if self.return_loc:
            return img, label, loc
        else:
            return img, label

    def __len__(self):
        return len(self.files)

# Maybe better off adding transfroms to the config, so not so much duplicated?

def make_generators_MNIST(files_dict_loc, batch_size, num_workers, img_size=32, 
                             path_colname='path', label_colname='class', label=None, return_loc=False):
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        }

    data_transforms['val'] = data_transforms['train']

    datasets = {}
    dataloaders = {}

    datasets = {x: FilesDFImageDataset(files_dict[x], base_path=None, transforms=data_transforms[x], 
                                       path_colname=path_colname, label_colname=label_colname, select_label=label, return_loc=return_loc, bw=True)
                                        for x in list(data_transforms.keys())}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                                                    for x in list(data_transforms.keys())}
    return dataloaders


def make_gen_single_shape(files_dict_loc, batch_size, num_workers, img_size=32, 
                        path_colname='path', label_colname='shape_type', label=None, return_loc=False):
    with open(files_dict_loc, 'rb') as f:
        files_dict = pickle.load(f)
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.10,), (0.24,))
        ])
        }

    data_transforms['val'] = data_transforms['train']

    datasets = {}
    dataloaders = {}

    datasets = {x: FilesDFImageDataset(files_dict[x], base_path=None, transforms=data_transforms[x], 
                                       path_colname=path_colname, label_colname=label_colname, select_label=label, return_loc=return_loc, bw=True)
                                        for x in list(data_transforms.keys())}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
                                                    shuffle=True, num_workers=num_workers)
                                                    for x in list(data_transforms.keys())}
    return dataloaders


# def make_generators_DF_art(files_dict, base_path=None, batch_size=50, IM_SIZE=64, select_label=None, return_loc=False, 
#                              path_colname='path', label_colname='label', num_workers=4):
#     """
#     Uses standard cifar augmentation and nomalization.
#     """
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomCrop(int(IM_SIZE), padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(IM_SIZE),
#             transforms.ToTensor(),
#             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#         ]),
#     }
#     datasets = {}
#     dataloaders = {}

#     datasets = {x: FilesDFImageDataset(files_dict[x], base_path=base_path, transforms=data_transforms[x], 
#                                        path_colname=path_colname, label_colname=label_colname, 
#                                        select_label=select_label, return_loc=return_loc)
#                                         for x in list(data_transforms.keys())}

#     dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, 
#                                                     shuffle=True, num_workers=num_workers)
#                                                     for x in list(data_transforms.keys())}
#     return dataloaders

