import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from random import randint
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--PATH', type=str)
parser.add_argument('--train_imgs', type=int, default=60000)
parser.add_argument('--val_imgs', type=int, default=10000)
parser.add_argument('--im_size', type=int, default=32)
parser.add_argument('--rand_size', dest='rand_size', action='store_true')
parser.add_argument('--rand_loc', dest='rand_loc', action='store_true')
parser.add_argument('--rand_fill', dest='rand_fill', action='store_true')
parser.add_argument('--rand_outline', dest='rand_outline', action='store_true')
parser.add_argument('--grid_width', type=int, default=1)
args = parser.parse_args()


def make_shape(im_size, shape_type, rand_size, rand_loc, rand_fill, rand_outline):
    """
    im_size: images are square, should be int >=16
    shape_type:
        0: square
        1: circle
        2: rectangle (h=2*w)
        3: ellipse (h=2*w)
        4: right traingle (h = w)
    rand_fill: Random Uniform (1, 128)
    rand_outline: Random Uniform (1, 128)
    rand_size: Random shape size between 25% and 75% of the image
    rand_loc: Random Uniform over domain, excepting the edges
    grid_width: Generate a grid of shapes rather than a single one
    """
    kwargs = {}
    
    if rand_fill:
        kwargs['fill'] = randint(1, 128)
    else: 
        kwargs['fill'] = 1
    if rand_outline:
        kwargs['outline'] = randint(1, 128)
    else: 
        kwargs['outline'] = kwargs['fill']
        
    if rand_size:
        height = randint(im_size/8, im_size*3/8)*2
        if shape_type in [0, 1, 4]:
            width=height
        elif shape_type in [2, 3]:
            width=height/2
    
    if rand_loc:
        top_l = (randint(1, int(im_size-width-2)), (randint(1, int(im_size-height-2))))
        bottom_r = (top_l[0] + width, top_l[1] + height)
    else:
        top_l = (int((im_size-width)/2), int((im_size-height)/2))
        bottom_r = (top_l[0] + width, top_l[1] + height)
        
    im = Image.fromarray(np.uint8(np.full((im_size,im_size), 255)))
    draw = ImageDraw.Draw(im)

    if shape_type in [0, 2]:
        kwargs['xy'] = (top_l, bottom_r)
        draw.rectangle(**kwargs)
    elif shape_type in [1, 3]:
        kwargs['xy'] = (top_l, bottom_r)
        draw.ellipse(**kwargs)
    elif shape_type in [4]:
        bottom_l = (top_l[0], top_l[1] + height)
        kwargs['xy'] = (top_l, bottom_l, bottom_r)
        draw.polygon(**kwargs)
    else:
        print(f'Invalid shape type: {shape_type}')
    del draw
    return im, shape_type, kwargs['fill'], kwargs['outline'], kwargs['xy']
    

def make_shape_grid(im_size, rand_size, rand_loc, rand_fill, rand_outline, grid_width):
    """Generate a grid of random shapes"""
    im = np.full((grid_width*im_size, grid_width*im_size), 255)
    for i in range(grid_width*grid_width):
        shape_type = randint(0, 4)
        single_shape, shape_type, fill, outline, xy = make_shape(im_size, shape_type, rand_size, rand_loc, rand_fill, rand_outline)
        im[(i//grid_width)*im_size: (1+i//grid_width)*im_size , (i%grid_width)*im_size: (1+i%grid_width)*im_size] = single_shape
    return Image.fromarray(np.uint8(im))

    
def make_shape_dataset(PATH, train_imgs, val_imgs, im_size, rand_size, rand_loc, rand_fill, rand_outline, grid_width):
    """Create a dataset of 5 shapes:  square, rectangle, circle, ellipse, right traingle
    
    Options:
    PATH: Where to save everything
    train_imgs: 
    val_imgs:
    im_size: size of output img
    rand_fill: Random Uniform (1, 128)
    rand_outline: Random Uniform (1, 128)
    rand_size: Random shape size between 25% and 75% of the image
    rand_loc: Random Uniform over domain, excepting the edges
    """
    
    # paths to store iamges
    PATH = Path(PATH)
    TRAIN_PATH = PATH / 'train'
    TRAIN_PATH.mkdir(parents=True, exist_ok=True)
    VAL_PATH = PATH / 'val'
    VAL_PATH.mkdir(parents=True, exist_ok=True)
    
    files = {}
    files['train'] = pd.DataFrame()
    files['val'] = pd.DataFrame()
    
    for i in range(train_imgs):
        loc = str(TRAIN_PATH) + '/train_'+str(i)+'.png'
        if grid_width>1: # make the grid of images
            im = make_shape_grid(im_size, rand_size, rand_loc, rand_fill, rand_outline, grid_width)
            files['train'] = files['train'].append({'path': loc}, ignore_index=True)
        else: # generate a single image and info
            shape_type = randint(0, 4)
            im, shape_type, fill, outline, xy = make_shape(im_size, shape_type, rand_size, rand_loc, rand_fill, rand_outline)
            files['train'] = files['train'].append({'path': loc, 'shape_type': shape_type,
                                            'fill': fill, 'outline': outline, 'xy': xy,
                                           }, ignore_index=True)
        im.save(loc)

    for i in range(train_imgs):
        loc = str(VAL_PATH) + '/val_'+str(i)+'.png'
        if grid_width>1: # make the grid of images
            im = make_shape_grid(im_size, rand_size, rand_loc, rand_fill, rand_outline, grid_width)
            files['val'] = files['val'].append({'path': loc}, ignore_index=True)
        else: # generate a single image and info
            shape_type = randint(0, 4)
            im, shape_type, fill, outline, xy = make_shape(im_size, shape_type, rand_size, rand_loc, rand_fill, rand_outline)
            files['val'] = files['val'].append({'path': loc, 'shape_type': shape_type,
                                        'fill': fill, 'outline': outline, 'xy': xy,
                                       }, ignore_index=True)
        im.save(loc)
        
    with open(str(PATH)+'/files_dict.pkl', 'wb') as f:
        pickle.dump(files, f, pickle.HIGHEST_PROTOCOL)

    # Make a 10% sample:
    sample_df ={}
    sample_df['train'] = files['train'].sample(frac=.1)
    sample_df['val'] = files['val'].sample(frac=.1)

    with open(str(PATH)+'/sample_dict.pkl', 'wb') as f:
        pickle.dump(sample_df, f, pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
    make_shape_dataset(args.PATH, args.train_imgs, args.val_imgs, args.im_size, 
                       args.rand_size, args.rand_loc, args.rand_fill, args.rand_outline,
                       args.grid_width)