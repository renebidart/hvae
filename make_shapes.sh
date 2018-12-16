python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size --im_size 32 --rand_size
python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_loc --im_size 32 --rand_size --rand_loc 
python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_loc_fill --im_size 32 --rand_size --rand_loc --rand_fill

python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_g8 --im_size 32 --rand_size --grid_width 8
python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_loc_g8 --im_size 32 --rand_size --rand_loc --grid_width 8
python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_loc_fill_g8 --im_size 32 --rand_size --rand_loc --rand_fill --grid_width 8

python make_shape_data.py --PATH /media/rene/data/hvae/shapes/sz32_rand_size_loc_g8_struc2 --im_size 32 --rand_size --rand_loc --grid_width 8 --structured_grid





