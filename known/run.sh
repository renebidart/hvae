python train_vae.py --files_df_loc /media/rene/data/MNIST/files_dict.pkl --SAVE_PATH /media/rene/data/hvae/mnist/models --dataset MNIST --net_type vae --layer_sizes 1 32 64 128 --latent_size 32 --batch_size 256 --epochs 150 --device "cuda:0" --lr .001 --epochs 300


python train_vae.py --files_df_loc /media/rene/data/learn-lr/MNIST/files_df.pkl --SAVE_PATH /media/rene/data/hvae/mnist/models_tmp --dataset MNIST --net_type vae --layer_sizes 1 32 64 128 --latent_size 32 --batch_size 256 --epochs 150 --device "cuda:0" --lr .001 --epochs 300
