# python train.py --device 0 -c configs/vae_001_rand_size.json
# python train.py --device 0 -c configs/vae_001_rand_size_loc.json
# python train.py --device 0 -c configs/vae_001_rand_size_loc_fill.json

# python train.py --device 1 -c configs/NotHVAE_L32_L32_rand_size_loc_g8.json
# python train.py --device 0 -c configs/NotHVAE_L32_L32_rand_size_loc_fill_g8.json


python train.py --device 1 -c configs/HVAE_pre_L64_L32_rand_size_loc_g8_struc.json
python train.py --device 1 -c configs/HVAE_pre_L64_L32_rand_size_loc_g8_struc2.json