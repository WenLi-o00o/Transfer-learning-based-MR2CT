python train.py --dataroot /datasets/DcmData/ --dataset_mode unalignedtrain --mr_max 1769.8 --mr_min -129.32 --ct_max 4218.1 --ct_min -102.75; test.py --dataroot /datasets/DcmData/ --dataset_mode unalignedtest --mr_max 1769.8 --mr_min -129.32 --ct_max 4218.1 --ct_min -102.75; python ManyTo1.py; python Npy2Dcm.py
