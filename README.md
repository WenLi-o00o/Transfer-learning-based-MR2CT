This is the code for published paper: "Synthesizing CT images from MR images with deep learning: model generalization for different datasets through transfer learning". 


The code was modified based on Jun-Yan Zhu and Taesung Park's work with pytorch, original link: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


In addition, if you want to use the synthetic CT for clinical dose calculation or other evaluation, we also provided the code to convert the synthetic npy format CT to Eclipse surpported DCM format, hope that will help.


You can modify the code to adapt your research work, or simply preparing your data and running according to below steps: 

1. All training MR image data should be put in "/datasets/DcmData/trainA/" folder;
    Your training CT image data should be put in "/datasets/DcmData/trainB/" folder;
    Testing MR image data should be put in "/datasets/DcmData/testA/" folder;
    Testing CT image data should be put in "/datasets/DcmData/testB/" folder.

2.1 The mr_max, mr_min, ct_max, ct_min of your dataset should be specified in MR2CT.sh before runing the code;
2.2 In ManyTo1.py, you need to specify the ct_max and ct_min again, the value should be the same as step 2.1. you also need to modify the path to your computer path.

3. In config.json, you need to change the path to your computer path.

4. This code can only test one patient each time at present, please remember to update corresponding testA and testB data each time for new patient, and export the generated results from "../results/generated_dicom/".

 
 
If you find this code is helpful, please cite:

@article{li2021synthesizing,
  title={Synthesizing CT images from MR images with deep learning: model generalization for different datasets through transfer learning},
  author={Li, Wen and Kazemifar, Samaneh and Bai, Ti and Nguyen, Dan and Weng, Yaochung and Li, Yafen and Xia, Jun and Xiong, Jing and Xie, Yaoqin and Owrangi, Amir and others},
  journal={Biomedical Physics \& Engineering Express},
  volume={7},
  number={2},
  pages={025020},
  year={2021},
  publisher={IOP Publishing}
}
