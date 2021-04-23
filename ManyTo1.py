#_*_ codidng:utf-8 _*_

import numpy as np
import os
import cv2

CT_max =   4218.1    ###max CT pixel intensity of all training CT images
CT_min =   -102.75  ### min CT pixel intensity of all training CT images
img_size = 256

input_path = "/data/wen/DCM_package/results/experiment_name/test_latest/images/" ###################################################
output_path = "/data/wen/DCM_package/results/"  ###################################################
FilenameList = []
for dirName, subdirList, fileList in os.walk(input_path):
    for filename in fileList:
        if "_fake_B.npy" in filename:
            FilenameList.append(os.path.join(dirName,filename))
print (len(FilenameList))
print (FilenameList)
FilenameList.sort()
print (FilenameList)

out=[]
for i in range(len(FilenameList)):
  a = np.load(FilenameList[i])
  out += [a]
  
out = np.array(out)
#print(out.max(),out.min())
out = (out*0.5+0.5)*(CT_max-CT_min)+CT_min
#print(out.max(),out.min())
out = out.reshape(len(FilenameList), img_size, img_size)
resized_out=np.zeros((len(out),512,512))
for i in range(len(out)):
    resized_out[i] = cv2.resize(out[i],(512,512))
np.save(output_path+"results.npy", resized_out)  #########################################################
print("Done!     ,output_shape:", resized_out.shape )