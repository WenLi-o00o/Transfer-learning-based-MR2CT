# -*- coding: utf-8 -*-
"""
Author: Anjali Balagopal, University of Texas Southwestern Medical center
Created on 09-17-2018
Modified by Wen Li
Modified on 06-04-2020
"""

import os
import numpy as np
import pydicom as dicom
from time import strftime, localtime
import json

with open('config.json', 'r') as f:
    cfg = json.load(f)
IDs = cfg['ID_list']
dicom_dir_new = cfg['dicom_dir_new']
dicom_dir = cfg['dicom_dir']
CT_dir = cfg['CT_dir']
org_root = cfg['organizationroot']

for n in range(len(IDs)):
    patientID = IDs[n]
    print(patientID)
    oldDicompath = dicom_dir
    print(oldDicompath)
    newDicompath = dicom_dir_new
    edited_ct = np.load(CT_dir + 'results.npy')
    print(edited_ct.shape)
    DCMFiles = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(oldDicompath):
        for filename in fileList:
            if filename == ".DS_Store":
                continue
            DCMFiles.append(os.path.join(dirName,filename))
    print (len(DCMFiles))
    #print (DCMFiles)
    DCMFiles.sort()
    print (DCMFiles)
    '''# loop through all the DICOM files and sort according to ImagePosition[2]
    i=0
    Image_Z_Positions = np.zeros(len(DCMFiles), dtype=float)
    for filenameDCM in DCMFiles:
        # read the file
        print(filenameDCM)
        ds = dicom.read_file(filenameDCM)
        Image_Z_Positions[i]=np.float64(ds.ImagePositionPatient[2])
        i=i+1
    Image_Z_Positions_index=np.argsort(-Image_Z_Positions)'''
    
    ##Create seriesInstanceUID
    OrganizationRoot = org_root
    ApplicationID = "62."
    ApplicationVersion = "1."
    PatientID=str(patientID)+"."
    SeriesNumber="5"
    year = strftime("%Y", localtime())
    mon = strftime("%m", localtime())
    day = strftime("%d", localtime())
    hour = strftime("%H", localtime())
    mins = strftime("%M", localtime())
    sec = strftime("%S", localtime())
    DateTime=year + mon + day + hour + mins + sec+ "."
    RN=np.random.choice(10, 7)
    RandomNumber=str(RN[0])+str(RN[1])+str(RN[2])+str(RN[3])+str(RN[4])+str(RN[5])+str(RN[6])
    SeriesInstanceUID=OrganizationRoot+ApplicationID+ApplicationVersion+PatientID+SeriesNumber+"."+DateTime+RandomNumber


    for i in range (0,edited_ct.shape[0]):
        #ds=dicom.read_file(DCMFiles[Image_Z_Positions_index[i+101]]) ###
        ds=dicom.read_file(DCMFiles[i]) ###
        ##Create SOPInstanceUID
        ImageNumber=str(i+1)+"."
        mins = strftime("%M", localtime())
        sec = strftime("%S", localtime())
        DateTime=year + mon + day + hour + mins + sec+ "."
        RN=np.random.choice(10, 6)
        RandomNumber=str(RN[0])+str(RN[1])+str(RN[2])+str(RN[3])+str(RN[4])+str(RN[5])
        SOPInstanceUID=OrganizationRoot+ApplicationID+ApplicationVersion+PatientID+SeriesNumber+"."+ImageNumber+DateTime+RandomNumber
        ds.SOPInstanceUID=SOPInstanceUID

        ds.InstanceCreationDate=year+mon+day
        ds.InstanceCreationTime=hour+mins+sec
        ##Your preferred name
        ds.Manufacturer='New CT'
        ds.SeriesInstanceUID=SeriesInstanceUID
        ds.Modality='CT'
        ds.SeriesNumber=SeriesNumber
        ds.InstanceNumber=str(i+1)
        ds.PixelData=(np.uint16(edited_ct[i,:,:]+1000)).tobytes()
        ds.RescaleIntercept="-1000"
        ds.RescaleSlope="1"
        ds.save_as(newDicompath+'/'+DCMFiles[i].split('/')[-1])
    


