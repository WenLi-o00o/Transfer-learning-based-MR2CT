import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import cv2
import torch
import pydicom as dicom


class UnalignedTrainDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.mr_max = opt.mr_max            #please specify the maximum MR intensity
        self.mr_min = opt.mr_min            #please specify the minimum MR intensity
        self.ct_max =  opt.ct_max              #please specify the maximum CT intensity
        self.ct_min = opt.ct_min               #please specify the minimum CT intensity
		
        self.is_train = True

    def resize(self, input, desired_size=286):
        old_size = input.shape[:2]  # old_size is in (height, width) format
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format
        output = cv2.resize(input, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        output = cv2.copyMakeBorder(output, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return output

    def __getparameters__(self, ndarray, output_size):
        if ndarray.ndim == 2:
            height, width = ndarray.shape
        else:
            _, height, width = ndarray.shape

        target_width, target_height = output_size
        if height == target_height and width == target_width:
            return 0, 0, height, width

        if self.is_train:
            h = random.randint(0, height - target_height)
            w = random.randint(0, width - target_width)
        else:
            h, w = 2, 2

        return h, w, target_height, target_width

    def __crop__(self, ndarray, output_size, parameters=None):
        if parameters is None:
            parameters = self.__getparameters__(ndarray, output_size)
        h, w, target_hegight, target_width = parameters
        if ndarray.ndim == 2:
            output = ndarray[h: h + target_hegight, w: w + target_width], parameters
        else:
            output = ndarray[:, h: h + target_hegight, w: w + target_width], parameters

        return output

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        #print("A_path:", A_path)
        #print("B_path:", B_path)
        #arrayA=np.loadtxt(A_path,delimiter=',')
        arrayA = dicom.read_file(A_path).pixel_array
        arrayA = cv2.resize(arrayA,(256,256))
        arrayB = dicom.read_file(B_path).pixel_array
        arrayB = cv2.resize(arrayB,(256,256))
		
        #arrayB=np.loadtxt(B_path,delimiter=',')
		
        mr = arrayA
        ct = arrayB
		
        # flip
        if self.is_train and random.random() < 0.5:
            mr = cv2.flip(mr, 1)
            ct = cv2.flip(ct, 1)

        # rotation
        if self.is_train and random.random() < 0.3:
            rotation_factor = 30
            rotation_angle = np.clip(random.random() * rotation_factor, -rotation_factor * 2, rotation_factor * 2)
            rows, cols = mr.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            mr = cv2.warpAffine(mr, rotation_matrix, (cols, rows))
            ct = cv2.warpAffine(ct, rotation_matrix, (cols, rows))

        # crop
        mr, params = self.__crop__(mr, [256, 256])
        ct, _ = self.__crop__(ct, [256, 256], params)

        # normalize
        mr = ((torch.from_numpy(mr.astype(float)) - self.mr_min) / (self.mr_max- self.mr_min) - 0.5) / 0.5
        ct = ((torch.from_numpy(ct.astype(float)) - self.ct_min) / (self.ct_max- self.ct_min) - 0.5) / 0.5
        mr = mr.float().unsqueeze(0)
        ct = ct.float().unsqueeze(0)
        return {'A': mr, 'B': ct, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedTrainDataset'
