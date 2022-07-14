from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from scipy.ndimage import grey_dilation, grey_erosion
from torchvision import transforms
from scipy.ndimage import morphology
import torch
import os
import torch.nn.functional as F
from PIL import Image


class MattingTransform(object):
    def __init__(self) -> None:
        super(MattingTransform, self).__init__()
    
    def __call__(self, img, mask, ref_size = 512):
        # 将短边缩短到512
        im_h, im_w, im_c = img.shape
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)

        img = cv2.resize(img, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)

        # 随机裁剪出512x512
        bigger_side = max(im_rw, im_rh)
        rand_ind = random.randint(0, bigger_side - ref_size)

        if im_rh > im_rw:
            img = img[rand_ind:rand_ind+ref_size, :]
            mask = mask[rand_ind:rand_ind+ref_size, :]
        else:
            img = img[:, rand_ind:rand_ind+ref_size]
            mask = mask[:, rand_ind:rand_ind+ref_size]
        
        # 随机翻转
        if random.random()<0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        return img, mask


class BaseDataset(Dataset):
    def __init__(self, root_dir, image_dir = "image", mask_dir = "matte", transform = None, ref_size = 512):
        self.root_dir = root_dir
        self.transform = transform
        self.ref_size = ref_size

        self.imgs = sorted([os.path.join(self.root_dir, image_dir, img) for img in os.listdir(os.path.join(self.root_dir, image_dir))])
        self.masks = sorted([os.path.join(self.root_dir, mask_dir, aph) for aph in os.listdir(os.path.join(self.root_dir, mask_dir))])
        print(len(self.imgs))
        assert len(self.imgs) == len(self.masks), 'the number of dataset is different, please check it.'

    def gen_trimap(self, alpha):
        foreground = alpha > 0
        ### 以下连续几行修复了，当alpha为全0时候出错，即没有前景的是时候
        res = None
        res = Image.fromarray(foreground).getbbox()
        if res is None:
            left, upper, right, ylower = 0, 0, alpha.shape[1], alpha.shape[0]
        else:
            left, upper, right, ylower = res

        bbox_size = ((right - left) + (ylower - upper)) // 2
        d_size = bbox_size // 256 * random.randint(10, 20)  # dilate kernel size
        e_size = bbox_size // 256 * random.randint(10, 20)  # erode kernel size
        alpha = alpha / 255.0  # numpy array of your matte (with values between [0, 1])
        trimap = (alpha >= 0.9).astype('float32')
        not_bg = (alpha > 0).astype('float32')
        trimap[np.where(
            (grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        return trimap

    # def gen_trimap(self, matte):
    #     trimap = (matte >= 0.9).astype(np.float32)
    #     not_bg = (matte > 0).astype(np.float32)
    #     d_size = self.ref_size // 256 * random.randint(10, 20)
    #     e_size = self.ref_size // 256 * random.randint(10, 20)
    #     trimap[np.where((grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
    #     return trimap
    
    # def getTrimap(self, alpha):
    #     fg = np.array(np.equal(alpha, 255).astype(np.float32))
    #     unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
    #     unknown = unknown - fg
    #     unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
    #     trimap = fg
    #     trimap[unknown] = 0.5
    #     return trimap
        # print(trimap[:, :, :1].shape)
        # return trimap[:, :, :1]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        if np.amax(mask) > 1:
            mask = mask / 255.0

        if self.transform is not None:
            img, mask = self.transform(img, mask, ref_size = self.ref_size)
        else:
            img = cv2.resize(img, (self.ref_size, self.ref_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.ref_size, self.ref_size), interpolation=cv2.INTER_AREA)
        
        trimap = self.gen_trimap(mask)

        # 左右镜像增广
        if np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...].copy()
            mask = mask[:, ::-1, ...].copy()
            trimap = trimap[:, ::-1, ...].copy()


        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])

        img = img / 255.0
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        
        return self.imgs[index], img, trimap, mask


    
