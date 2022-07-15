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
import glob

class BaseDataset(Dataset):
    def __init__(self, root_dir, img_dir = "images", alpha_dir = "masks", ref_size = 512):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.alpha_dir = alpha_dir
        self.ref_size = ref_size

        self.samples = []
        self.add_samples(self.root_dir, self.img_dir, self.alpha_dir)
    
    def add_samples(self, root_dir, img_dir = "image", alpha_dir = "masks"):
        alpha_path_pattern = os.path.sep.join([root_dir, alpha_dir, "*"])
        alpha_paths = sorted(glob.glob(alpha_path_pattern))
        for alpha_path in alpha_paths:
            img_path = os.path.splitext(alpha_path.replace(alpha_dir, img_dir))[0] + '.jpg'
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')
            if not os.path.exists(img_path):
                print(f"Not found image for mask file: {alpha_path}")
                continue
            self.samples.append({
                "img_path": img_path,
                "alpha_path": alpha_path
            })
        print(f"{len(self.samples)} samples")

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
        
        trimap = (alpha >= 0.9).astype('float32')
        not_bg = (alpha > 0).astype('float32')
        trimap[np.where(
            (grey_dilation(not_bg, size=(d_size, d_size)) - grey_erosion(trimap, size=(e_size, e_size))) != 0)] = 0.5
        return trimap
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path = self.samples[index]["img_path"]
        alpha_path = self.samples[index]["alpha_path"]

        img = Image.open(img_path)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        alpha = np.array(Image.open(alpha_path))
        alpha = alpha[..., -1] if len(alpha.shape) > 2 else alpha
        
        img, alpha = self.resize_and_crop(img, alpha, self.ref_size)

        if np.amax(alpha) > 1:
            alpha = alpha / 255.0 # numpy array of your matte (with values between [0, 1])
        
        trimap = self.gen_trimap(alpha)

        img, alpha, trimap = self.augment(img, alpha, trimap)

        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])

        img = img / 255.0
        img = img - 0.5
        img = img / 0.5

        return alpha_path, img, trimap, alpha
    
    def resize_and_crop(self, img, alpha, ref_size = 512, random_scale = 1.5):
        # 将短边缩短到512
        im_h, im_w, im_c = img.shape
        # 非标准512x512图片，resize到短边为ref_size~ref_size*random_scale
        # 然后center crop 或 random crop
        if not (im_h == ref_size and im_w == ref_size):
            random_size = np.random.randint(ref_size, int(ref_size * random_scale))
            if im_w >= im_h:
                im_rh = random_size
                im_rw = int(im_w / im_h * random_size)
            elif im_w < im_h:
                im_rw = random_size
                im_rh = int(im_h / im_w * random_size)

            img = cv2.resize(img, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(alpha, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)

            if np.random.random() < 0.2:
                # center crop
                x0 = (im_rw - ref_size) // 2
                y0 = (im_rh - ref_size) // 2
                img = img[y0:y0+ref_size, x0:x0+ref_size, ...]
                alpha = alpha[y0:y0+ref_size, x0:x0+ref_size, ...]
            else:
                # random crop
                x0 = random.randint(0, im_rw - ref_size)
                y0 = random.randint(0, im_rh - ref_size)
                img = img[y0:y0 + ref_size, x0:x0 + ref_size, ...]
                alpha = alpha[y0:y0 + ref_size, x0:x0 + ref_size, ...]
        
        return img, alpha
    
    def augment(self, img, alpha, trimap):
        # 左右镜像增广
        if np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            alpha = alpha[:, ::-1, ...]
            trimap = trimap[:, ::-1, ...]
        
        return img, alpha, trimap
    


    
