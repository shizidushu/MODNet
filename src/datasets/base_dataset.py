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
    def __init__(self, root_dir, img_dir = "images", alpha_dir = "masks", ref_size = 512, sample_weight = 1.0):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.alpha_dir = alpha_dir
        self.ref_size = ref_size

        self.samples = []
        self.add_samples(self.root_dir, self.img_dir, self.alpha_dir, sample_weight = sample_weight)
    
    def add_samples(self, root_dir, img_dir = "image", alpha_dir = "masks", sample_weight = 1.0):
        alpha_path_pattern = os.path.sep.join([root_dir, alpha_dir, "*"])
        alpha_paths = sorted(glob.glob(alpha_path_pattern))
        for alpha_path in alpha_paths:
            img_path = os.path.splitext(alpha_path.replace(alpha_dir, img_dir))[0] + '.jpg'
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')
            if not os.path.exists(img_path):
                img_path = img_path.replace('-profile.png', '.jpg')               
            if not os.path.exists(img_path):
                print(f"Not found image for mask file: {img_path}")
                continue
            self.samples.append({
                "img_path": img_path,
                "alpha_path": alpha_path,
                "sample_weight": sample_weight
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
        # print(alpha_path)
        img = Image.open(img_path)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        alpha = np.array(Image.open(alpha_path))
        if alpha.shape[-1] == 4:
            if len(np.unique(alpha[..., -1])) == 1:
                alpha = alpha[..., :-1]
        alpha = alpha[..., -1] if len(alpha.shape) > 2 else alpha

        if 'profile' in self.samples[index]["alpha_path"]:
            alpha = 255 - alpha
        
        img, alpha = self.resize_and_crop_by_bbox(img, alpha, self.ref_size)
        # img = cv2.resize(img, (self.ref_size, self.ref_size), interpolation=cv2.INTER_LINEAR)
        # alpha = cv2.resize(alpha, (self.ref_size, self.ref_size), interpolation=cv2.INTER_LINEAR)

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

        return alpha_path, self.samples[index]["sample_weight"], img, trimap, alpha
    
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

            # # center crop
            # x0 = (im_rw - ref_size) // 2
            # y0 = (im_rh - ref_size) // 2
            # img = img[y0:y0+ref_size, x0:x0+ref_size, ...]
            # alpha = alpha[y0:y0+ref_size, x0:x0+ref_size, ...]

            # crop center in x and bottom in y
            # This may give more weights to foot
            # x0 = (im_rw - ref_size) // 2
            # y0 = im_rh - ref_size
            # img = img[y0:y0 + ref_size, x0:x0 + ref_size, ...]
            # alpha = alpha[y0:y0 + ref_size, x0:x0 + ref_size, ...]

            # random crop
            x0 = random.randint(0, im_rw - ref_size)
            y0 = random.randint(0, im_rh - ref_size)
            img = img[y0:y0 + ref_size, x0:x0 + ref_size, ...]
            alpha = alpha[y0:y0 + ref_size, x0:x0 + ref_size, ...]
        
        return img, alpha
    
    def simple_resize_and_crop(self, img, alpha, ref_size = 512):
        # 将短边缩短到512
        im_h, im_w, im_c = img.shape
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)

        img = cv2.resize(img, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)

        # prefer center or bottom
        if im_rh > im_rw:
            y0 = im_rh - ref_size
            img = img[y0:y0+ref_size, :]
            alpha = alpha[y0:y0+ref_size, :]
        else:
            x0 = (im_rw - ref_size) // 2
            img = img[:, x0:x0+ref_size]
            alpha = alpha[:, x0:x0+ref_size]

        # 随机裁剪出512x512
        # bigger_side = max(im_rw, im_rh)
        # rand_ind = random.randint(0, bigger_side - ref_size)
        # if im_rh > im_rw:
        #     img = img[rand_ind:rand_ind+ref_size, :]
        #     alpha = alpha[rand_ind:rand_ind+ref_size, :]
        # else:
        #     img = img[:, rand_ind:rand_ind+ref_size]
        #     alpha = alpha[:, rand_ind:rand_ind+ref_size]
        
        return img, alpha
    
    def resize_and_crop_by_alpha(self, img, alpha, ref_size = 512, random_scale = 1.5):
        rect = get_bbox(alpha)
        rect_width = rect[2] - rect[0]
        rect_height = rect[3] - rect[1]

        im_h, im_w, im_c = img.shape

        # 先取alpha的BBOX
        ## 如果BBOX的长边过小(不足ref_size的一半），则放大BBOX长边到(ref_size // 2, ref_size)
        random_size = np.random.randint(ref_size // 2, ref_size - 1)
        if rect_width >= rect_height:
            r_scale = random_size / rect_width
        else:
            r_scale = random_size / rect_height
        r_scale = 1 if r_scale < 1 else r_scale
        im_rw = int(im_w * r_scale)
        im_rh = int(im_h * r_scale)
        # print(f'phase 1: {im_rh, im_rw}')
        ## 如果放大后的图片短边不足ref_size，或者之前没有放大
        ## 则将图片短边Resize到(ref_size, ref_size * 1.5)
        if min(im_rh, im_rw) < ref_size or r_scale == 1:
            random_size = np.random.randint(ref_size, int(ref_size * random_scale))
            if im_rw >= im_rh:
                im_rw = int(im_rw / im_rh * random_size)
                im_rh = random_size
            elif im_rw < im_rh:
                im_rh = int(im_rh / im_rw * random_size)
                im_rw = random_size
            # print(f'phase 2: {im_rh, im_rw}')
        
        img = cv2.resize(img, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
        rect = get_bbox(alpha)
        rect_width = rect[2] - rect[0]
        rect_height = rect[3] - rect[1]
        # print(rect)
        # print(f'new rect height, wdith: {rect_height, rect_width}')

        # random crop
        x0 = random.randint(max(0, rect[0] - ref_size), min(im_rw - ref_size, rect[2]))
        y0 = random.randint(max(0, rect[1] - ref_size), min(im_rh - ref_size, rect[3]))

        img = img[y0:y0 + ref_size, x0:x0 + ref_size, ...]
        alpha = alpha[y0:y0 + ref_size, x0:x0 + ref_size, ...]
        
        return img, alpha
        
    
    def resize_and_crop_by_bbox(self, img, alpha, ref_size = 512, random_scale = 1.5):
        rect = get_bbox(alpha)
        rect_width = rect[2] - rect[0]
        rect_height = rect[3] - rect[1]

        im_h, im_w, im_c = img.shape

        # 将BBOX裁剪出来，H和W最多再各取rect对应一半
        # width_pad = int(rect_width / 4.0)
        # height_pad = int(rect_height / 4.0)
        # x_start = max(rect[0] - width_pad, 0)
        # x_end = min(rect[2] + width_pad, im_w - 1)
        # y_start = max(rect[1] - height_pad, 0)
        # y_end = min(rect[3] + height_pad, im_h - 1)
        # img = img[y_start:y_end, x_start:x_end, ...]
        # alpha = alpha[y_start:y_end, x_start:x_end, ...]

        # 将BBOX裁剪出来，H和W最多再各取rect长边的一半
        width_pad = int(rect_width / 2.0)
        height_pad = int(rect_height / 2.0)
        pad_for_long_side = max(width_pad, height_pad)
        pad_for_short_side = pad_for_long_side + int(abs(rect_height - rect_width) / 2.0)
        if im_h >= im_w:
            x_start = max(rect[0] - pad_for_short_side, 0)
            x_end = min(rect[2] + pad_for_short_side, im_w - 1)
            y_start = max(rect[1] - pad_for_long_side, 0)
            y_end = min(rect[3] + pad_for_long_side, im_h - 1)
        else:
            x_start = max(rect[0] - pad_for_long_side, 0)
            x_end = min(rect[2] + pad_for_long_side, im_w - 1)
            y_start = max(rect[1] - pad_for_short_side, 0)
            y_end = min(rect[3] + pad_for_short_side, im_h - 1)
        alpha = alpha[y_start:y_end, x_start:x_end, ...]

        # 将短边缩短到512
        im_h, im_w, im_c = img.shape
        # 非标准512x512图片，resize到短边为ref_size~ref_size*random_scale
        # 然后center crop 或 random crop
        random_size = np.random.randint(ref_size, int(ref_size * random_scale))
        if im_w >= im_h:
            im_rh = random_size
            im_rw = int(im_w / im_h * random_size)
        elif im_w < im_h:
            im_rw = random_size
            im_rh = int(im_h / im_w * random_size)

        img = cv2.resize(img, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (im_rw, im_rh), interpolation=cv2.INTER_LINEAR)

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
    

def get_bbox(alpha):
    foreground = alpha > 0.0
    res = None
    res = Image.fromarray(foreground).getbbox()
    if res is None:
        left, upper, right, ylower = 0, 0, alpha.shape[1], alpha.shape[0]
    else:
        left, upper, right, ylower = res
    return (left, upper, right, ylower)
