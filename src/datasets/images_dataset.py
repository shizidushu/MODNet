from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from scipy.ndimage import grey_dilation, grey_erosion
from torchvision import transforms
from scipy.ndimage import morphology
import os
import glob
from PIL import Image



class ImagesDataset(Dataset):
    def __init__(self, root, transform=None, ref_size=512):
        self.root = root
        self.transform = transform
        self.tensor = transforms.Compose([transforms.ToTensor()])
        self.ref_size = ref_size
        self.alphas = []
        self.alphas += sorted(glob.glob(self.root+'/masks/*'))   
        print('total imgs:', len(self.alphas))

    def getTrimap(self, alpha):
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
        trimap = fg
        trimap[unknown] = 0.5
        # print(trimap[:, :, :1].shape)
        return trimap  # [:, :, :1]

    def get_trimap(self, alpha):
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


    def __len__(self):
        return len(self.alphas)

    def __getitem__(self, idx):
        try:
            item = self.load_item(idx)
        except Exception as e:
            print('loading error: ', self.alphas[idx], e)
            item = self.load_item(0)  # 防止异常
        return item

    def load_item(self, idx):
        alpha = np.array(Image.open(self.alphas[idx]))  # cv2.imread(self.alphas[idx], -1), 防止libpng warning: iCCP
        alpha = alpha[..., -1] if len(alpha.shape) > 2 else alpha
        img_f = os.path.splitext(self.alphas[idx].replace('masks', 'images'))[0] + '.jpg'
        if not os.path.exists(img_f):
            img_f = img_f.replace('.jpg', '.png')
        # print(img_f)
        img = Image.open(img_f)  # rgb cv2.imread(img_f)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        im_h, im_w, im_c = img.shape
        #  非标准512x512图片，resize到短边为512~800，然后random crop
        if not (im_h == self.ref_size and im_w == self.ref_size):
            random_size = np.random.randint(512, 1201)
            if im_w >= im_h:
                im_rh = random_size
                im_rw = int(im_w / im_h * random_size)
            else:
                im_rw = random_size
                im_rh = int(im_h / im_w * random_size)

            img = cv2.resize(img, (im_rw, im_rh), cv2.INTER_CUBIC)
            alpha = cv2.resize(alpha, (im_rw, im_rh), cv2.INTER_CUBIC)
            # center crop
            # x0 = (im_rw - self.ref_size) // 2
            # y0 = (im_rh - self.ref_size) // 2
            # img = img[y0:y0+self.ref_size, x0:x0+self.ref_size, ...]
            # alpha = alpha[y0:y0+self.ref_size, x0:x0+self.ref_size, ...]
            # random crop
            x0 = random.randint(0, im_rw - self.ref_size + 1)
            y0 = random.randint(0, im_rh - self.ref_size + 1)
            img = img[y0:y0 + self.ref_size, x0:x0 + self.ref_size, ...]
            alpha = alpha[y0:y0 + self.ref_size, x0:x0 + self.ref_size, ...]



        trimap = self.get_trimap(alpha)
        # print(trimap.shape)

        # 左右镜像增广
        if np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...].copy()
            alpha = alpha[:, ::-1, ...].copy()
            trimap = trimap[:, ::-1, ...].copy()

        if self.transform:
            img = self.transform(img)
        alpha = self.tensor(alpha)
        trimap = self.tensor(trimap)
        return self.alphas[idx], img, trimap, alpha