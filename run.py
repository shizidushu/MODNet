import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import morphology
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import copy
import logging

logging.basicConfig(filename='train.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('--------------------------------')
torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class ImagesDataset(Dataset):
    def __init__(self, root, transform=None, w=1024, h=576):
        self.root = root
        self.transform = transform
        self.tensor = transforms.Compose([transforms.ToTensor()])
        self.w = w
        self.h = h
        self.imgs = sorted([os.path.join(self.root, 'image', img) for img in os.listdir(os.path.join(self.root, 'image'))])
        self.alphas = sorted([os.path.join(self.root, 'alpha', aph) for aph in os.listdir(os.path.join(self.root, 'alpha'))])
        print(len(self.imgs))
        assert len(self.imgs) == len(self.alphas), 'the number of dataset is different, please check it.'


    def getTrimap(self, alpha):
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(10, 20)
        trimap = fg
        trimap[unknown] = 0.5
        return trimap[:, :, :1]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        alpha = cv2.imread(self.alphas[idx])
        # h, w, c = img.shape
        # rh = 512
        # rw = int(w / h * 512)
        # rh = rh - rh % 32 #512
        # rw = rw - rw % 32 #896    1024 or 576 %32==0
        img = cv2.resize(img, (self.w, self.h))
        alpha = cv2.resize(alpha, (self.w, self.h))
        trimap = self.getTrimap(alpha)
        if self.transform:
            img = self.transform(img)
        alpha = self.tensor(alpha[:, :, 0])
        return self.imgs[idx], img, trimap, alpha



class ImagesSocDataset(Dataset):
    def __init__(self, root, transform=None, w=1024, h=576):
        self.root = root
        self.w = w
        self.h = h
        self.transform = transform
        self.imgs = [os.path.join(self.root, img) for img in os.listdir(self.root)]
        # self.imgs = self.addFlip()


    def addFlip(self):
        """flip image"""
        all_imgs = []
        for file in self.imgs:
            img = cv2.imread(file)
            f_img = cv2.flip(img, 1)
            all_imgs.append(img)
            all_imgs.append(f_img)
        return all_imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if type(img)==type('a'):
            img = cv2.imread(img)
        img = cv2.resize(img, (self.w, self.h))
        if self.transform:
            img = self.transform(img)
        return img





def main(root, resume=True, std=0):
    save_model_dir = 'SaveModel'
    if resume:
        VModel=sorted(os.listdir(save_model_dir))[-1]
        pretrained_ckpt = os.path.join(save_model_dir, VModel)
    else:
        pretrained_ckpt = './modnet_webcam_portrait_matting.ckpt'
    print(pretrained_ckpt)
    logging.info(f"model load {pretrained_ckpt}")

    modnet = MODNet()
    modnet = nn.DataParallel(modnet)
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    bs = 6  # batch size
    lr = 0.0001  # learn rate
    epochs = 100  # total epochs
    num_workers = 16
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 13, 23, 33], gamma=0.1)
    # dataloader = CREATE_YOUR_DATALOADER(bs)  # NOTE: please finish this function
    dataset = ImagesDataset(root, torch_transforms)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)
    for epoch in range(std, epochs):
        if std == 0:
            epoch += 1
        mattes = []
        for idx, (img_file, image, trimap, gt_matte) in enumerate(dataloader, start=1):
            try:
                trimap = np.transpose(trimap, (0, 3, 1, 2)).float().cuda()
            except:
                print('----------------------',img_file, optimizer.param_groups[0]['lr'])
                continue
            image = image.cuda()
            gt_matte = gt_matte.cuda()
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap,
                                                                              gt_matte)
            info = f"epoch: {epoch}/{epochs} semantic_loss: {semantic_loss}, detail_loss: {detail_loss}, matte_loss： {matte_loss}"
            if semantic_loss > 1 or detail_loss > 1 or matte_loss > 1:
                logging.info(img_file)
            print(idx, info, optimizer.param_groups[0]['lr'])
            mattes.append(float(matte_loss))
        avg_matte = float(np.mean(mattes))
        logging.info(f"epoch: {epoch}/{epochs}, matte_loss: {avg_matte}")
        lr_scheduler.step()
        torch.save(modnet.state_dict(), os.path.join(save_model_dir, 'matting_{:0>4d}.ckpt'.format(epoch)))
        print(f'----------{epoch}--------------save model over-----------------------------------')
        logging.info(f'------save model------{epoch}  {epoch}.ckpt')


def mainSoc(root, std=0):
    save_model_dir = 'SaveModel'
    VModel=sorted(os.listdir(save_model_dir))[-1]
    pretrained_ckpt = os.path.join(save_model_dir, VModel)
    print(pretrained_ckpt)
    logging.info(f"model load {pretrained_ckpt}")

    modnet = MODNet()
    modnet = nn.DataParallel(modnet)
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    bs = 8  # batch size
    lr = 0.00001  # learn rate 0.00001
    epochs = 60  # total epochs 10
    num_workers = 16
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    dataset = ImagesSocDataset(root, torch_transforms)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)

    for epoch in range(std, epochs):
        semantic_loss=[]
        detail_loss=[]
        backup_modnet = copy.deepcopy(modnet)
        for idx, image in enumerate(dataloader, start=1):
            image = image.cuda()
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
            info = f"epoch: {epoch}/{epochs} soc_semantic_loss: {soc_semantic_loss}, soc_detail_loss: {soc_detail_loss}"
            if soc_semantic_loss > 1 or soc_detail_loss>1:
                print(idx, info)
            print(idx, info)
            semantic_loss.append(float(soc_semantic_loss))
            detail_loss.append(float(soc_detail_loss))
        avg_semantic_loss = float(np.mean(semantic_loss))
        avg_detail_loss = float(np.mean(detail_loss))
        logging.info(f"epoch: {epoch}/{epochs}, avg_semantic_loss: {avg_semantic_loss}, avg_detail_loss: {avg_detail_loss}")
        torch.save(modnet.state_dict(), os.path.join(save_model_dir, 'soc_{:0>2d}.ckpt'.format(epoch)))
        print(f'------save soc model------{epoch}  {epoch}.ckpt')


if __name__ == '__main__':
    path = 'MattingDataset/train'
    soc_path="MattingDataset/trainSoc"
    main(path, std=10)
    mainSoc(soc_path)


