from src.models.modnet import MODNet

from torchvision import transforms
import torch
import torch.nn as nn
import os
import logging
from src.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset, DataLoader
from src.trainer import supervised_training_iter, soc_adaptation_iter
import numpy as np
import copy

def main(dataset, output_dir = '/home/ubuntu/data/yong/projects/MODNet/output', resume=False, batch_size = 22):
    modnet = MODNet()
    modnet = nn.DataParallel(modnet)

    last_checkpoint = None
    if resume:
        checkpoints = [output_dir + "/" + name for name in os.listdir(output_dir)]
        if len(checkpoints) > 0:
            last_checkpoint = max(checkpoints, key=os.path.getctime)

    modnet = modnet.cuda()

    pretrained_ckpt = '/home/ubuntu/data/yong/projects/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
    print(f"load pretrained {pretrained_ckpt}")
    modnet.load_state_dict(torch.load(pretrained_ckpt))

    if last_checkpoint:
        print(f"load last checkpoint {last_checkpoint}")
        modnet.load_state_dict(torch.load(last_checkpoint))
    
    bs = batch_size  # batch size
    lr = 0.001  # learn rate
    epochs = 1000  # total epochs
    num_workers = 16
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                   gamma=0.1)  # step_size 学习率下降迭代间隔次数， default: 每10次降低一次学习率
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)

    if resume and last_checkpoint is not None:
        start_epoch = int(last_checkpoint.split(".")[0].split('_')[-1]) + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        mattes = []
        for idx, (img_file, image, trimap, gt_matte) in enumerate(dataloader):
            image = image.cuda()
            gt_matte = gt_matte.cuda()
            trimap = trimap.cuda()
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(
                modnet, optimizer, image, trimap, gt_matte
            )
            info = f"epoch: {epoch}/{epochs} semantic_loss: {semantic_loss}, detail_loss: {detail_loss}, matte_loss: {matte_loss}"
            if semantic_loss > 1 or detail_loss > 1 or matte_loss > 1:
                logging.info(img_file)
            print(idx, info, optimizer.param_groups[0]['lr'])
            mattes.append(float(matte_loss))
        avg_matte = float(np.mean(mattes))
        logging.info(f"epoch: {epoch}/{epochs}, matte_loss: {avg_matte}")
        lr_scheduler.step()
        torch.save(modnet.state_dict(), os.path.join(output_dir, 'matting_{:0>4d}.ckpt'.format(epoch)))
        print(f'----------{epoch}--------------save model over-----------------------------------')
        logging.info(f'------save model------{epoch}  {epoch}.ckpt')


def train_soc(dataset, output_dir = '/home/ubuntu/data/yong/projects/MODNet/output'):
    modnet = MODNet()
    modnet = nn.DataParallel(modnet)

    checkpoints = [output_dir + "/" + name for name in os.listdir(output_dir)]
    pretrained_ckpt = max(checkpoints, key=os.path.getctime)

    
    print(pretrained_ckpt)
    logging.info(f"model load {pretrained_ckpt}")

    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
    
    bs = 18  # batch size
    lr = 0.00001  # learn rate
    epochs = 60  # total epochs
    num_workers = 16
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)

    for epoch in range(0, epochs):
        semantic_loss=[]
        detail_loss=[]
        backup_modnet = copy.deepcopy(modnet)
        for idx, (img_file, image, trimap, gt_matte) in enumerate(dataloader):
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
        torch.save(modnet.state_dict(), os.path.join(output_dir, 'soc_{:0>2d}.ckpt'.format(epoch)))
        print(f'------save soc model------{epoch}  {epoch}.ckpt')



if __name__ == '__main__':
    dataset = BaseDataset(
        "/home/ubuntu/data/workspace/deeplabv3_plus/people_segmentation",
        "images",
        "masks",
        ref_size=512)
    dataset.add_samples(
        "/home/ubuntu/data/yong/projects/MODNet/data/PPM-100",
        "image",
        "matte"
    )
    main(dataset, resume=True)