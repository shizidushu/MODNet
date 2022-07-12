from src.models.modnet import MODNet
import torch
import torch.nn as nn
import os
import logging
from src.datasets.base_dataset import BaseDataset, MattingTransform
from torch.utils.data import Dataset, DataLoader
from src.trainer import supervised_training_iter
import numpy as np

def main(root_dir, image_dir = "image", mask_dir = "matte", output_dir = '/home/ubuntu/data/yong/projects/MODNet/output', resume=False):
    modnet = MODNet()
    modnet = nn.DataParallel(modnet)
    if resume:
        VModel = sorted(os.listdir(output_dir))[-1]
        pretrained_ckpt = os.path.join(output_dir, VModel)
    else:
        pretrained_ckpt = './modnet_webcam_portrait_matting.ckpt'
    
    print(pretrained_ckpt)
    logging.info(f"model load {pretrained_ckpt}")

    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
    
    bs = 4  # batch size
    lr = 0.01  # learn rate
    epochs = 1000  # total epochs
    num_workers = 16
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                                   gamma=0.1)  # step_size 学习率下降迭代间隔次数， default: 每10次降低一次学习率
    train_transform = MattingTransform()
    dataset = BaseDataset(root_dir, image_dir= "images", mask_dir="masks", transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)

    for epoch in range(0, epochs):
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


if __name__ == '__main__':
    main(
        "/home/ubuntu/data/workspace/deeplabv3_plus/people_segmentation", image_dir= "images", mask_dir="masks")