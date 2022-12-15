import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import MultiStepLR

from loguru import logger

import utils.config as config
from engine.engine import inference
from model import build_segmenter
from utils.dataset import RefDataset
from utils.misc import setup_logger

from tqdm import tqdm
from model.visual_prompt import ResnetGenerator, pseudo_text_loss
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
from tensorboardX import SummaryWriter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    # folder define
    args.output_dir = os.path.join('exp/generator', args.exp_name)
    args.writer = SummaryWriter(args.output_dir)
    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)    
    
    # build dataset & dataloader
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                           mask_dir=args.mask_root,
                           dataset=args.dataset,
                           split=args.train_split,
                           mode='train',
                           input_size=args.input_size,
                           word_length=args.word_len,
                           visual_prompting = args.visual_prompting)
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=64,
                                              pin_memory=True)
    args.base_lr = args.base_lr*1000
    # build model
    model, _ = build_segmenter(args)
    # model = torch.nn.DataParallel(model).cuda()
    logger.info(model)
    model = model.cuda()
    
    # network define
    visual_promptor = ResnetGenerator()
    visual_promptor = visual_promptor.cuda()

    visual_promptor_param = []
    for k, v in visual_promptor.named_parameters():
        if v.requires_grad:
            visual_promptor_param.append(v)
    param_list = [{
        'params': visual_promptor_param,
        'initial_lr': args.base_lr
    }]
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=[15,30,40,45],
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()
   
    visual_promptor.train()
    model.eval()
    l1loss = torch.nn.L1Loss()
    for epoch in range(args.start_epoch, args.epochs):
        datalength = len(train_loader)
        for i, (image, text, target) in enumerate(train_loader):
            # data
            if args.visual_prompting is not None:
                image = image.permute(1,0,2,3,4)
                vp_img = image[1]
                vp_img = vp_img.cuda(non_blocking=True)
                image = image[0]
            image = image.cuda(non_blocking=True)
            text = text.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).unsqueeze(1)
            # # multi-scale training
            # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')
            generated_img = visual_promptor(image,target)
            vp_cls = model.clip_visenc(generated_img.type(model.clip_visenc.conv1.weight.dtype),output_cls=True)
            vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * torch.sqrt(model.backbone.logit_scale) # should wrap no grad
            # forward
            with torch.no_grad():
                _, state = model.backbone.encode_text(text)
                state = state / state.norm(dim=-1, keepdim=True) * torch.sqrt(model.backbone.logit_scale)
            loss1 = pseudo_text_loss(state, vp_cls).mean()
            # print(l1loss(generated_img,image))
            loss = loss1

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"epoch:{epoch}, iteration:{i}/{datalength}, loss:{loss}, loss1: {loss1.mean()}")
            args.writer.add_scalar('textloss',loss1,epoch * len(train_loader) + (i + 1))
            # args.writer.add_scalar('L1loss',loss2,epoch * len(train_loader) + (i + 1))
            # break
        
        # update lr
        scheduler.step(epoch)
        if epoch%2==0:
            args.model_dir = os.path.join(args.output_dir, "visual_promptor.pth")
            # save model
            torch.save(visual_promptor.state_dict(),args.model_dir)
            with torch.no_grad():
                image1 = visual_promptor.to_img(image[0].cpu()).numpy().astype(np.uint8)
                generated_img1 = visual_promptor.to_img(generated_img[0].cpu()).detach().numpy().astype(np.uint8)
                target1 = target[0].cpu().squeeze().numpy().astype(np.uint8)
                target1 = target1*255
            
            args.writer.add_image('image1',image1,epoch)
            args.writer.add_image('generated image1',generated_img1,epoch)
            args.writer.add_image('mask1',target1,epoch,dataformats='HW')
            
            torch.cuda.empty_cache()
    args.writer.close()
            

    
if __name__ == '__main__':
    main()