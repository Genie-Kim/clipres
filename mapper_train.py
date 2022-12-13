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
from model.visual_prompt import NoiseMapper

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


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
    args.output_dir = os.path.join('exp/mapper', args.exp_name)

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
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=64,
                                              pin_memory=True)
    args.base_lr = args.base_lr*10
    # build model
    model, _ = build_segmenter(args)
    # model = torch.nn.DataParallel(model).cuda()
    logger.info(model)
    model = model.cuda()
    
    mapper = NoiseMapper(512)
    mapper = mapper.cuda()

    mapper_param = []
    for k, v in mapper.named_parameters():
        if v.requires_grad:
            mapper_param.append(v)
    param_list = [{
        'params': mapper_param,
        'initial_lr': args.base_lr
    }]
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()
   
    mapper.train()
    model.eval()
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

            # forward
            with torch.no_grad():
                vp_cls = model.clip_visenc(vp_img.type(model.clip_visenc.conv1.weight.dtype),output_cls=True) # b, 512
                vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * torch.sqrt(model.backbone.logit_scale) # should wrap no grad
                _, state = model.backbone.encode_text(text)
                state = state / state.norm(dim=-1, keepdim=True) * torch.sqrt(model.backbone.logit_scale)
            noised_pseudo = mapper(vp_cls) + vp_cls
            noised_pseudo = noised_pseudo / noised_pseudo.norm(dim=-1, keepdim=True) * torch.sqrt(model.backbone.logit_scale) # should wrap no grad
            loss = mapper.loss(state, noised_pseudo)
            loss = loss.mean()

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"epoch:{epoch}, iteration:{i}/{datalength}, loss:{loss}")
        
        # update lr
        scheduler.step(epoch)

    args.model_dir = os.path.join(args.output_dir, "noise_mapper.pth")
    # save model
    torch.save(mapper.state_dict(),args.model_dir)

    
if __name__ == '__main__':
    main()
