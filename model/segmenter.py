import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
import copy
from .layers import FPN, Projector, TransformerDecoder


def freeze_all(model):
    """Freeze params and norm stats."""
    try:
        for m in model.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    except AttributeError:
        model.requires_grad = False
                
class NoiseMapper(nn.Module):
    def __init__(self, inputdim=512):
        super(NoiseMapper, self).__init__()
        self.linear1 = torch.nn.Linear(inputdim, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 1024)
        self.linear4 = torch.nn.Linear(1024, inputdim)
        self.linear5 = torch.nn.Linear(inputdim, 1024)
        self.linear6 = torch.nn.Linear(1024, 1024)
        self.linear7 = torch.nn.Linear(1024, 1024)
        self.linear8 = torch.nn.Linear(1024, inputdim)

    def forward(self, x):
        mu = F.leaky_relu(self.linear1(x))
        mu = F.leaky_relu(self.linear2(mu))
        mu = F.leaky_relu(self.linear3(mu))
        mu = self.linear4(mu)
        std = F.leaky_relu(self.linear5(x))
        std = F.leaky_relu(self.linear6(std))
        std = F.leaky_relu(self.linear7(std))
        std = self.linear8(std)
        return mu + std.exp()*(torch.randn(mu.shape).to(x.device))
    
    def loss(self, real, fake, temp=0.1, lam=0.5):
        sim = torch.cosine_similarity(real.unsqueeze(1), fake.unsqueeze(0), dim=-1)
        if temp > 0.:
            sim = torch.exp(sim/temp)
            sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
            sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
            if 0.<lam < 1.:
                return -(lam*torch.log(sim1) + (1.-lam)*torch.log(sim2))
            elif lam == 0:
                return -torch.log(sim2)
            else:
                return -torch.log(sim1)
        else:
            return -torch.diagonal(sim)

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        self.nodecoder = cfg.nodecoder
        if not self.nodecoder:
            # Decoder
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                            d_model=cfg.vis_dim,
                                            nhead=cfg.num_head,
                                            dim_ffn=cfg.dim_ffn,
                                            dropout=cfg.dropout,
                                            return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.vispt = True if cfg.visual_prompting is not None else False
        if self.vispt:
            self.clip_visenc = copy.deepcopy(self.backbone.visual)
            del self.clip_visenc.attnpool.connect
            self.backbone.logit_scale.requires_grad = False
            freeze_all(self.clip_visenc)
        self.textfreeze = cfg.textfreeze
        if self.textfreeze:
            # freeze_all(self.backbone.token_embedding)
            freeze_all(self.backbone.positional_embedding)
            freeze_all(self.backbone.transformer)
            freeze_all(self.backbone.text_projection)
            self.backbone.logit_scale.requires_grad = False
        self.vispt_perturb = cfg.vispt_perturb
        if self.vispt_perturb is not None:
            if self.vispt_perturb =='vanilla_gaussian':
                self.aug_level = 0.75
                
            elif self.vispt_perturb =='learned_perturb':
                self.mapper = NoiseMapper(512)
                self.mapper.load_state_dict(torch.load('exp/mapper/CRIS_R101_blur3_addnorm/noise_mapper.pth', map_location='cpu')) # path to the noise mapping network
                freeze_all(self.mapper)
    
    def forward(self, img, word, mask=None, vp_img = None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
            vp_img: b, 3, h, w
        '''
        if self.nodecoder:
            if self.training and self.vispt:
                assert vp_img is not None
                # padding mask used in decoder
                pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
                # vis: C3 / C4 / C5
                # word: b, length, 1024
                # vp_cls: b, 1024
                vis = self.backbone.encode_image(img)
                with torch.no_grad():
                    vp_cls = self.clip_visenc(vp_img.type(self.clip_visenc.conv1.weight.dtype),output_cls=True) # b, 512
                    vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * torch.sqrt(self.backbone.logit_scale) # should wrap no grad
                    if self.vispt_perturb is not None:
                        if self.vispt_perturb =='vanilla_gaussian':
                            random_noise = torch.randn(vp_cls.shape).to(img.device)
                            random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True)
                            vp_cls = vp_cls*(1-self.aug_level) + random_noise*self.aug_level
                            vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * torch.sqrt(self.backbone.logit_scale) # should wrap no grad
                            
                        elif self.vispt_perturb =='learned_perturb':
                            vp_cls = self.mapper(vp_cls) + vp_cls
                            vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * torch.sqrt(self.backbone.logit_scale) # should wrap no grad

                # b, 512, 26, 26 (C4)
                fq = self.neck(vis, vp_cls)
                b, c, h, w = fq.size()
                # fq = self.decoder(fq, word, pad_mask)
                # fq = fq.reshape(b, c, h, w)
                # b, 1, 104, 104
                pred = self.proj(fq, vp_cls)
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                return pred.detach(), mask, loss
            else:
                # padding mask used in decoder
                pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

                # vis: C3 / C4 / C5
                # word: b, length, 1024
                # state: b, 1024
                vis = self.backbone.encode_image(img)
                _, state = self.backbone.encode_text(word)
                state = state / state.norm(dim=-1, keepdim=True) * torch.sqrt(self.backbone.logit_scale)
                
                # b, 512, 26, 26 (C4)
                fq = self.neck(vis, state)
                b, c, h, w = fq.size()

                # b, 1, 104, 104
                pred = self.proj(fq, state)

                if self.training:
                    # resize mask
                    if pred.shape[-2:] != mask.shape[-2:]:
                        mask = F.interpolate(mask, pred.shape[-2:],
                                            mode='nearest').detach()
                    loss = F.binary_cross_entropy_with_logits(pred, mask)
                    return pred.detach(), mask, loss
                else:
                    return pred.detach()

        else:
            # padding mask used in decoder
            pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

            # vis: C3 / C4 / C5
            # word: b, length, 1024
            # state: b, 1024
            vis = self.backbone.encode_image(img)
            word, state = self.backbone.encode_text(word)

            # b, 512, 26, 26 (C4)
            fq = self.neck(vis, state)
            b, c, h, w = fq.size()
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)

            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                return pred.detach(), mask, loss
            else:
                return pred.detach()
