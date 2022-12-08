import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
import copy
from .layers import FPN, Projector, TransformerDecoder


def freeze_all(model):
    """Freeze params and norm stats."""
    for m in model.modules():
        m.eval()
        for param in m.parameters():
            param.requires_grad = False


class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
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
            freeze_all(self.clip_visenc)
    
    def forward(self, img, word, mask=None, vp_img = None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
            vp_img: b, 3, h, w
        '''
        if self.vispt:
            if self.training:
                assert vp_img is not None
                # padding mask used in decoder
                pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
                # vis: C3 / C4 / C5
                # word: b, length, 1024
                # vp_cls: b, 1024
                vis = self.backbone.encode_image(img)
                with torch.no_grad():
                    vp_cls = self.clip_visenc(vp_img.type(self.clip_visenc.conv1.weight.dtype),output_cls=True) # b, 512
                    vp_cls = vp_cls / vp_cls.norm(dim=-1, keepdim=True) * self.backbone.logit_scale # should wrap no grad
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
                state = state / state.norm(dim=-1, keepdim=True) * self.backbone.logit_scale
                
                # b, 512, 26, 26 (C4)
                fq = self.neck(vis, state)
                b, c, h, w = fq.size()

                # b, 1, 104, 104
                pred = self.proj(fq, state)

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
