import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

def object_crop(img, mask, context=0.0, square=False, image_size=224):
    img_crop, bbox = crop_mask(img, mask, context=context, square=square)
    img_crop = pad_to_square(img_crop, channel_dim=0)
    img_crop = torch.nn.functional.interpolate(img_crop.unsqueeze(0), (image_size, image_size)).squeeze(0)
    return img_crop

def crop_mask(img, mask, context=0.0, square=False):
    
    assert img.shape[1:] == mask.shape
    
    bbox = [mask.max(0).values.argmax(), mask.size(0) - mask.max(0).values.flip(0).argmax()]
    bbox += [mask.max(1).values.argmax(), mask.size(1) - mask.max(1).values.flip(0).argmax()]
    bbox = [int(x) for x in bbox]
    
    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])

    # square mask
    if square:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

        width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
        if height > width:
            bbox[2] = int(max(0, (bbox[2] - 0.5*height)))
            bbox[3] = bbox[2] + height
        else:
            bbox[0] = int(max(0, (bbox[0] - 0.5*width)))
            bbox[1] = bbox[0] + width
    else:
        bbox[0] = int(max(0, bbox[0] - context * height))
        bbox[1] = int(min(mask.size(0), bbox[1] + context * height))
        bbox[2] = int(max(0, bbox[2] - context * width))
        bbox[3] = int(min(mask.size(1), bbox[3] + context * width))

    width, height = (bbox[3] - bbox[2]), (bbox[1] - bbox[0])
    img_crop = img[:, bbox[2]: bbox[3], bbox[0]: bbox[1]]
    return img_crop, bbox


def pad_to_square(img, channel_dim=2, fill=0):
    """


    add padding such that a squared image is returned """
    
    from torchvision.transforms.functional import pad

    if channel_dim == 2:
        img = img.permute(2, 0, 1)
    elif channel_dim == 0:
        pass
    else:
        raise ValueError('invalid channel_dim')

    h, w = img.shape[1:]
    pady1 = pady2 = padx1 = padx2 = 0

    if h > w:
        padx1 = (h - w) // 2
        padx2 = h - w - padx1
    elif w > h:
        pady1 = (w - h) // 2
        pady2 = w - h - pady1

    img_padded = pad(img, padding=(padx1, pady1, padx2, pady2), padding_mode='constant')

    if channel_dim == 2:
        img_padded = img_padded.permute(1, 2, 0)

    return img_padded

def visual_prompt_eng(image, mask,image_size, blur=0, grayscale=False, center_context=None, rect=False, rect_color=(1,0,0), rect_width=2, 
                   brightness=1.0, bg_fac=1, colorize=False, outline=False):

    rw = rect_width

    # image : H,W,3 , RGB. [0,255]
    image = image.astype(np.float32)/255.
    img = image.cpu() if isinstance(image, torch.Tensor) else torch.from_numpy(image)
    img = img.permute(2,0,1) # c,H,W float [0,1]
    mask = mask.cpu() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask) # H,W float {0,1}
    
    img *= brightness
    img_bl = img
    if blur > 0: # best 5
        img_bl = torch.from_numpy(cv2.GaussianBlur(img.permute(1,2,0).numpy(), (15, 15), blur)).permute(2,0,1)
    
    if grayscale:
        img_bl = img_bl[1][None]

    #img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl
    # img_inp = img_ratio*img*mask + (1-img_ratio)*img_bl * (1-mask)
    img_inp = img*mask + (bg_fac) * img_bl * (1-mask)
    
    if rect:
        _, bbox = crop_mask(img, mask, context=0.1)
        img_inp[:, bbox[2]: bbox[3], max(0, bbox[0]-rw):bbox[0]+rw] = torch.tensor(rect_color)[:,None,None]
        img_inp[:, bbox[2]: bbox[3], max(0, bbox[1]-rw):bbox[1]+rw] = torch.tensor(rect_color)[:,None,None]
        img_inp[:, max(0, bbox[2]-1): bbox[2]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]
        img_inp[:, max(0, bbox[3]-1): bbox[3]+rw, bbox[0]:bbox[1]] = torch.tensor(rect_color)[:,None,None]

    if center_context is not None:
        # ex: center_context=0.5
        img_inp = object_crop(img_inp, mask, context=center_context, image_size=image_size)

    if colorize:
        img_gray = cv2.cvtColor(img.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)
        img_gray = torch.stack([torch.from_numpy(img_gray)]*3)
        img_inp = torch.tensor([1,0.2,0.2])[:,None,None] * img_gray * mask + bg_fac * img_gray * (1-mask)

    if outline:
        cont = cv2.findContours(mask.byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_img = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(outline_img, cont[0], -1, thickness=5, color=(255, 255, 255))
        outline_img = torch.stack([torch.from_numpy(outline_img)]*3).float() / 255.
        img_inp = torch.tensor([1,0,0])[:,None,None] *  outline_img + img_inp * (1- outline_img)
    
    img_inp = (img_inp.permute(1,2,0).numpy()*255.).astype(np.uint8)
    return img_inp


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length,visual_prompting=None,vispt_inval=False):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None
        self.visual_prompting = visual_prompting
        self.vispt_inval = vispt_inval

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            # mask transform
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            if self.visual_prompting is not None:
                vp_img = visual_prompt_eng(img,mask,self.input_size[0],**self.visual_prompting)
                vp_img = self.convert(vp_img)[0]
                img, mask = self.convert(img, mask)
                img = torch.stack([img,vp_img])
            else:
                img, mask = self.convert(img, mask)
            return img, word_vec, mask

        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            if self.visual_prompting is not None and self.vispt_inval:
                # mask transform
                mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                mask = cv2.warpAffine(mask,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
                mask = mask / 255.
                
                vp_img = visual_prompt_eng(img,mask,self.input_size[0],**self.visual_prompting)
                vp_img = self.convert(vp_img)[0]
                img = self.convert(img)[0]
                img = torch.stack([img,vp_img])
            else:
                img = self.convert(img)[0]
                
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params
        else:
            # sentence -> vector
            if self.visual_prompting is not None and self.vispt_inval:
                # mask transform
                mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                    cv2.IMREAD_GRAYSCALE)
                mask = cv2.warpAffine(mask,
                                    mat,
                                    self.input_size,
                                    flags=cv2.INTER_LINEAR,
                                    borderValue=0.)
                mask = mask / 255.
                vp_img = visual_prompt_eng(img,mask,self.input_size[0],**self.visual_prompting)
                vp_img = self.convert(vp_img)[0]
                img = self.convert(img)[0]
                img = torch.stack([img,vp_img])
            else:
                img = self.convert(img)[0]
                
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)
