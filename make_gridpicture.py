from PIL import Image
import glob
import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import time
from typing import Any, Callable, Iterable, Optional, Tuple, Union




def filepath2mask(filepath):
    filename = os.path.split(filepath)[1]
    folderpath = os.path.split(filepath)[0]
    
    return os.path.join(folderpath,f"{filename.split('-')[0]}-mask_merged.png")


def sort_criteria(filepath):
    filename = os.path.basename(filepath)
    outputs = filename.split('-')
    output = f"{outputs[0]}{outputs[-1]}"
    return output



def parallel_process(inputs: Iterable, fn: Callable, multiprocessing: int = 0):
    start = time.time()
    if multiprocessing:
        print('Starting multiprocessing')
        with Pool(multiprocessing) as pool:
            for _ in tqdm(pool.imap(fn, inputs), total=len(inputs)):
                pass
    else:
        for inp in tqdm(inputs):
            fn(inp)
    print(f'Finished in {time.time() - start:.1f}s')

def readimgs_savegrid(
    inp: list,
    xnum: int,
    ynum: int,
    basesize: Tuple[int, int],
    margins: Tuple[int, int],
    outputdir: str,
):
    basewidth, baseheight  = basesize
    xmargin, ymargin  = margins
    
    name = inp[0][0]
    name = sort_criteria(name)
    # change filenames to pil image file
    images = []
    for yy in range(ynum):
        xtemp=[]
        for xx in range(xnum): 
            xtemp.append(Image.open(inp[yy][xx]).convert('RGB'))
        images.append(xtemp)
    
    yaxis = []
    cc=1
    for imgxlist in images:
        widths, heights = zip(*((basewidth,baseheight) for j in imgxlist))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGBA', (total_width, max_height))

        x_offset = 0
        for im in imgxlist:
            new_im.paste(im.resize((basewidth,baseheight),Image.Resampling.BICUBIC), (x_offset,0))
            x_offset += basewidth+xmargin
        yaxis.append(new_im)
        if cc != len(images):
            new_im = Image.new('RGBA', (total_width, ymargin))
            yaxis.append(new_im)
            cc+=1

    widths, heights = zip(*(j.size for j in yaxis))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGBA', (max_width, total_height))

    y_offset = 0
    for im in yaxis:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    new_im.save(os.path.join(outputdir,name))
        
    
if __name__ == '__main__':
    figure_dir = './analysis/qualitative/'




    # folderlist = [
    #     ["./figure/new/momentdetr_pos_all_confi","./figure/new/momentdetr_cont_all_confi","./figure/new/momentdetr_weight_all_confi","./figure/tsm"],
    #     ["./figure/new/dab_pos_all_confi","./figure/new/dab_cont_all_confi","./figure/new/dab_weight_all_confi","./figure/tsm"],
    #     ["./figure/new/ours_pos_all_confi","./figure/new/ours_cont_all_confi","./figure/new/ours_weight_all_confi","./figure/tsm"]
    # ]

    # folderlist = [
    #     ["./figure/new/momentdetr_pos_all_confi","./figure/new/momentdetr_cont_all_confi","./figure/new/momentdetr_weight_all_confi","./figure/tsm"],
    #     ["./figure/new/dab_pos_all_confi","./figure/new/dab_cont_all_confi","./figure/new/dab_weight_all_confi","./figure/tsm"],
    #     ["./figure/new/ours_pos_all_confi","./figure/new/ours_attn_cont_all_confi_test","./figure/new/ours_weight_all_confi","./figure/tsm"]
    # ]

    # folderlist = [
    #     ["./figure/new/momentdetr_pos_initial_layernorm_last_confi","./figure/new/dab_pos_initial_layernorm_last_confi","./figure/new/ours_pos_initial_layernorm_last_confi","./figure/tsm"],
    #     ["./figure/new/momentdetr_cont_initial_layernorm_last_confi","./figure/new/dab_cont_initial_layernorm_last_confi","./figure/new/ours_cont_initial_layernorm_last_confi","./figure/tsm"],
    #     ["./figure/new/momentdetr_weight_initial_layernorm_last_confi","./figure/new/dab_weight_initial_layernorm_last_confi","./figure/new/ours_weight_initial_layernorm_last_confi","./figure/tsm"]
    # ]


    folderlist = [
        ["exp/refcoco/221208_094309_CRIS_R101_textfreeze/vis","exp/refcoco/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05/vis","exp/refcoco/221208_180923_CRIS_R101_blur3_addnorm/vis","exp/refcoco/221213_001301_CRIS_R101_blur3_perturbVG/vis","exp/refcoco/221213_094704_CRIS_R101_blur3_peturblearn/vis"],
        ["exp/refcoco/221208_094309_CRIS_R101_textfreeze/vis","exp/refcoco/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05/vis_usingvispt","exp/refcoco/221208_180923_CRIS_R101_blur3_addnorm/vis_usingvispt","exp/refcoco/221213_001301_CRIS_R101_blur3_perturbVG/vis_usingvispt","exp/refcoco/221213_094704_CRIS_R101_blur3_peturblearn/vis_usingvispt"]
    ]
    # do not set maskcoord ==0,0
    maskcoord = (1,0) # start 0, (column index, row index) replace mask path
    multiprocessing = 32 # number of process
    
    ynum=len(folderlist)
    xnum=len(folderlist[0])
    globfolderlist = []
    for yy in range(ynum):
        templist = []
        tempdict = {}
        for xx in range(xnum):
            temp = glob.glob(folderlist[yy][xx]+"/*-iou=*_merged.png")
            for filepath in temp:
                tempdict[sort_criteria(filepath)]=filepath
            sorted_dict = sorted(tempdict.items())
            temp = [v for k,v in sorted_dict]
            # temp.sort()
            templist.append(temp)
        globfolderlist.append(templist)

    outputdir =os.path.join(figure_dir,'test1')
    os.makedirs(outputdir,exist_ok=True)
    # basesize=Image.open(globfolderlist[0][0][0]).size
    basesize=(480, 640)
    basewidth, baseheight = basesize
    margins = (5,int(baseheight/5))

    processinputlist = []
    for i in tqdm(range(len(globfolderlist[0][0]))):
        images = []
        for yy in range(ynum):
            xtemp=[]
            for xx in range(xnum): 
                if (yy,xx) == maskcoord:
                    xtemp.append(filepath2mask(globfolderlist[yy][xx][i]))
                else:
                    xtemp.append(globfolderlist[yy][xx][i])
                    
                # print(sort_criteria(globfolderlist[yy][xx][i]))
            # one nxn filename list
            images.append(xtemp)
        processinputlist.append(images)
        
    kwargs = dict(xnum=xnum, ynum=ynum, basesize=basesize,margins=margins, outputdir=outputdir)
    print(kwargs)
    fn = partial(readimgs_savegrid, **kwargs)
    inputs = processinputlist
    parallel_process(inputs, fn, multiprocessing)
        
        
        
       