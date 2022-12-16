# CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05.yaml
# CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_180923_CRIS_R101_blur3_addnorm/221208_180923_CRIS_R101_blur3_addnorm.yaml
# CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_181027_CRIS_R101/221208_181027_CRIS_R101.yaml
# CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_094309_CRIS_R101_textfreeze/221208_094309_CRIS_R101_textfreeze.yaml

CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05/221208_180849_CRIS_R101_blur3_bgfac05_ccrop05.yaml
CUDA_VISIBLE_DEVICES=0 python -u test_trainset.py --config exp/refcoco/221208_180923_CRIS_R101_blur3_addnorm/221208_180923_CRIS_R101_blur3_addnorm.yaml



# # dump scripts
# ff = json.load(open('val.json', 'r'))

# def find_segids(segid,ff):
#     imgname=None
#     for item in ff:
#         if item['segment_id']==segid:
#                 imgname=item['img_name']
#                 break
#     for item in ff:
#         if item['img_name']==imgname:
#                 print(item['segment_id'])