import json
import os
root = '/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/'
a = open('/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sharegpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json', 'r')
object=json.load(a)

total_num = 0
exists_num = 0
nonexists_num = 0

nonexists_list = []
for obj in object:
    try:
        obj['image']
    except:
        continue
    # if 'sam/images' in obj['image']:
    total_num += 1
    if os.path.isfile(root + obj['image']):
        exists_num += 1
    else:
        nonexists_num += 1
        nonexists_list.append(obj['image'])

print("ShareGPT4-V")
print(f'Total num of {len(object)}. Among them, instruction with images has {total_num}, but there are only {exists_num} and ther are not exists for {nonexists_num}')
print('ok')


import json
import os
root = '/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/'
a = open('/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sharegpt4v/lbk.json', 'r')
object=json.load(a)

total_num = 0
exists_num = 0
nonexists_num = 0
exists_box_num = 0
nonexists_box_num = 0

nonexists_list = []
for obj in object:
    try:
        obj['image']
    except:
        continue
    # if 'sam/images' in obj['image']:
    total_num += 1
    if os.path.isfile(root + obj['image']):
        exists_num += 1
    else:
        nonexists_num += 1
        nonexists_list.append(obj['image'])
    if 'boxes' in obj:
        exists_box_num += 1
    else:
        nonexists_box_num += 1

print("LBKLLaVA")
print(f'Total num of {len(object)}. Among them, instruction with images has {total_num}, but there are only {exists_num} and ther are not exists for {nonexists_num}')
print(exists_box_num)
print(nonexists_box_num)
print('ok')