ROOT_PATH = "/mnt/hard/lbk-cvpr/checkpoints/"
BAKLLAVA_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--llava-hf--bakLlava-v1-hf/snapshots/f038f156966ff4d24078b260e9e9761fd480d325"
LLAVA_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/6b7135519bd7a7f93a03c1f8ddae0ce9dfa1a7af"
KOSMOS2_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--microsoft--kosmos-2-patch14-224/snapshots/e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c"
CLIPLARGE_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
BLIP2_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/e5025a34e3e769e72e2aab7f7bfd00bc84d5fd77"
INSTRUCTBLIP_LOCAL_PATH = ROOT_PATH + "huggingface/hub/models--Salesforce--instructblip-vicuna-7b/snapshots/ef9d8b3bcb7a0422d7b33a8917e867944312ef22"
LLAMA2_LOCAL_PATH = ROOT_PATH + "llama-2-7b_hf"


# Color List
color_list = ['white',
            'red', 
            'orange', 
            'coral', 
            'yellow', 
            'green', 
            'blue', 
            'navy', 
            'gold',
            'pink', 
            'purple', 
            'brown', 
            'violet', 
            'olive', 
            'lime', 
            'cyan',
            'magenta',
            'silver', 
            'gray', 
            'black']

# color_list = ['white',
#             'red', 
#             'orange', 
#             'yellow', 
#             'green', 
#             'blue', 
#             'pink', 
#             'purple', 
#             'brown', 
#             'black']

def list2string(_list):
    out = ''
    for i, x in enumerate(_list):
        out+=str(x)
        if i!=len(_list)-1: out+=', '
    out += ''
    return out

def box2string(box):
    out = '['
    for i, x in enumerate(box):
        out+=f"{round(x.item(), 3):.3f}"
        if i!=len(box)-1: out+=', '
    out += ']'
    return out

def boxes2string(boxes):
    out = ''
    for i, x in enumerate(boxes):
        out+=box2string(x)
        if i!=len(boxes)-1: out+=', '
    return out

def classescolors2string(classes, colors, class_num=1000):
    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, colors)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"{y} ({x} #{count[x]})"
        if i!=len(classes)-1: out+=', '
    return out


def classesboxes2string(classes, boxes, class_num=1000):
    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, boxes)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"{box2string(y)} ({x} #{count[x]})"
        if i!=len(classes)-1: out+=', '
    return out

def classes2string(classes):
    count = {}
    out = ''
    for i, x in enumerate(classes):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"{x} (#{count[x]})"
        if i!=len(classes)-1: out+=', '
    return out