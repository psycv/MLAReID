###################################3
# Visualize HearMap by sum
# Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2018): 13.
###################################

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from modeling import build_model
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from config import cfg
from data import make_data_loader

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description="Attribute PCB Re-ID Baseline")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
# parser.add_argument('--data_dir',default='/root/code/dataset/market1501/pytorch',type=str, help='./test_data')

parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

# args.config_file = "./configs/DGC_baseline-l4_resnet101_ibn_a-384_128-all_tricks.yml"
if args.config_file != "":
    cfg.merge_from_file(args.config_file)
# cfg.SOLVER.IMS_PER_BATCH = 1
cfg.merge_from_list(args.opts)
cfg.freeze()
# args.data_dir = os.path.join(cfg.DATASETS.ROOT_DIR, cfg.DATASETS.NAMES, 'images_detected/pytorch')

def vis_heatmap(img, arr, name):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title='ID:'+str(lbl))
    ax1 = fig.add_subplot(122, title="Heatmap")
    im = Image.open(img)
    im = im.resize((200, 600))
    ax0.imshow(im)
    heatmap = ax1.imshow(arr, cmap='viridis')
    fig.colorbar(heatmap)
    #plt.show()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, name+'.eps'))
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, name+'.png'), dpi=512)


# data_transforms = transforms.Compose([
#         transforms.Resize(cfg.INPUT.IMG_SIZE, interpolation=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir,x) ,data_transforms) for x in ['train']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
#                                              shuffle=True, num_workers=1) for x in ['train']}

# imgpath = image_datasets['train'].imgs


# _, _, num_classes, _ = make_data_loader(cfg)
i = args.query_index

# _, _, num_classes, image_datasets = make_data_loader(cfg)
dataloaders, image_datasets, num_classes = make_data_loader(cfg)

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=1) for x in ['train']}
# _, _, _, imgpath = image_datasets['train'][i]
# imgpath = image_datasets['train'][3]

model = build_model(cfg, num_classes)
model = model.cuda()
model.load_param(cfg.TEST.WEIGHT)
model = model.eval()
print('model loaded!')
# import ipdb; ipdb.set_trace()
# data = next(iter(dataloaders['train']))

img, lbl, _, query_path = image_datasets['query'][i]
img = torch.unsqueeze(img, 0)
print(query_path)
# img, label, _, imgpath = data
with torch.no_grad():
    x = model.base.conv1(img.cuda())
    x = model.base.bn1(x)
    # x = model.base.relu(x)
    x = model.base.maxpool(x)
    x = model.base.layer1(x)
    x = model.base.layer2(x)
    output = model.base.layer3(x)
    output1 = model.base.layer4(output)

# print(output.shape, output1.shape)
heatmap = output.squeeze().sum(dim=0).cpu().numpy()
heatmap1 = output1.squeeze().sum(dim=0).cpu().numpy()
# print(heatmap.shape, heatmap1.shape)
#test_array = np.arange(100 * 100).reshape(100, 100)
# Result is saved tas `heatmap.png`
# print(imgpath)
vis_heatmap(query_path, heatmap, 'heatmap_l3_{}'.format(lbl))
vis_heatmap(query_path, heatmap1, 'heatmap_l4_{}'.format(lbl))