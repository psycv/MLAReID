import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data import make_data_loader
from config import cfg
from PIL import Image
from torchvision import transforms as T

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description="Attribute PCB Re-ID")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

# args.config_file = "./configs/DGC_baseline-l4_resnet101_ibn_a-384_128-all_tricks.yml"
if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()


# dataloaders, image_datasets, num_query, num_classes, dataset_sizes = make_data_loader(opt)
# dataloaders, num_query, num_classes, image_datasets = make_data_loader(cfg)
dataloaders, image_datasets, num_classes = make_data_loader(cfg)

#######################################################################
# Evaluate
# parser = argparse.ArgumentParser(description='Demo')
# parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
# parser.add_argument('--test_dir',default='/root/code/dataset/market1501/pytorch',type=str, help='./test_data')

# opts = parser.parse_args()

# data_dir = opts.test_dir
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x)) for x in ['gallery','query']}

#####################################################################
# def pinjie(paths):
#     im_list = [Image.open(p) for p in paths]
#     ims = []
#     for i in im_list:
#         new_img = i.resize((20, 80), Image.BILINEAR)
#         ims.append(new_img)
#     width, height = ims[0].size
#     result = Image.new(ims[0].mode, (width*len(ims), height))
#     for i, im in enumerate(ims):
#         result.paste(im, box=(i * height, 0))
#     result.save('')    

# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    # im = plt.imread(path)
    im = Image.open(path)
    im = im.resize((200, 600))
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat(os.path.join(cfg.OUTPUT_DIR, 'pytorch_result.mat'))
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile(os.path.join(cfg.OUTPUT_DIR, 'multi_query.mat'))

if multi:
    m_result = scipy.io.loadmat(os.path.join(cfg.OUTPUT_DIR, 'multi_query.mat'))
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = args.query_index
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result

# query_path, _ = image_datasets['query'].imgs[i]
# import ipdb; ipdb.set_trace()

_, lbl, _, query_path = image_datasets['query'][i]
# save xls
'''
import xlwt
book = xlwt.Workbook()
sheet_label = book.add_sheet('index_label')
sheet_label.write(0, 0, 'index')
sheet_label.write(0, 1, 'label')
sheet_label.write(0, 2, 'path')
for id in range(len(image_datasets['query'])):
    sheet_label.write(id+1, 0, str(id))
    sheet_label.write(id+1, 1, str(image_datasets['query'][id][1]))
    sheet_label.write(id+1, 2, image_datasets['query'][id][3])
book.save("{}_query_index_label.xls".format(cfg.DATASETS.NAMES))
'''
#
query_label = query_label[i]
# print('query_label', query_label, lbl)
print(query_path)
print('Top 10 images are as follow:')
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure()
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path, 'ID:'+str(query_label))
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        # img_path, _ = image_datasets['gallery'].imgs[index[i]]
        _, lbl, _, img_path = image_datasets['gallery'][index[i]]
        # print('gallery {}, {}'.format(i, lbl))
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        # img_path = image_datasets.imgs[index[i]]
        img_path = image_datasets[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig(os.path.join(cfg.OUTPUT_DIR, "demo_{}.eps").format(lbl))
fig.savefig(os.path.join(cfg.OUTPUT_DIR, "demo_{}.png").format(lbl), dpi=512)