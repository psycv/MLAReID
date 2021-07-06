import scipy.io
import torch
import numpy as np
import time
from utils.re_ranking import re_ranking
import os
from config import cfg
import argparse
from utils.logger import setup_logger
import torch.backends.cudnn as cudnn



parser = argparse.ArgumentParser(description="Attribute PCB Re-ID")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# logger = setup_logger("reid_baseline", output_dir, 0)
# logger.info("Using {} GPUS".format(num_gpus))
# logger.info(args)

# if args.config_file != "":
#     logger.info("Loaded configuration file {}".format(args.config_file))
#     with open(args.config_file, 'r') as cf:
#         config_str = "\n" + cf.read()
#         logger.info(config_str)
# logger.info("Running with config:\n{}".format(cfg))

if cfg.MODEL.DEVICE == "cuda":
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
cudnn.benchmark = True
#######################################################################
# Evaluate
def evaluate(score, ql, qc, gl, gc):
    index = np.argsort(score)  # from small to large
    # index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat(os.path.join(cfg.OUTPUT_DIR, 'pytorch_result.mat'))
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

# re-ranking
print('calculate initial distance')
q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
q_q_dist = np.dot(query_feature, np.transpose(query_feature))
g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))
# import ipdb; ipdb.set_trace()
since = time.time()
re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)
time_elapsed = time.time() - since
print('Reranking complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(re_rank[i, :], query_label[i], query_cam[i], gallery_label, gallery_cam)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

doc = open(os.path.join(cfg.OUTPUT_DIR, 'rerank_result.txt'), 'a')
print('Reranking Evaluate Dataset:%s time:%s\n Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cfg.DATASETS.NAMES, time.strftime('%Y-%m-%d %H-%M-%S'), CMC[0], CMC[4], CMC[9], ap / len(query_label)), file=doc)
doc.close()