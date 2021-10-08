import numpy as np
import torch
import os
import argparse
from config import cfg
import scipy.io
import torch.backends.cudnn as cudnn
from utils.eval_reid import eval_func
from utils.re_ranking_z import re_ranking


parser = argparse.ArgumentParser(description="Attribute PCB Re-ID Baseline")
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

result = scipy.io.loadmat(os.path.join(cfg.OUTPUT_DIR, 'pytorch_result.mat'))
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

feats = torch.cat((query_feature, gallery_feature), dim=0)
feat_norm = 'on'
num_query = query_feature.size(0)

if feat_norm == 'on':
    print("The test feature is normalized")
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
# query
qf = feats[:num_query]
q_pids = np.asarray(query_label)
q_camids = np.asarray(query_cam)
# gallery
gf = feats[num_query:]
g_pids = np.asarray(gallery_label)
g_camids = np.asarray(gallery_cam)
m, n = qf.shape[0], gf.shape[0]
distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
distmat.addmm_(1, -2, qf, gf.t())
# del gf
# del qf
# del feats
distmat = distmat.cpu().numpy()
cmc, mAP, mINP, all_ap = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
result = {'cmc': cmc, 'mAP': mAP, 'mINP': mINP, 'all_ap': all_ap}
scipy.io.savemat(os.path.join(cfg.OUTPUT_DIR, 'cmc_result.mat'), result)

# del result
# del distmat
print("test result")
print("mINP: {:.1%}".format(mINP))
print("mAP: {:.1%}".format(mAP))
for r in [1, 5, 10]:
    print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

print("Enter reranking")
distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
cmc, mAP, mINP, all_ap = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
result = {'cmc': cmc, 'mAP': mAP, 'mINP': mINP, 'all_ap': all_ap}
scipy.io.savemat(os.path.join(cfg.OUTPUT_DIR, 'reranking_cmc_result.mat'), result)

# del result
# del distmat
print("reranking test result")
print("mINP: {:.1%}".format(mINP))
print("mAP: {:.1%}".format(mAP))
for r in [1, 5, 10]:
    print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
