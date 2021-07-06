# -*- coding: utf-8 -*-

from __future__ import print_function, division

from config import cfg
import torch
import torch.backends.cudnn as cudnn
import os
import scipy.io
from utils.extract_features import extract_feature
from utils.eval_reid import eval_func
from utils.re_ranking_z import re_ranking
import numpy as np
import argparse
from utils.logger import setup_logger
from data import make_data_loader
from modeling import build_model
######################################################################
# Options
# --------

def do_test(model, cfg, data_loader, datasets, logger):
    model.eval()
    with torch.no_grad():
        gallery_feature = extract_feature(cfg, model, data_loader['gallery'])
        query_feature = extract_feature(cfg, model, data_loader['query'])
    query_cam = []
    query_label = []
    for data in datasets['query']:
        # import ipdb
        # ipdb.set_trace()
        img, pid, camid, img_path, _ = data
        query_cam.append(camid)
        query_label.append(pid)

    gallery_cam = []
    gallery_label = []
    for data in datasets['gallery']:
        img, pid, camid, img_path, _ = data
        gallery_cam.append(camid)
        gallery_label.append(pid)
    
    result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
    scipy.io.savemat(os.path.join(cfg.OUTPUT_DIR, 'pytorch_result.mat'), result)

    feats = torch.cat((query_feature, gallery_feature), dim=0)
    feat_norm = 'on'
    num_query = query_feature.size(0)

    if feat_norm == 'on':
        logger.info("The test feature is normalized")
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
    # result = {'cmc': cmc, 'mAP': mAP, 'mINP': mINP, 'all_ap': all_ap}
    # scipy.io.savemat(os.path.join(cfg.OUTPUT_DIR, 'cmc_result.mat'), result)

    # del result
    # del distmat
    logger.info("test result")
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if cfg.TEST.RE_RANKING == 'on':
        logger.info("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        re_cmc, re_mAP, re_mINP, re_all_ap = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # result = {'re_cmc': re_cmc, 're_mAP': re_mAP, 're_mINP': re_mINP, 're_all_ap': re_all_ap}
        # scipy.io.savemat(os.path.join(cfg.OUTPUT_DIR, 'reranking_cmc_result.mat'), result)

        # del result
        # del distmat
        logger.info("reranking test result")
        logger.info("mINP: {:.1%}".format(re_mINP))
        logger.info("mAP: {:.1%}".format(re_mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, re_cmc[r - 1]))

        return cmc, mAP, mINP, all_ap, re_cmc, re_mAP, re_mINP, re_all_ap
    return cmc, mAP, mINP, all_ap


if __name__ == "__main__":
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

    logger = setup_logger("Attribute_reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            # logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    data_loader, datasets, num_classes = make_data_loader(cfg)
    num_attrs = (30 if cfg.DATASETS.NAMES == 'market1501' else 23)
    print(num_attrs)
    model = build_model(cfg, num_attrs, num_classes)

    if 'cpu' not in cfg.MODEL.DEVICE:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device=cfg.MODEL.DEVICE)

    if cfg.TEST.EVALUATE_ONLY == 'on':
        logger.info("Evaluate Only")
        model.load_param(cfg.TEST.WEIGHT)
    do_test(model, cfg, data_loader, datasets, logger)
