# encoding: utf-8

import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from .datasets import init_dataset, ImageDataset
from .triplet_sampler import RandomIdentitySampler
from .transforms import build_transforms


def train_collate_fn(batch):
    imgs, pids, _, _, attrid = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    attrid = torch.from_numpy(np.asarray(attrid, dtype=np.float32))
    return torch.stack(imgs, dim=0), pids, attrid


def val_collate_fn(batch):
    imgs, pids, camids, _, attrid = zip(*batch)
    attrid = torch.from_numpy(np.asarray(attrid, dtype=np.float32))
    return torch.stack(imgs, dim=0), pids, camids, attrid


def make_data_loader(cfg):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset(dataset.train, transforms['train'])
    data_loader={}
    if cfg.DATALOADER.PK_SAMPLER == 'on':
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    query_set = ImageDataset(dataset.query, transforms['eval'])
    data_loader['query'] = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    gallery_set = ImageDataset(dataset.gallery, transforms['eval'])
    data_loader['gallery'] = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    datasets = {}
    datasets['train'] = train_set
    datasets['query'] = query_set
    datasets['gallery'] = gallery_set

    return data_loader, datasets, num_classes
