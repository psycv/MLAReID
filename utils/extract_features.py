import torch
from torch.autograd import Variable
import math
from torch import nn
from tqdm import tqdm


# def extract_features(feat, opt):
#     n = feat[0].size(0)
#     f = torch.FloatTensor(n, opt.test_dim).zero_()
#     features = torch.FloatTensor()
#     for i in range(2):
#         feat_cat = torch.cat([f for f in feat], 1)
#         feat_cat_cpu = feat_cat.data.cpu()
#         feat_cat_cpu = feat_cat_cpu + f
#     fnorm = torch.norm(feat_cat_cpu, p=2, dim=1, keepdim=True)
#     feat_cat_cpu = feat_cat_cpu.div(fnorm.expand_as(feat_cat_cpu))
#     features = torch.cat((features, feat_cat_cpu), 0)
#     return features

ms = [1]

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(cfg, model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    with tqdm(dataloaders, ascii=True) as tq:
        for data in tq:
            # import ipdb
            # ipdb.set_trace()
            img, label, _, _ = data
            n, c, h, w = img.size()
            count += n
            # print(count)
            ff = torch.FloatTensor(n, cfg.TEST.FEAT_DIM).zero_().cuda()

            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                    outputs = model(input_img) 
                    ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff.data.cpu()), 0)
        return features
