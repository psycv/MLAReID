# encoding: utf-8

import torch
from torch import nn
from torch.nn import init
import collections
from .backbones.resnet import ResNet, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnet_nl import ResNetNL
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, GeneralizedMeanPooling, GeneralizedMeanPoolingP


def weights_init_kaiming2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

def weights_init_kaiming_V3(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier_V3(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)

class ClassBlock_V2(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock_V2, self).__init__()
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming_V3)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming_V3)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier_V3)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block1(x)
        x = self.add_block2(x)
        y = self.classifier(x)
        return x, y

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_attrs, num_classes, last_stride, model_path, model_name, gem_pool, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(last_stride)
        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.num_classes = num_classes

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.num_attrs = num_attrs
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier_g = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming2)
        self.classifier_g.apply(weights_init_classifier2)

        for c in range(self.num_classes):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.in_planes, class_num=1, activ='sigmoid') )

        self.part = 6

        self.global_pool_V2 = nn.AdaptiveAvgPool2d((self.part, 1))

        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock_V2(self.in_planes, self.num_classes, num_bottleneck=512))

    def forward(self, x):
        x = self.base(x)

        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # import ipdb; ipdb.set_trace()
        pred_label = [self.__getattr__('class_%d' % c)(feat) for c in range(self.num_attrs)]
        pred_label = torch.cat(pred_label, dim=1)

        x = self.global_pool_V2(x)
        feat_V2 = []
        cls_score = []
        feat_block = []
        for i in range(self.part):
            tmp = x[:,:,i,:]
            tmp = tmp.view(tmp.size(0), -1)
            feat_V2.append(tmp)
            tmp_feat, tmp_cls = self.classifiers[i](tmp)
            feat_block.append(tmp_feat)
            cls_score.append(tmp_cls)

        if not self.training:
            feat_block = torch.cat(feat_block, dim=1)
            feat_block = torch.cat((feat_block, feat), dim=1)
            return feat_block

        cls_score_g = self.classifier_g(feat)
        return cls_score_g, cls_score, pred_label, global_feat, feat_V2

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if not isinstance(param_dict, collections.OrderedDict):
            param_dict = param_dict.state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)
        criterion['bce'] = nn.BCELoss()

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_total(score_g, score, attr_score, feat_g, feat, target, attr_target):
            # import ipdb; ipdb.set_trace()
            # loss = []
            loss_xent = [criterion['xent'](s, target) for s in score]
            loss_xent = sum(loss_xent) / len(loss_xent)
            loss_trip = [criterion['triplet'](f, target)[0] for f in feat]
            loss_trip = sum(loss_trip) / len(loss_trip)
            loss_bce = criterion['bce'](attr_score, attr_target)
            loss_xent_g = criterion['xent'](score_g, target)
            loss_trip_g = criterion['triplet'](feat_g, target)[0]
            loss = [loss_xent, loss_xent_g, loss_trip, loss_trip_g, loss_bce]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss_cent = [criterion['center'](f, target) for f in feat]
                loss_cent = sum(loss_cent) / len(loss_cent)
                loss_cent_g = criterion['center'](feat_g, target)
                loss.append(loss_cent)
                loss.append(loss_cent_g)
            return loss

        criterion['total'] = criterion_total

        return criterion

