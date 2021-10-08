import sys
sys.path.append('.')
from config import cfg
from data import make_data_loader
from modeling import build_model
from utils.lr_scheduler import WarmupMultiStepLR
import argparse
import os
from torch.backends import cudnn
from utils.logger import setup_logger
import torch
from tqdm import tqdm
import torch.nn as nn
import time
import scipy.io
import matplotlib.pyplot as plt
from modeling.layer.triplet_loss import CrossEntropyLabelSmooth, TripletLoss
from test import do_test



# do_train(cfg,
#     model,
#     data_loader,
#     optimizer,
#     scheduler,
#     criterion,
#     num_query,
#     start_epoch
# )

def do_train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0
        # Iterate over data.
        with tqdm(data_loader['train'], ascii=True) as tq:
            for inputs, labels, attrs_labels in tq:
                # get the inputs
                # inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < cfg.SOLVER.IMS_PER_BATCH:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                inputs = inputs.cuda()
                labels = labels.cuda()
                attrs_labels = attrs_labels.cuda()
                # zero the parameter gradients
                optimizer['model'].zero_grad()

                # forward
                outputs_g, outputs, attrs_outputs, features_g, features = model(inputs)

                acc_list = [(output.max(1)[1] == labels).float().mean() for output in outputs]
                acc_list.append((outputs_g.max(1)[1] == labels).float().mean())
                acc = sum(acc_list) / len(acc_list)
                loss_list = criterion['total'](outputs_g, outputs, attrs_outputs, features_g, features, labels, attrs_labels)
                if cfg.MODEL.CENTER_LOSS == 'on':
                    loss = sum(loss_list[:2]) + sum(loss_list[2:5]) + cfg.SOLVER.CENTER_LOSS_WEIGHT * sum(loss_list[5:])
                else:
                    loss = sum(loss_list[:2]) + sum(loss_list[2:])
                loss.backward()
                optimizer['model'].step()

            if cfg.MODEL.CENTER_LOSS == 'on':
                logger.info('Loss: {:.2f}({:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}) Acc: {:.2f}({:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}) Lr: {}'.format(
                    loss, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], 
                    cfg.SOLVER.CENTER_LOSS_WEIGHT * loss_list[5], cfg.SOLVER.CENTER_LOSS_WEIGHT * loss_list[6], 
                    acc, acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4], acc_list[5], acc_list[6], scheduler.get_lr()[0]))
            else:
                logger.info('Loss: {:.2f}({:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}) Acc: {:.2f}({:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}) Lr: {}'.format(
                    loss, loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], 
                    acc, acc_list[0], acc_list[1], acc_list[2], acc_list[3], acc_list[4], acc_list[5], acc_list[6], scheduler.get_lr()[0]))
            y_loss.append(loss.cpu().data)
            y_err.append(1.0 - acc)
            # deep copy the model
            last_model_wts = model.state_dict()
            if epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                save_network(model, epoch)
            # draw_curve(epoch)

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            if cfg.TEST.RE_RANKING == 'on':
                cmc, mAP, mINP, all_ap, re_cmc, re_mAP, re_mINP, re_all_ap = do_test(model, cfg, data_loader, datasets, logger)
            else:
                cmc, mAP, mINP, all_ap = do_test(model, cfg, data_loader, datasets, logger)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    if cfg.TEST.RE_RANKING == 'on':
        result = {'error': y_err, 'loss': y_loss, 'cmc':cmc, 'mAP':mAP, 'mINP':mINP, 'all_ap':all_ap, 're_cmc':re_cmc, 're_mAP':re_mAP, 're_mINP':re_mINP, 're_all_ap':re_all_ap}
        file = os.path.join(cfg.OUTPUT_DIR, 're_log.mat')
    else:
        result = {'error': y_err, 'loss': y_loss, 'cmc':cmc, 'mAP':mAP, 'mINP':mINP, 'all_ap':all_ap}
        file = os.path.join(cfg.OUTPUT_DIR, 'log.mat')
    scipy.io.savemat(file, result)


# def draw_curve(current_epoch):
#     x_epoch.append(current_epoch)
#     ax0.plot(x_epoch, y_loss, 'bo-', label='train')
#     ax1.plot(x_epoch, y_err, 'bo-', label='train')
#     if current_epoch == 0:
#         ax0.legend()
#         ax1.legend()
#     fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(cfg.OUTPUT_DIR, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()





if __name__ == '__main__':

    version = torch.__version__
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

    logger = setup_logger("Attribute_PCB", output_dir, 0)
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
        # do_test(cfg, model, data_loader, num_query)

    criterion = model.get_creterion(cfg, num_classes)
    optimizer = model.get_optimizer(cfg, criterion)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
        optimizer['model'].load_state_dict(torch.load(path_to_optimizer))
        criterion['center'].load_state_dict(torch.load(path_to_center_param))
        optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    y_loss = []
    y_err = []
    do_train(model, criterion, optimizer, scheduler, num_epochs=cfg.SOLVER.MAX_EPOCHS)
