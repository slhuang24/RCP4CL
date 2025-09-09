import argparse
import logging
import os
import pprint
import random
import sys
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataset.acdc import ACDCDataset
from models.unet import UNet
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.MRS import select_mixregion_u,select_mixregion_l


parser = argparse.ArgumentParser(
    description='Rethinking Copy-Paste for Consistency Learning in Medical Image Segmentation')

parser.add_argument('--config', default="configs/acdc.yaml", type=str)
parser.add_argument('--savemodel-path-root', type=str, default="./result/Unet/semi")
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--max_topk', default="3", type=int)
parser.add_argument('--weight_decay', default=0.0001, type=float)
parser.add_argument('--filter_foreground', type=list, default=[0.4])
parser.add_argument('--block_size', type=list, default=[32]) #This paper only adopts a single scale for region selection;
# in the future, this function can be used to further explore multi-scale region selection strategies.
parser.add_argument('--stride', type=list, default=[32])
parser.add_argument('--set',default="xxx",type=str)
parser.add_argument('--mu', default=0.5, type=float)
parser.add_argument("--acdc-labeled-id-path", type=str, default="./splits/acdc/7/labeled.txt")
parser.add_argument("--acdc-unlabeled-id-path", type=str, default="./splits/acdc/7/unlabeled.txt")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Logger(object):

    def __init__(self,args,stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(args.set)
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    args = parser.parse_args()
    if not os.path.exists(args.savemodel_path_root):
        os.makedirs(args.savemodel_path_root)
    model_name = '{}'.format(args.set)
    args.savemodel_path = os.path.join(args.savemodel_path_root, model_name)
    if not os.path.exists(args.savemodel_path):
        os.makedirs(args.savemodel_path)
    sys.stdout = Logger(args,sys.stdout)
    sys.stderr = Logger(args,sys.stderr)
    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.set_num_threads(2)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args), 'ngpus': "0"}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.savemodel_path)

    model = UNet(1,cfg['nclass'])
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=args.weight_decay)
    model.cuda()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])


    if cfg['dataset']== "acdc":
        trainset_u = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['crop_size'], args.acdc_unlabeled_id_path)
        trainset_l = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['crop_size'], args.acdc_labeled_id_path, nsample=len(trainset_u.ids))
        valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')

        train_l_dataloader = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=0,
                                   shuffle=True,drop_last=True)

        train_u_dataloader = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=0,
                                   shuffle=True,drop_last=True)
        val_dataloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)


    total_epochs=cfg['epochs']
    total_iters = len(train_u_dataloader) * total_epochs
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.savemodel_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.savemodel_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)




    for epoch in range(epoch+1, total_epochs):

        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()

        loader = zip(train_l_dataloader, train_u_dataloader)


        for i, ((img_x, mask_x),
                (img_u_w, img_u_s)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s = img_u_s.cuda()
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            mask_mix_u = select_mixregion_u(pred_u_w,max_topk=args.max_topk,block_size=args.block_size,filter_foreground=args.filter_foreground,stride=args.stride)
            mask_mix_l = select_mixregion_l(pred_x,mask_x.squeeze(1),max_topk=args.max_topk,block_size=args.block_size,filter_foreground=args.filter_foreground,stride=args.stride)

            img_s1, img_s2= img_x.clone(), img_u_s.clone()

            img_s1[mask_mix_u.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_w[mask_mix_u.unsqueeze(1).expand(img_u_s.shape) == 1]

            img_s2[mask_mix_l.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_x[mask_mix_l.unsqueeze(1).expand(img_u_s.shape) == 1]


            pred_u_s1, pred_u_s2= model(torch.cat((img_s1, img_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1).float()

            conf_x = torch.ones_like(conf_u_w)

            mask_x = mask_x.float()
            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_x.clone(), conf_x.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[mask_mix_u == 1] = mask_u_w[mask_mix_u == 1]
            conf_u_w_cutmixed1[mask_mix_u == 1] = conf_u_w[mask_mix_u == 1]

            mask_u_w_cutmixed2[mask_mix_l == 1] = mask_x[mask_mix_l == 1]
            conf_u_w_cutmixed2[mask_mix_l == 1] = conf_x[mask_mix_l == 1]

            loss_x = (criterion_ce(pred_x, mask_x.long()) + criterion_dice(pred_x.softmax(dim=1),
                                                                    mask_x.unsqueeze(1))) / 2.0

            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())

            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())


            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float())

            loss = (loss_x + loss_u_s1 * args.mu + 0.5 * args.mu * (loss_u_s2 + loss_u_w_fp)) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            iters = epoch * len(train_u_dataloader) + i

            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            if (i % (len(train_u_dataloader) // 8) == 0):
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f},Loss s1: {:.3f},Loss s2: {:.3f},Loss w_fp: {:.3f}'
                    .format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_s1.avg,
                                    total_loss_s2.avg,total_loss_w_fp.avg))

        model.eval()
        dice_class = [0] * 3
        with torch.no_grad():
            for img, mask in val_dataloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)
                img = img.permute(1, 0, 2, 3)
                pred = model(img)
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)
                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(val_dataloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        for (cls_idx, dice) in enumerate(dice_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                        '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))
        writer.add_scalar('eval/MeanDice', mean_dice, epoch)
        for i, dice in enumerate(dice_class):
            writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
        torch.save(checkpoint, os.path.join(args.savemodel_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.savemodel_path, 'best.pth'))




if __name__ == '__main__':
    main()
