import argparse
import logging
import os
import pprint
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from Metrics.performance_metrics import calculate_metric_single_ACDC
from dataset.acdc import ACDCDataset
from models.unet import UNet
from util.classes import CLASSES
from util.utils import count_params, init_log


parser = argparse.ArgumentParser()

parser.add_argument('--config', default="configs/acdc.yaml", type=str)
parser.add_argument('--savemodel-path', type=str, default="./result/Unet/semi/epoch300,3")

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
    model = UNet(1,cfg['nclass'])
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model.cuda()

    testset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'test')
    test_dataloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)

    if os.path.exists(os.path.join(args.savemodel_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.savemodel_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])

        model.eval()
        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        with torch.no_grad():
            for img, mask in test_dataloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)
                img = img.permute(1, 0, 2, 3)

                pred = model(img)
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                first_metric,second_metric,third_metric=calculate_metric_single_ACDC(pred,mask)
                first_total += np.asarray(first_metric)
                second_total += np.asarray(second_metric)
                third_total += np.asarray(third_metric)

        avg_metric = [first_total / len(test_dataloader), second_total / len(test_dataloader), third_total / len(test_dataloader)]
        avg_types_metric=(avg_metric[0]+avg_metric[1]+avg_metric[2])/3

        for (cls_idx,_ )in enumerate(avg_metric):
            logger.info('***** Test ***** >>>> Class [{:} {:}] Dice: '
                        '{:.2f},>>JAC: {:.2f},>>95HD: {:.2f},>>ASD: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], avg_metric[cls_idx][0],avg_metric[cls_idx][1],
                                                                            avg_metric[cls_idx][2],avg_metric[cls_idx][3]))
        logger.info('***** Test ***** >>>> MeanDice: {:.2f},>>>MeanJAC: {:.2f},>>>Mean95HD: {:.2f},>>>MeanASD: {:.2f}\n'.format(avg_types_metric[0],avg_types_metric[1],
                                                                                                                       avg_types_metric[2],avg_types_metric[3]))




if __name__ == '__main__':
    main()
