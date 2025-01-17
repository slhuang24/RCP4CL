from medpy import metric
import numpy as np


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice*100.0, jc*100.0, hd95, asd

def calculate_metric_single_ACDC(pred, mask):
    prediction_m = pred.cpu().detach().numpy()
    mask_m = mask.cpu().detach().numpy()
    if np.sum(prediction_m == 1) == 0:
        first_metric = 0, 0, 0, 0
    else:
        first_metric = calculate_metric_percase(prediction_m == 1, mask_m == 1)

    if np.sum(prediction_m == 2) == 0:
        second_metric = 0, 0, 0, 0
    else:
        second_metric = calculate_metric_percase(prediction_m == 2, mask_m == 2)

    if np.sum(prediction_m == 3) == 0:
        third_metric = 0, 0, 0, 0
    else:
        third_metric = calculate_metric_percase(prediction_m == 3, mask_m == 3)

    return first_metric,second_metric,third_metric




