import torch
import cv2
from torch.utils.data import DataLoader
import os
from dataset import PoseDataset
import argparse
import matplotlib.pyplot as plt
from estimator import Estimator
import logging
from utils import setup_logging
from tqdm import tqdm
import numpy as np

ANNOTATIONS_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/annotations/hand_detection/' \
                   'part1_01_01_2020/v3_val_3clients_sco_part1_01_01_2020_24200.json'
IMAGES_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/images/part1/'
BATCH_SIZE = 16


def process_gt(gt_peaks):
    out_gt = []
    gt_peaks = gt_peaks.cpu().numpy()
    for i in range(gt_peaks.shape[0]):
        left = gt_peaks[i, :, 0, :]
        last_ind = np.where(left == -1)
        left = left[:last_ind[0][0], :]
        right = gt_peaks[i, :, 1, :]
        last_ind = np.where(right == -1)
        right = right[:last_ind[0][0], :]

        out_gt.append([left, right])
    return out_gt

def dist(pos1,pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def match_predictions(pred_peaks, gt_peaks, confidence):
    results = {'40':{'left':[],'right':[]},'80':{'left':[],'right':[]},'120':{'left':[],'right':[]}}
    sides = ['left', 'right']
    thresholds = [40, 80, 120]
    total_gt = {'left':0,'right':0}
    for img_ind in range(len(pred_peaks)):
        for threshold in thresholds:
            for i, gt_side in enumerate(gt_peaks[img_ind]):
                total_gt[sides[i]] += gt_side.shape[0]
                used = np.zeros(pred_peaks[img_ind][i].shape[0])
                for j in range(gt_side.shape[0]):
                    dists = np.zeros(pred_peaks[img_ind][i].shape[0])
                    for k in range(pred_peaks[img_ind][i].shape[0]):
                        dists[k] = dist(pred_peaks[img_ind][i][k], gt_side[j])
                    min_ind = np.argsort(dists)
                    for sorted_ind in min_ind:
                        if used[min_ind[sorted_ind]] == 0 and dists[min_ind[sorted_ind]] <= threshold:
                            results[str(threshold)][sides[i]].append((True,confidence[img_ind][i][min_ind[sorted_ind]]))
                            used[min_ind[sorted_ind]] = 1
                            break
                for j,use in enumerate(used):
                    if use == 0:
                        results[str(threshold)][sides[i]].append((False, confidence[img_ind][i][j]))
    for key in total_gt.keys():
        total_gt[key] /= 3
    return results, total_gt

def calculate_mAP(results_dict, total_gt):
    mAP = {'40':0, '80':0, '120':0}
    for dist in results_dict.keys():
        for side in results_dict[dist].keys():
            tuples = results_dict[dist][side]
            tuples = sorted(tuples, key=lambda tup: tup[1],reverse=True)
            prec, rec = np.zeros(len(tuples)), np.zeros(len(tuples))
            ap=0
            for i in range(len(prec)):
                rec[i] = rec[i-1] if i>0 else 0
                rec[i] += 1/total_gt[side] if tuples[i][0] else 0
                prev_tp = len([t for t in tuples[:i+1] if t[0]])
                prec[i] = prev_tp/(i+1)
            for i in range(len(prec)-1):
                ap += (rec[i+1]-rec[i]) * np.max(prec[1+1:])
            mAP[dist] += ap
        mAP[dist] = mAP[dist]/2

    print(mAP)
    return mAP


def eval(checkpoint_path, annotations_path):
    if not annotations_path:
        logging.info('Using default annotations path')
        annotations_path = ANNOTATIONS_PATH

    if not os.path.exists(checkpoint_path):
        logging.error('Checkpoint does not exist!')
        exit(1)
    model = torch.load(checkpoint_path)
    logging.info('Loaded model checkpoint')

    dataset = PoseDataset(path=IMAGES_PATH)
    dataset.load_annotations_from_json(annotations_path)
    logging.info(f'Loaded {dataset.ann_count} annotations')

    estimator = Estimator(model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=10)
    total_results = {'40':{'left':[],'right':[]},'80':{'left':[],'right':[]},'120':{'left':[],'right':[]}}
    total_gt = {'left':0, 'right':0}
    for i,batch in tqdm(enumerate(dataloader)):
        images, _, _, gt_peaks = batch[0], batch[1], batch[2], batch[3]
        pred_peaks, confidences = estimator.inference(images)
        gt_peaks = process_gt(gt_peaks)
        batch_results, batch_gt = match_predictions(pred_peaks, gt_peaks, confidences)

        for key1 in batch_results.keys():
            for key2 in batch_results[key1].keys():
                total_results[key1][key2] += batch_results[key1][key2]

        for key1 in batch_gt.keys():
            total_gt[key1] += batch_gt[key1]
    calculate_mAP(total_results, total_gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--annotations-path', type=str, default=None, help='Path to the annotation json')

    args = parser.parse_args()
    setup_logging()
    eval(args.checkpoint_path, args.annotations_path)













