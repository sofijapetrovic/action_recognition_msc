import argparse
import json

import cv2
import numpy as np
from utils.setup_logging import setup_logging
import logging
from dataset.video_dataset import VideoDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.action_network import ActionModel

VALIDATION_PATH = '/media/data/pose_data_3/validation'


def plot_confusion_matrix(cm):
    image = np.ones((224,224, 3)) * 1
    cv2.line(image, (0, 112), (223, 112), color=(0, 0, 0))
    cv2.line(image, (112, 0), (112, 223), color=(0, 0, 0))
    cv2.putText(image, str(cm[0, 0]), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, str(cm[0, 1]), (164, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, str(cm[1, 0]), (50, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, str(cm[1, 1]), (164, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    image = np.transpose(image, (2, 0, 1))
    return image


def calculate_metrics(gt_class_ids, pred_class_ids):
    drop_drop = len([class_id for j, class_id in enumerate(gt_class_ids) if int(class_id) == 1
                 and int(gt_class_ids[j]) == int(pred_class_ids[j])])
    drop_nodrop = len([class_id for j, class_id in enumerate(gt_class_ids) if int(class_id) == 1
                   and int(gt_class_ids[j]) != int(pred_class_ids[j])])
    nodrop_drop = len([class_id for j, class_id in enumerate(gt_class_ids) if int(class_id) == 0
                   and int(gt_class_ids[j]) != int(pred_class_ids[j])])
    nodrop_nodrop = len([class_id for j, class_id in enumerate(gt_class_ids) if int(class_id) == 0
                     and int(gt_class_ids[j]) == int(pred_class_ids[j])])
    confusion = np.array([[drop_drop, drop_nodrop], [nodrop_drop, nodrop_nodrop]])
    acc = (drop_drop + nodrop_nodrop) / len(gt_class_ids)
    return acc, confusion


def eval(batch_size=32, pose_checkpoint=None, appearance_checkpoint=None, net=None, print_logs=True):
    '''Load dateset'''
    data = VideoDataset(VALIDATION_PATH)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=10)

    if net is None and pose_checkpoint and appearance_checkpoint:
        net = ActionModel()
        net.pose_action_network.load_state_dict(torch.load(pose_checkpoint))
        net.pose_action_network.eval()
        net.appearance_action_network.load_state_dict(torch.load(appearance_checkpoint))
        net.pose_action_network.eval()

    gt_class_ids, pose_class_ids, app_class_ids, combined_class_ids = [], [], [], []
    for i_batch, batch in tqdm(enumerate(dataloader)):
        pose_features, app_features, act_class = batch[0], batch[1], batch[2]
        pose_features = torch.transpose(pose_features, -1, 1)
        app_features = torch.transpose(app_features, -1, 1)
        gt_class_ids += list(act_class.cpu().detach())

        app_out1, app_out = net.appearance_action_network(app_features.cuda())
        pose_out1, pose_out = net.pose_action_network(pose_features.cuda())

        app_out = torch.squeeze(app_out)
        pose_out = torch.squeeze(pose_out)

        pose_class_ids += list(torch.argmax(pose_out, dim=-1).cpu().detach())
        app_class_ids += list(torch.argmax(app_out, dim=-1).cpu().detach())
        out = (pose_out + app_out) / 2
        combined_class_ids += list(torch.argmax(out, dim=-1).cpu().detach())

    pose_acc, _ = calculate_metrics(gt_class_ids, pose_class_ids)
    app_acc, _ = calculate_metrics(gt_class_ids, app_class_ids)
    combined_acc, confusion = calculate_metrics(gt_class_ids, combined_class_ids)
    if print_logs:
        logging.info(f'total number of samples: {len(gt_class_ids)}')
        logging.info(f'pose network accuracy: {pose_acc} \nappearance network accuracy {app_acc} '
                     f'\ncombined network accuracy {combined_acc}')
        logging.info(confusion[0, :])
        logging.info(confusion[1, :])
    return pose_acc, app_acc, combined_acc, confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--config-path', type=str, default='action_config.json', help='Path to the training config')
    parser.add_argument('--appearance-checkpoint-path', type=str, default='',
                        help='Path to the checkpoint')
    parser.add_argument('--pose-checkpoint-path', type=str, default='', help='Path to the checkpoint')

    setup_logging()
    args = parser.parse_args()
    config = json.load(open(args.config_path, 'r'))
    eval(config['batch_size'], args.pose_checkpoint_path, args.appearance_checkpoint_path)

