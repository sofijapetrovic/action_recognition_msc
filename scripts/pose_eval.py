from torch.utils.data import DataLoader
import os
from dataset.pose_dataset import PoseDataset
import argparse
from estimator import SkeletonEstimator
from skeleton.skeletons import pair_joints
from skeleton.annotated_skeletons import AnnotatedSkeletons
import logging
from utils.setup_logging import setup_logging
from tqdm import tqdm
import numpy as np

ANNOTATIONS_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/annotations/hand_detection/' \
                   'part1_01_01_2020/v3_valid_3clients_sco_part1_01_01_2020_24200.json'
IMAGES_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/images/part1/'
BATCH_SIZE = 16


def calculate_mAP(dists_dict, thresholds=(40, 80, 120)):
    threshold_ap = {}
    for tr in thresholds:
        joint_ap = {}
        for joint_type in dists_dict:
            conf_list = []
            for dist, confidence in dists_dict[joint_type]:
                conf_list.append((dist <= tr, confidence))
            conf_list.sort(key=lambda tup: tup[1], reverse=True)
            prec, rec = np.zeros(len(conf_list)), np.zeros(len(conf_list))
            running_tp = 0
            total_tp = len([e for e in conf_list if e[0]])
            for i in range(len(conf_list)):
                running_tp += 1 if conf_list[i][0] else 0
                prec[i] = running_tp / (i + 1)
                rec[i] = running_tp / total_tp
            ap = 0
            for i in range(len(prec) - 1):
                ap += (rec[i + 1] - rec[i]) * np.max(prec[i + 1:])
            joint_ap[joint_type] = ap
            print(f'threshold: {tr} class: {joint_type} ap: {ap}')
        threshold_ap[tr] = np.mean([joint_ap[joint_type] for joint_type in joint_ap.keys()])
    print(threshold_ap)
    return threshold_ap


def eval(checkpoint_path, annotations_path, config_path, input_shape=[224, 224]):
    if not annotations_path:
        logging.info('Using default annotations path')
        annotations_path = ANNOTATIONS_PATH

    if not os.path.exists(checkpoint_path):
        logging.error('Checkpoint does not exist!')
        exit(1)
    logging.info('Loaded model checkpoint')

    dataset = PoseDataset(path=IMAGES_PATH, apply_augmentation=False)
    dataset.load_annotations_from_json(annotations_path)
    logging.info(f'Loaded {dataset.ann_count} annotations')

    estimator = SkeletonEstimator(checkpoint_path, config_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=10)

    dists_dict = {}
    for i, batch in tqdm(enumerate(dataloader)):
        images, _, _, joint_tables = batch[0], batch[1], batch[2], batch[3]
        skeletons_pred_batch, _, _, _ = estimator.inference(images)
        pairs_dict = {}
        for j in range(joint_tables.shape[0]):
            skeletons_gt = AnnotatedSkeletons(joint_table=joint_tables[j])
            skeletons_pred = skeletons_pred_batch[j]
            skeletons_pred.scale(skeletons_gt.shape)
            new_dict = (pair_joints(skeletons_gt, skeletons_pred))
            for key in new_dict.keys():
                if key in pairs_dict.keys():
                    pairs_dict[key] += new_dict[key]
                else:
                    pairs_dict[key] = new_dict[key]
        if not pairs_dict:
            continue
        for joint_type in pairs_dict.keys():
            for pair in pairs_dict[joint_type]:
                dist = np.sqrt((pair[0].x - pair[1].x) ** 2 + (pair[0].y - pair[1].y) ** 2)
                if joint_type in dists_dict.keys():
                    dists_dict[joint_type].append((dist, pair[1].confidence))
                else:
                    dists_dict[joint_type] = [(dist, pair[1].confidence)]
    calculate_mAP(dists_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--annotations-path', type=str, default=None, help='Path to the annotation json')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')

    args = parser.parse_args()
    setup_logging()
    eval(args.checkpoint_path, args.annotations_path, args.config_path)













