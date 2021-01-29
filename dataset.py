import numpy as np
import logging
import cv2
import os
import torch
from os.path import join
import json
from utils import setup_logging
import time
import imgaug.augmenters as iaa
from torch.utils.data import Dataset

ANNOTATIONS_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/annotations/' \
                   'hand_detection/part1_01_01_2020/v3_train_3clients_sco_part1_01_01_2020_175014.json'
IMAGES_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/images/part1/'
PAF_DIST = 5
SIGMA = 5
KEYPOINTS = 7
class PoseDataset(Dataset):
    def __init__(self, path=IMAGES_PATH, paf_dist=PAF_DIST, sigma=SIGMA, keypoints=KEYPOINTS):
        super(Dataset, self).__init__()
        self.images = []
        self.path = path
        self.annotations = {}
        self.ann_count = 0
        self.keypoint_dict = {'left_hand':0, 'left_elbow':1, 'left_shoulder':2, 'head':3, 'right_shoulder':4,
                              'right_elbow':5, 'right_hand':6}
        self.paf_dist = paf_dist
        self.sigma = sigma
        self.keypoints = keypoints
        self.aug = iaa.Sequential([
                    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                                scale=(0.7, 1.5), rotate=(-2, 2)),
                    iaa.Fliplr(0.5),
        ])
        self.image_shape = (224, 398)


    def load_annotations_from_json(self, path=ANNOTATIONS_PATH):
        anns = json.load(open(path,'r'))
        anns = anns['annotations']
        for ann in anns:
            keypoints = ann['keypoints']
            processed_ann = {}
            for keypoint in keypoints:
                for body_part in keypoint.keys():
                    for key in keypoint[body_part].keys():
                        if key in ['x', 'y']:
                            if keypoint[body_part]['person_id'] in processed_ann.keys():
                                processed_ann[keypoint[body_part]['person_id']][body_part] = {'x': keypoint[body_part]['x'],
                                                                                              'y':keypoint[body_part]['y']}
                            else:
                                processed_ann[keypoint[body_part]['person_id']] = {}
            self.annotations[ann['path']] = processed_ann
        self.ann_count = len(list(self.annotations.keys()))
        shuffled_keys = np.random.permutation(list(self.annotations.keys()))
        self.annotations = {k: self.annotations[k] for k in shuffled_keys}

    def generate_ground_truth(self, img, annotations, original_shape):
        paf_map = np.zeros((img.shape[0], img.shape[1], (self.keypoints-1)*2), dtype='float')
        keypoint_map = np.zeros((img.shape[0], img.shape[1], self.keypoints), dtype='float')
        people_no = len(list(annotations.keys()))
        '''Augmentation'''
        keypoint_list = []
        for person_key in annotations.keys():
            for body_key in annotations[person_key].keys():
                if not body_key in self.keypoint_dict.keys():
                    continue
                curr_dict = annotations[person_key][body_key]
                y = min(int(curr_dict['y'] / original_shape[0] * self.image_shape[0]), self.image_shape[0]-1)
                x = min(int(curr_dict['x'] / original_shape[1] * self.image_shape[1]), self.image_shape[1]-1)
                if not 0 <= y < self.image_shape[0] or not 0 <= x < self.image_shape[1]:
                    continue
                keypoint_list.append((x,y))

        img, keypoint_list = self.aug(images=img[np.newaxis,:,:,:], keypoints=[keypoint_list])
        keypoint_list = keypoint_list[0]
        img = img[0,:,:,:]


        img = img/255
        #normalization for vgg19

        annotations = annotations.copy()
        for person_key in annotations.keys():
            for body_key in annotations[person_key].keys():
                if not body_key in self.keypoint_dict.keys():
                    continue
                keypoint = keypoint_list[0]
                keypoint_list = keypoint_list[1:]
                annotations[person_key][body_key]['y'] = keypoint[1]
                annotations[person_key][body_key]['x'] = keypoint[0]

        '''Generating the ground truth'''
        peaks = -np.ones((10,2,2))
        left_ind = 0
        right_ind = 0
        for person_key in annotations.keys():
            for body_key in annotations[person_key].keys():
                if not body_key in self.keypoint_dict.keys():
                    continue
                curr_dict = annotations[person_key][body_key]
                y = min(int(curr_dict['y']), self.image_shape[0]-1)
                x = min(int(curr_dict['x']), self.image_shape[1]-1)
                if not 0 <= y < self.image_shape[0] or not 0 <= x < self.image_shape[1]:
                    continue
                if body_key =='left_hand':
                    peaks[left_ind, 0, 0] = int(y * original_shape[0] / self.image_shape[0])
                    peaks[left_ind, 0, 1] = int(x * original_shape[1] / self.image_shape[1])
                    left_ind +=1
                elif body_key == 'right_hand':
                    peaks[right_ind, 1, 0] = int(y * original_shape[0] / self.image_shape[0])
                    peaks[right_ind, 1, 1] = int(x * original_shape[1] / self.image_shape[1])
                    right_ind +=1
                empty_image = np.zeros_like(keypoint_map[:, :, self.keypoint_dict[body_key]])
                empty_image[y, x] = 1
                empty_image = cv2.GaussianBlur(empty_image, (0, 0), self.sigma)
                empty_image = (empty_image-np.min(empty_image))/(np.max(empty_image) - np.min(empty_image))
                keypoint_map[:, :, self.keypoint_dict[body_key]] = np.maximum(
                    keypoint_map[:, :, self.keypoint_dict[body_key]], empty_image)
            for i, j in zip(range(len(list(self.keypoint_dict.keys())) - 1),
                            range(1, len(list(self.keypoint_dict.keys())))):
                key1 = list(self.keypoint_dict.keys())[i]
                key2 = list(self.keypoint_dict.keys())[j]
                curr_dict = annotations[person_key]
                if key1 in curr_dict.keys() and key2 in curr_dict.keys():
                    y1 = min(int(curr_dict[key1]['y']), self.image_shape[0]-1)
                    x1 = min(int(curr_dict[key1]['x']), self.image_shape[1]-1)
                    y2 = min(int(curr_dict[key2]['y']), self.image_shape[0]-1)
                    x2 = min(int(curr_dict[key2]['x']), self.image_shape[1]-1)
                    ang = np.arctan2(y2 - y1, x2 - x1)
                    v = (np.cos(ang), np.sin(ang))
                    paf_map[:, :, i*2] += cv2.line(np.zeros_like(paf_map[:, :, 0]), (x1, y1), (x2, y2), color=v[0],
                                                thickness=self.paf_dist) / people_no
                    paf_map[:, :, i*2+1] += cv2.line(np.zeros_like(paf_map[:, :, 1]), (x1, y1), (x2, y2), color=v[1],
                                                thickness=self.paf_dist) / people_no

        return img, keypoint_map, paf_map, peaks


    def __getitem__(self, idx):
        image = list(self.annotations.keys())[idx]
        image_path = join(self.path, image)
        annotations = self.annotations[image]
        if not os.path.exists(image_path):
            logging.error(f'Image {image_path} does not exist')
            return
        img = cv2.imread(image_path)
        original_shape = img.shape
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        img, keypoint_map, paf_map, peaks = self.generate_ground_truth(img, annotations, original_shape)
        img = np.transpose(img,(2,0,1))
        paf_map= np.transpose(paf_map,(2,0,1))
        keypoint_map = np.transpose(keypoint_map,(2,0,1))

        img = torch.FloatTensor(img)
        paf_map = torch.FloatTensor(paf_map)
        keypoint_map = torch.FloatTensor(keypoint_map)

        return [img, paf_map, keypoint_map, peaks]

    def __len__(self):
        return self.ann_count


if __name__ == '__main__':
    setup_logging()
    data = PoseDataset()
    start = time.time()
    data.load_annotations_from_json(ANNOTATIONS_PATH)
    logging.info(f'Time taken to load annotations {time.time()-start}')
    start = time.time()
    batch_size=32*50
    for b in range(batch_size):
        data.__getitem__(0)
    logging.info(f'Time taken to load  images and generate ground truth for {batch_size} files {time.time()-start}s')

