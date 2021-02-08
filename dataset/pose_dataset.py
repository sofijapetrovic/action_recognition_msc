import numpy as np
import logging
import cv2
import os
import torch
from os.path import join
import json
from utils.setup_logging import setup_logging
import time
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
import imgaug as ia
import matplotlib.pyplot as plt
from skeleton.joint import JointType, Joint
from skeleton.annotated_skeletons import AnnotatedSkeletons
ANNOTATIONS_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/annotations/' \
                   'hand_detection/part1_01_01_2020/v3_train_3clients_sco_part1_01_01_2020_175014.json'
IMAGES_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/images/part1/'
PAF_DIST = 5
SIGMA = 5
KEYPOINTS = 7


class PoseDataset(Dataset):
    def __init__(self, path=IMAGES_PATH, paf_dist=PAF_DIST, sigma=SIGMA, keypoints=KEYPOINTS, image_shape=(224, 398),
                 apply_augmentation=True):
        super(Dataset, self).__init__()
        self.images = []
        self.path = path
        self.annotations = {}
        self.ann_count = 0
        self.apply_augmentation = apply_augmentation
        self.keypoint_dict = {'left_hand': JointType.LEFT_HAND,
                              'left_elbow': JointType.LEFT_ELBOW,
                              'left_shoulder': JointType.LEFT_SHOULDER,
                              'head': JointType.HEAD,
                              'right_shoulder': JointType.RIGHT_SHOULDER,
                              'right_elbow': JointType.RIGHT_ELBOW,
                              'right_hand': JointType.RIGHT_HAND}
        self.paf_dist = paf_dist
        self.sigma = sigma
        self.keypoints = keypoints
        self.aug = iaa.Sequential([iaa.Sometimes(0.6, iaa.GaussianBlur(sigma=(0, 0.5))),
                                   iaa.Sometimes(0.3, iaa.LinearContrast((0.75, 1.5))),
                    iaa.Sometimes(0.4, iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                                scale=(0.8, 1.5), rotate=(-15, 15)))], random_order=True)

        self.image_shape = image_shape

    def load_annotations_from_json(self, path=ANNOTATIONS_PATH):
        """
        Populates annotations dict with joints in format dict["image_path"] = AnnotatedSKeleton(joints_in_image)
        Args:
            path (str): path to the annotations json file
        """
        anns = json.load(open(path,'r'))
        self.annotations = anns['annotations']
        self.ann_count = len(self.annotations)

    def generate_ground_truth(self, image_path, annotations):
        """
        Generate paf and heat maps for given image
        Args:
            img (np.array): original image
        Returns:

        """
        img = cv2.imread(image_path)
        joints = []
        original_shape = img.shape
        for keypoint in annotations:
            for body_part in keypoint.keys():
                if body_part not in self.keypoint_dict.keys():
                    continue
                joints.append(Joint(x=keypoint[body_part]['x'],
                                    y=keypoint[body_part]['y'],
                                    type=self.keypoint_dict[body_part],
                                    id=keypoint[body_part]['person_id']))
        skeletons = AnnotatedSkeletons(joints)
        skeletons.set_shape(original_shape)
        skeletons.scale(self.image_shape)
        skeletons.remove_out_of_bounds()
        '''resizing for the network'''
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        '''Augmentation'''
        if self.apply_augmentation:
            keypoint_list = skeletons.to_list()
            img, keypoint_list = self.aug(images=img[np.newaxis, :, :, :], keypoints=[keypoint_list])
            keypoint_list = keypoint_list[0]
            skeletons.from_list(keypoint_list)
            skeletons.remove_out_of_bounds()
            img = img[0, :, :, :]
        img = img/255
        '''Generating the ground truth'''
        keypoint_map = skeletons.get_keypoint_map(self.sigma)
        paf_map = skeletons.get_paf_map(self.paf_dist)
        skeletons.scale(original_shape)
        return img, keypoint_map, paf_map, skeletons.to_joint_table()

    def __getitem__(self, idx):
        '''reding the image from the saved path'''
        image_name = self.annotations[idx]['path']
        annotation = self.annotations[idx]['keypoints']
        image_path = join(self.path, image_name)
        if not os.path.exists(image_path):
            logging.error(f'Image {image_path} does not exist')
            return
        '''generating the ground truth maps and skeletons'''
        img, keypoint_map, paf_map, joint_table = self.generate_ground_truth(image_path, annotation)
        '''transposing for the torch input'''
        img = np.transpose(img, (2, 0, 1))
        paf_map= np.transpose(paf_map, (2, 0, 1))
        keypoint_map = np.transpose(keypoint_map, (2, 0, 1))
        '''converting to torch.floattensor'''
        img = torch.FloatTensor(img)
        paf_map = torch.FloatTensor(paf_map)
        keypoint_map = torch.FloatTensor(keypoint_map)

        return [img, paf_map, keypoint_map, joint_table]

    def __len__(self):
        return self.ann_count


if __name__ == '__main__':
    setup_logging()
    data = PoseDataset()
    start = time.time()
    data.load_annotations_from_json(ANNOTATIONS_PATH)
    logging.info(f'loaded {data.__len__()} annotations')
    logging.info(f'Time taken to load annotations {time.time()-start}')
    start = time.time()
    batch_size=32*50
    for b in range(batch_size):
        print(b)
        img, paf_map, keypoint_map, skeletons = data.__getitem__(b)
        plt.figure()
        img = np.transpose(img,(1,2,0))
        keypoint_map = np.transpose(keypoint_map, (1,2,0))
        paf_map = np.transpose(paf_map, (1, 2, 0))
        plt.suptitle('Heat')
        plt.subplot(4, 2, 1)
        plt.imshow(img)
        for i in range(keypoint_map.shape[-1]):
            plt.subplot(4, 2, i+2)
            plt.imshow(keypoint_map[:,:,i])
        plt.show()
        plt.figure()
        plt.suptitle('Paf')
        plt.subplot(5,3,1)
        plt.imshow(img)
        for i in range(paf_map.shape[-1]):
            plt.subplot(5, 3, i+2)
            plt.imshow(paf_map[:, :, i])
        plt.show()
    logging.info(f'Time taken to load  images and generate ground truth for {batch_size} files {time.time()-start}s')

