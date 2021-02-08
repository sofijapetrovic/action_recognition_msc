import os
import pickle
from os.path import join
from collections import namedtuple
from torch.utils.data import Dataset
import numpy as np
import torch
VIDEOS_PATH = '/media/data/pose_data_3/train'


video_features = namedtuple('VideoFeatures', ['path', 'action_class'])
classes = {'drop': 1, 'no_drop': 0}


class VideoDataset(Dataset):
    def __init__(self, path=VIDEOS_PATH, frames=None):
        super(Dataset, self).__init__()
        self.frames = frames
        self.data = []
        for class_folder in os.listdir(path):
            folder_path = join(path, class_folder)
            for file in os.listdir(folder_path):
                if file[-3:] != 'pkl':
                    continue
                file_path = join(folder_path, file)
                self.data.append(video_features(file_path, classes[class_folder]))
        self.count = len(self.data)

    def __getitem__(self, idx):
        data_info = self.data[idx]
        pose_features, visual_features = pickle.load(open(data_info.path, 'rb'))
        a_class = torch.zeros(2, dtype=torch.long)
        a_class[data_info.action_class] = 1
        a_class = torch.unsqueeze(a_class, -1)
        return [torch.FloatTensor(np.float32(pose_features)), torch.FloatTensor(np.float32(visual_features)), data_info.action_class]

    def __len__(self):
        return self.count


if __name__ == '__main__':
    dataset = VideoDataset()
    output = dataset.__getitem__(np.random.randint(dataset.__len__()))
    print(output[0].shape, output[1].shape, output[2])
