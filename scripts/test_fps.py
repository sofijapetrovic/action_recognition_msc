import esis
from model import PoseNet
import time
import numpy as np
import argparse
import torch
import cv2
from estimator import Estimator

def calcualte_fps(estimator, video_path, averaging_window=50):
    decoder = esis.create(video_path)
    fps = []
    while True:
        try:
            ret_val, image, timestamp = decoder.GetFrame()
        except:
            break
        if ret_val != esis.FrameState_eOk:
            continue
        start = time.time()
        estimator.inference(image)
        fps.append(1/(time.time()-start))
        if len(fps) % averaging_window == 0:
            print(np.mean(fps[-averaging_window:]))
    return np.mean(fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model fps on the dvr video')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--video-path', type=str, default='', help='Path to the dvr video')

    args = parser.parse_args()
    model = torch.load(args.checkpoint_path)
    estimator = Estimator(model)
    fps = calcualte_fps(estimator, args.video_path)
    print('Total average fps: ', fps)