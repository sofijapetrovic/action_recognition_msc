import esis
from model import PoseNet
import time
import numpy as np
import argparse
import torch
import cv2
from estimator import Estimator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model fps on the dvr video')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--video-path', type=str, default='', help='Path to the dvr video')

    args = parser.parse_args()

    model = torch.load(args.checkpoint_path)
    estimator = Estimator(model, model_threshold=0.5)
    decoder = esis.create(args.video_path)
    fps = []
    while True:
        try:
            ret_val, image, timestamp = decoder.GetFrame()
        except:
            break
        if ret_val != esis.FrameState_eOk:
            continue
        pred_peaks, confidences = estimator.inference(image)
        #paf, heat = estimator.get_output_images(image)
        image = cv2.resize(image, (1920,1080))
        if len(pred_peaks) > 0:
            for j in range(len(pred_peaks[0][0])):
                cv2.circle(image,(pred_peaks[0][0][j][1], pred_peaks[0][0][j][0]), radius=10, color=(0,0,255),thickness=-1)
            for j in range(len(pred_peaks[0][1])):
                cv2.circle(image,(pred_peaks[0][1][j][1], pred_peaks[0][1][j][0]), radius=10, color=(255,0,0),thickness=-1)

        image = cv2.resize(image,(512,288))
        cv2.imshow('JustDrop smoothed result', image)
        if cv2.waitKey(1) == 27:
            break