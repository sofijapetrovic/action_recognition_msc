import esis
import time
import argparse
import cv2
from estimator import SkeletonEstimator
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model fps on the dvr video')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--video-path', type=str, default='', help='Path to the dvr video')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output-video-file', type=str, default='output_video.avi')
    parser.add_argument('--overlay-raw', action='store_true')
    args = parser.parse_args()

    estimator = SkeletonEstimator(args.checkpoint_path)
    decoder = esis.create(args.video_path)
    out = None
    if args.save_video:
        out = cv2.VideoWriter(args.output_video_file, cv2.VideoWriter_fourcc(*'XVID'), 25, (512, 288))
    fpss = []
    while True:
        start = time.time()
        try:
            ret_val, image, timestamp = decoder.GetFrame()
        except:
            break
        if ret_val != esis.FrameState_eOk:
            continue
        skeletons, _, _, _ = estimator.inference(image/255)
        fpss.append(time.time() - start)
        image = cv2.resize(image, (1920,1080))
        skeletons[0].scale(image.shape)
        skeletons[0].draw(image)

        image = cv2.resize(image, (512, 288))
        if args.overlay_raw:
            paf, heat = estimator.get_output_images(image / 255)
            heat = cv2.resize(heat, (512, 288)) * 255
            cv2.addWeighted(heat, 0.5, image, 0.5, 0, image,dtype=0)
        if args.save_video:
            out.write(image)

        cv2.imshow('JustDrop smoothed result', image)
        if cv2.waitKey(1) == 27:
            break
    if out:
        out.release()
    print(f'average fps: {1/np.mean(fpss)}')