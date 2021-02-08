from dataset.make_video_dataset import make_input_tensors, scale_area
import argparse
import esis
from estimator import SkeletonEstimator
from model.action_network import ActionModel
import torch
import time
import logging
from utils.setup_logging import setup_logging
from collections import deque
import json
import os
from os import listdir
from datetime import datetime
import cv2
import numpy as np


def dump_avi(images, output_folder,  suffix, video_name):
    if not images:
        return
    timestamp = images[-1][0]
    h = int(timestamp/60/60)
    m = int((timestamp-h*60*60)/60)
    s = timestamp - m * 60 - h * 60 * 60
    time_str = f'_{h}:{m}:{s}_{suffix}.avi'
    filepath = os.path.join(output_folder, video_name + time_str)
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), 10, (images[0][1].shape[1], images[0][1].shape[0]))
    for i in range(len(images)):
        out.write(images[i][1])
    out.release()


def cut_array(images, timestamp, frame_num=100):
    times = np.array([i[0] for i in images])
    ind = np.where(times == timestamp)[0]
    ind = ind[-1]
    return images[ind-frame_num:ind]


def evaluate_expert_system(video_path, pose_checkpoint_path, pose_config_path, pose_act_checkpoint_path,
                           appearance_act_checkpoint_path, display_video=False, dump_avis_folder=None, print_logs=False):
    decoder = esis.create(video_path)
    print_logs and logging.info('Loaded video')

    '''load models'''
    estimator = SkeletonEstimator(pose_checkpoint_path, pose_config_path, connect_joints=True)
    net = ActionModel()
    net.pose_action_network.load_state_dict(torch.load(pose_act_checkpoint_path))
    net.pose_action_network.eval()
    net.appearance_action_network.load_state_dict(torch.load(appearance_act_checkpoint_path))
    net.pose_action_network.eval()
    print_logs and logging.info('Networks loaded')

    '''make deques'''
    app_features = deque(maxlen=150)
    pose_features = deque(maxlen=150)

    '''load annotations and config'''
    bundle_path, video_name = os.path.split(video_path)
    save = [s for s in listdir(bundle_path) if s[-4:] == 'save']
    save_path = os.path.join(bundle_path, save[0])
    save_file = json.load(open(save_path, 'r'))
    events = save_file['events']
    config = json.load(open(os.path.join(bundle_path, 'video_config.json'), 'r'))
    drop_area = config['jdDropArea']

    '''extract drop times'''
    drop_times = []
    for event in events:
        time_gt = datetime.strptime(event['time'].split('.')[0], "%H:%M:%S").time()
        drop_times.append(time_gt.hour * 60 * 60 + time_gt.minute * 60 + time_gt.second)

    scaled = False
    drop_counter = 0
    no_drop_counter = 0
    dropped = False
    predicted_drops = []
    gt_drops = drop_times.copy()
    gif_images = [] if dump_avis_folder else None
    frame = 0
    fpss = []
    while True:
        start = time.time()
        try:
            ret_val, image, timestamp = decoder.GetFrame()
        except:
            break
        if ret_val != esis.FrameState_eOk:
            continue
        frame += 1
        skeletons, _, heat, F = estimator.inference(image / 255)
        if not scaled:
            scale_area(config["jdImageShape"], heat, drop_area)
            scaled = True
        orig_timestamp = datetime.utcfromtimestamp(timestamp / 1000)
        time_str = datetime.strftime(orig_timestamp, "%H:%M:%S")
        timestamp = orig_timestamp.hour * 60 * 60 + orig_timestamp.minute * 60 + orig_timestamp.second

        joint_features, visual_features = make_input_tensors(skeletons[0], heat, F, drop_area)
        pose_features.append(joint_features)
        app_features.append(visual_features)

        if len(pose_features) < pose_features.maxlen:
            continue
        joint_features = torch.unsqueeze(torch.stack(list(pose_features), dim=0), dim=0)
        visual_features = torch.unsqueeze(torch.stack(list(app_features), dim=0), dim=0)

        joint_features = torch.transpose(joint_features, -1, 1)
        visual_features = torch.transpose(visual_features, -1, 1)

        _, app_out = net.appearance_action_network(visual_features.cuda())
        _, pose_out = net.pose_action_network(joint_features.cuda().float())
        out = (pose_out + app_out) / 2
        out = out.cpu().detach()
        out = np.squeeze(out)
        combined_class = int(out[1] > 0.9) #torch.argmax(out, dim=-1).cpu().detach()
        if combined_class == 0:
            out_tuple = ['', '']
            drop_counter = 0
            no_drop_counter += 1
        else:
            drop_counter += 1
            out_tuple = ['drop', '']
            no_drop_counter = 0
        for time_gt in drop_times:
            if timestamp >= time_gt:
                out_tuple[1] = 'drop'
                print_logs and logging.info(f'{time_str} ground truth drop')
                drop_times.remove(time_gt)
            break
        if drop_counter > 20 and not dropped:
            predicted_drops.append(timestamp)
            print_logs and logging.info(f'{time_str} predicted drop')
            dropped = True

        if no_drop_counter > 0 and dropped:
            dropped = False

        if gif_images is not None and (frame % 5 == 0 or (gif_images and gif_images[-1][0] != timestamp and
                                                          (out_tuple[0] == 'drop' or out_tuple[1] == 'drop'))):
            gif_images.append((timestamp, cv2.resize(image, (288, 160))))
        fpss.append(1/(time.time() - start))
        if display_video:
            image = cv2.resize(image, (512, 288))
            cv2.putText(image, f"{out_tuple[0]}/{out_tuple[1]}", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        1)
            if drop_counter > 20:
                cv2.putText(image, f"{out_tuple[0]}/{out_tuple[1]}", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
            cv2.imshow('Video stream', image)
            if cv2.waitKey(1) == 27:
                break

    print('average fps', np.mean(fpss))
    assigned = np.zeros(len(gt_drops))
    tps = []
    fps = []
    fns = []
    for pd in predicted_drops:
        tp = False
        for i, gd in enumerate(gt_drops):
            if assigned[i]:
                continue
            if abs(pd - gd) < 7:
                assigned[i] = 1
                tps.append(pd)
                tp = True
                break
        if not tp:
            fps.append(pd)
    for i, a in enumerate(assigned):
        if not assigned[i]:
            fns.append(gt_drops[i])
    print_logs and logging.info(f'Expert system results:\nTP: {len(tps)}\nFP:{len(fps)}\nFN: {len(fns)}')

    if gif_images is not None:
        for fp in fps:
            dump_avi(cut_array(gif_images, fp), dump_avis_folder,  'fp', video_name)
        for fn in fns:
            dump_avi(cut_array(gif_images, fn), dump_avis_folder,  'fn', video_name)
    return len(tps), len(fps), len(fns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video-path', type=str, default='',
                        help='Path to the video tu evaluate')
    parser.add_argument('--pose-checkpoint-path', type=str, default='saved_checkpoints/checkpoint_5.pth',
                        help='Path to the pose model checkpoint')
    parser.add_argument('--pose-config-path', type=str, default='pose_config.json', help='Path to the training config')
    parser.add_argument('--appearance-act-checkpoint-path', type=str,
                        default='checkpoints/action_network/checkpoint_appearance_18.pth',
                        help='Path to the appearance action recognition model checkpoint')
    parser.add_argument('--pose-act-checkpoint-path', type=str,
                        default='checkpoints/action_network/checkpoint_pose_18.pth',
                        help='Path to the appearance action recognition model checkpoint')
    parser.add_argument('--display-video', action='store_true')
    parser.add_argument('--dump-avis-folder', type=str, default=None)

    args = parser.parse_args()
    setup_logging()
    evaluate_expert_system(args.video_path, args.pose_checkpoint_path, args.pose_config_path, args.pose_act_checkpoint_path,
                           args.appearance_act_checkpoint_path, args.display_video, dump_avis_folder=args.dump_avis_folder,
                           print_logs=True)

