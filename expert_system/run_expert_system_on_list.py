import argparse
from run_expert_system_on_video import evaluate_expert_system
from utils.setup_logging import setup_logging
import json
import logging
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video-list', type=str, default='expert_system_videos.json',
                        help='Path to the video list to evaluate')
    parser.add_argument('--pose-checkpoint-path', type=str, default='saved_checkpoints/checkpoint_5.pth',
                        help='Path to the pose model checkpoint')
    parser.add_argument('--pose-config-path', type=str, default='pose_config.json', help='Path to the training config')
    parser.add_argument('--appearance-act-checkpoint-path', type=str,
                        default='checkpoints/action_network/checkpoint_appearance_18.pth',
                        help='Path to the appearance action recognition model checkpoint')
    parser.add_argument('--pose-act-checkpoint-path', type=str,
                        default='checkpoints/action_network/checkpoint_pose_18.pth',
                        help='Path to the appearance action recognition model checkpoint')
    parser.add_argument('--dump-avis-folder', type=str, default=None)

    args = parser.parse_args()
    setup_logging()
    video_list = json.load(open(args.video_list,'r'))
    total_tp, total_fp, total_fn = 0, 0, 0
    for video_path in tqdm(video_list):
        logging.info(f'Video: {video_path}')
        logging.disable(logging.CRITICAL + 1)
        tp, fp, fn = evaluate_expert_system(video_path, args.pose_checkpoint_path, args.pose_config_path,
                                            args.pose_act_checkpoint_path, args.appearance_act_checkpoint_path,
                                            dump_avis_folder=args.dump_avis_folder)
        logging.disable(logging.NOTSET)
        logging.info(f'tp: {tp}, fp: {fp}, fn: {fn}')
        total_tp += tp
        total_fn += fn
        total_fp += fp
        logging.info(f'total_tp: {total_tp}, total_fp: {total_fp}, total_fn: {total_fn}')
