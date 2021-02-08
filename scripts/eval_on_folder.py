import os
from os import listdir
import pose_eval
import action_eval
import argparse
from os.path import join
from utils.setup_logging import setup_logging
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model evaluation on all the checkpoints in given folder')
    parser.add_argument('--checkpoint-folder', type=str, default='checkpoints/', help='Path to the folder containing model checkpoints')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')

    args = parser.parse_args()
    setup_logging()
    checkpoints = [c for c in listdir(args.checkpoint_folder)]
    logging.info(f'Found checkpoints: {checkpoints}')
    if args.config_path.find('pose') >= 0:
        checkpoints = sorted(checkpoints, key=lambda name: int(name.split("_")[1][:-4]), reverse=True)
        for checkpoint in checkpoints:
            logging.info(f'current checkpoint: {checkpoint}')
            pose_eval.eval(os.path.join(args.checkpoint_folder, checkpoint), None, args.config_path)
    else:
        appearance = [c for c in checkpoints if c.find('appearance') >= 0]
        appearance = sorted(appearance, key=lambda name: int(name.split("_")[-1][:-4]), reverse=True)
        pose = [c for c in checkpoints if c.find('pose') >= 0]
        pose = sorted(pose, key=lambda name: int(name.split("_")[-1][:-4]), reverse=True)
        if len(appearance) != len(pose):
            logging.error(f'Number of appearance network checkpoints and pose network checkpoints is not the same!'
                  f' {len(appearance)} != {len(pose)}')
            exit(1)
        for j, app in enumerate(appearance):
            logging.info(f'current checkpoints: {appearance[j]} / {pose[j]}')
            action_eval.eval(args.config_path, join(args.checkpoint_folder, pose[j]), join(args.checkpoint_folder, appearance[j]))
