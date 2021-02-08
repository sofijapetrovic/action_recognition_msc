import argparse
import json
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split-file', type=str, default='dataset_split.json', help='Path to the split file')
    parser.add_argument('--output-file', type=str, default='expert_system_videos.json')
    parser.add_argument('--video-number', type=int, default=100)

    args = parser.parse_args()

    split_dict = json.load(open(args.split_file, 'r'))
    chosen = list(np.random.choice(split_dict['validation'], args.video_number))
    json.dump(chosen, open(args.output_file, 'w'), indent=2)
    print(f'Dumped the list of {args.video_number} video paths from the validation set to the {args.output_file}')