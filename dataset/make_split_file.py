import argparse
import os
import json
from tqdm import tqdm
from os import listdir
import logging
from utils.setup_logging import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video-repository', type=str, default='', help='Path to json')
    parser.add_argument('--validation-ratio', type=float, default=0.2, help='percentage of the test set')
    parser.add_argument('--output-file', type=str, default='dataset_split.json')
    args = parser.parse_args()

    setup_logging()
    eval_file_folder, _ = os.path.split(args.video_repository)
    video_list = json.load(open(args.video_repository, 'r'))
    drop_num_dict = {}
    drop_num = {}
    for vid in tqdm(video_list):
        video_path = os.path.join(eval_file_folder,vid)
        bundle_path, video_name = os.path.split(video_path)

        save = [s for s in listdir(bundle_path) if s[-4:] == 'save']
        if len(save) != 1:
            logging.info(f'Bundle {bundle_path} has more than one save file!')
            exit(1)
        save_path = os.path.join(bundle_path, save[0])
        save_file = json.load(open(save_path, 'r'))
        events = save_file['events']
        lane = '_'.join(video_name.split('_')[1:3])
        if lane in drop_num_dict:
            drop_num_dict[lane].append({'video': video_path, 'drops': len(events)})
            drop_num[lane] += len(events)
        else:
            drop_num_dict[lane] = [{'video': video_path, 'drops': len(events)}]
            drop_num[lane] = len(events)

    total_train, total_val = 0, 0
    split_dict = {'train':[], 'validation': []}
    for lane in drop_num_dict:
        drops = drop_num[lane]
        val_drops = int(args.validation_ratio * drops)
        train_drops = drops - val_drops
        videos = sorted(drop_num_dict[lane], key=lambda x: x['drops'], reverse=True)
        current_val, current_train = 0, 0
        for video in videos:
            if current_train < train_drops:
                '''we need to add more drops to the training set'''
                if video['drops'] + current_train < train_drops:
                    '''video has fewer drops than should be added'''
                    split_dict['train'].append(video['video'])
                    current_train += video['drops']
                    total_train += video['drops']
                else:
                    '''video has more drops than should be aded'''
                    total_drops = total_val + total_train
                    if total_train < total_drops - int(total_drops * args.validation_ratio):
                        '''total number of drops in train set is smaller than expected based on proportions'''
                        split_dict['train'].append(video['video'])
                        current_train += video['drops']
                        total_train += video['drops']
                    else:
                        '''total number of drops in train set is bigger than expected based on proportions'''
                        split_dict['validation'].append(video['video'])
                        current_val += video['drops']
                        total_val += video['drops']
            else:
                '''we don't need to add more drops to the training set'''
                split_dict['validation'].append(video['video'])
                current_val += video['drops']
                total_val += video['drops']
    print(total_train, total_val)
    json.dump(split_dict, open(args.output_file, 'w'), indent=2)

    logging.info('Testing split')
    for cur_set in ['train', 'validation']:
        bundles = set()
        total_drops = 0
        for video_path in tqdm(split_dict[cur_set]):
            if video_path in bundles:
                print(video_path)
            bundles.add(video_path)
            bundle_path, video_name = os.path.split(video_path)
            save = [s for s in listdir(bundle_path) if s[-4:] == 'save']
            save_path = os.path.join(bundle_path, save[0])
            save_file = json.load(open(save_path, 'r'))
            events = save_file['events']
            total_drops += len(events)
        logging.info(f'{cur_set}: total videos {len((split_dict[cur_set]))}, unique videos {len(bundles)}, total drops {total_drops}')




