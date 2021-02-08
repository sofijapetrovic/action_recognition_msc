import argparse
import os
import pickle
import numpy as np
from os import listdir
import logging
from utils.setup_logging import setup_logging
import esis
import cv2
from datetime import datetime
import json
from tqdm import tqdm
import torch
from estimator import SkeletonEstimator
from multiprocessing import Process
from skeleton.joint import JointType, LimbType


def add_no_drop_times(drop_times, time_tr):
    if len(drop_times) == 0:
        return []
    no_drop_times = [drop_times[0] - time_tr]
    for j, drop_time in enumerate(drop_times):
        if j == 0:
            continue
        i = 1
        while drop_times[j-1] + time_tr*(i+1) < drop_times[j]:
            time = drop_times[j-1] + time_tr*i
            i += 1
            no_drop_times.append(time)
    no_drop_times.append(drop_times[-1] + time_tr)
    if len(drop_times) + 3 < len(no_drop_times):
        no_drop_times = list(np.random.choice(no_drop_times, len(drop_times) + 3, replace=False))
    return no_drop_times


def scale_area(config_size, output_tensor, area_array):
    target_size = (output_tensor.shape[3], output_tensor.shape[2])
    for j, area in enumerate(area_array):
        area[0] *= target_size[0]/config_size[0]
        area[2] *= target_size[0] / config_size[0]
        area[1] *= target_size[1] / config_size[1]
        area[3] *= target_size[1] / config_size[1]
        area_array[j] = [int(round(a)) for a in area]


def distance_from_area(joint, area_array):
    min_dist = None
    for area in area_array:
        center = ((area[0] + area[2])/2, (area[1] + area[3])/2)
        dist = np.sqrt((joint.x - center[0])**2 + (joint.y - center[1])**2)
        if min_dist is not None and dist < min_dist:
            min_dist = dist
        elif min_dist is None:
            min_dist = dist
    return min_dist


def make_input_tensors(skeletons, heat, F, drop_area=None):
    heat = torch.squeeze(heat)
    heat = torch.transpose(torch.transpose(heat, 0, 1), 1, 2)
    if drop_area:
        drop_map = torch.zeros_like(heat[:, :, 0])
        for da in drop_area:
            drop_map[da[1]:da[3], da[0]:da[2]] = 1
        drop_map = torch.unsqueeze(drop_map, dim=-1)
        heat = torch.cat([heat, drop_map], dim=-1)
    F = torch.squeeze(F)
    F = torch.transpose(torch.transpose(F, 0, 1), 1, 2)
    heat = torch.stack([heat] * F.shape[-1], -1)
    F = torch.stack([F] * heat.shape[-2], -2)
    visual_features = torch.sum(heat * F, dim=[0, 1])

    joints = {}
    joint_tensor = np.ones((len(JointType), 2)) * -1
    '''check how many persons are in the image'''
    for join_type in JointType:
        if join_type not in skeletons.joints_dict:
            continue
        '''for every joint type'''
        added = False
        if len(skeletons.joints_dict[join_type]) == 0:
            '''if there is no joint of the target type continue'''
            continue
        if len(skeletons.joints_dict[join_type]) == 1:
            '''if there is exactly one joint of the target type add it to the list'''
            joints[join_type] = skeletons.joints_dict[join_type][0]
            continue
        '''if there are multiple joints of the target type see which one belongs to the person closes to the drop area'''
        dists = []
        '''find all the limbs that the target joint type can belong to'''
        possible_limbs = []
        if LimbType.has_value(join_type.value) and LimbType(join_type.value) in skeletons.limbs_dict:
            possible_limbs += skeletons.limbs_dict[LimbType(join_type.value)]
        if LimbType.has_value(join_type.value - 1) and LimbType(join_type.value - 1) in skeletons.limbs_dict:
            possible_limbs.append += skeletons.limbs_dict[LimbType(join_type.value - 1)]
        '''go over all the joints of the target type'''
        for joint in skeletons.joints_dict[join_type]:
            '''go over all the limbs that joint chould belong to'''
            for limb in possible_limbs:
                '''see if the joint belongs to the limb that already has other joint added to the joint list'''
                if joint in limb.get_joints() and any(j in limb.get_joints() for j in joints):
                    joints[join_type] = joint
                    added = True
                    break
            if added:
                break
            '''if the joint doesn't belong to the limb save its distance from the drop area'''
            dists.append(distance_from_area(joint, drop_area))
        if not added:
            '''add the joint that is closest to the drop area'''
            joints[join_type] = skeletons.joints_dict[join_type][np.argmin(dists)]

    for joint_type in JointType:
        if joint_type in joints:
            joint_tensor[joint_type.value, :] = [joints[joint_type].y, joints[joint_type].x]
    '''if some joints were not found'''
    all_found = list(set(np.where(joint_tensor != -1)[0]))
    if len(all_found) > 0:
        '''if there is at least one joint found'''
        for i in range(joint_tensor.shape[0]):
            if i not in all_found:
                '''if the joint was not found in the image'''
                previous = [a for a in all_found if a < i]
                next = [a for a in all_found if a > i]
                if len(previous) and len(next):
                    '''if there are two surrounding joints take the point between them'''
                    joint_tensor[i, :] = (joint_tensor[previous[-1]] + joint_tensor[next[0]])/2
                elif len(previous):
                    '''take the coordinates of the last previous joint'''
                    joint_tensor[i, :] = joint_tensor[previous[-1]]
                elif len(next):
                    '''take the coordinates of the first following joint'''
                    joint_tensor[i, :] = joint_tensor[next[0]]
    joint_tensor = torch.tensor(joint_tensor)
    return joint_tensor.cpu().detach(), visual_features.cpu().detach()


def save_to_pkl(path, saved_frames, frame_num):
    if len(saved_frames['visual_features']) < frame_num:
        logging.warning(f'Not enough frames! Available frames {len(saved_frames["visual_features"])}, required frames: '
                        f'{frame_num}')
        return
    joint_features = torch.stack(saved_frames['joint_features'][-frame_num:], dim=0)
    visual_features = torch.stack(saved_frames['visual_features'][-frame_num:], dim=0)
    pickle.dump((joint_features, visual_features), open(path, 'wb'))


def dump_videos(video_list, process_num, args):
    np.random.shuffle(video_list)
    estimator = None
    if not args.dump_avis:
        estimator = SkeletonEstimator(args.checkpoint_path, args.config_path, connect_joints=True)
    for ind, video_path in enumerate(video_list):
        bundle_path, video_name = os.path.split(video_path)
        logging.info(f'Process {process_num}: {ind + 1}/{len(video_list)} {video_name}')
        save = [s for s in listdir(bundle_path) if s[-4:] == 'save']
        save_path = os.path.join(bundle_path, save[0])
        save_file = json.load(open(save_path, 'r'))
        events = save_file['events']
        config = json.load(open(os.path.join(bundle_path, 'video_config.json'), 'r'))
        drop_area = config['jdDropArea']

        drop_times = []
        for event in events:
            time = datetime.strptime(event['time'].split('.')[0], "%H:%M:%S").time()
            drop_times.append(time.hour * 60 * 60 + time.minute * 60 + time.second)
        no_drop_times = add_no_drop_times(drop_times, time_tr=5)

        decoder = esis.create(video_path)
        times = sorted(drop_times + no_drop_times)
        outs = None
        saved_frames = {'visual_features': [], 'joint_features': []}
        if args.dump_avis:
            outs = {time: None for time in times}
        while True:
            try:
                ret_val, image, timestamp = decoder.GetFrame()
            except:
                break
            if ret_val != esis.FrameState_eOk:
                continue
            timestamp = datetime.utcfromtimestamp(timestamp/1000)
            timestamp = timestamp.hour * 60*60 +timestamp.minute * 60 + timestamp.second
            if not args.dump_avis:
                skeletons, _, heat, F = estimator.inference(image / 255)
                if not saved_frames['visual_features']:
                    scale_area(config["jdImageShape"], heat, drop_area)
                joint_features, visual_features = make_input_tensors(skeletons[0], heat, F, drop_area)
                saved_frames['visual_features'].append(visual_features)
                saved_frames['joint_features'].append(joint_features)
                for time in times:
                    if timestamp >= time + 2:
                        folder = 'drop' if time in drop_times else 'no_drop'
                        hours = int(time / 3600)
                        minutes = int((time - hours * 3600) / 60)
                        seconds = time - hours * 3600 - minutes * 60
                        time_str = f'{hours}_{minutes}_{seconds}'
                        path = os.path.join(args.output_folder, folder, f'{video_name[:-4]}_{time_str}.pkl')
                        save_to_pkl(path, saved_frames, args.video_length*30)
                        times.remove(time)
                    break
            else:
                for time in times:
                    if -0.2 <= (time - timestamp) <= args.video_length:
                        image = cv2.resize(image, (512, 288))
                        if outs[time] is None:
                            hours = int(time/3600)
                            minutes = int((time-hours*3600)/60)
                            seconds = time - hours*3600 - minutes * 60
                            time_str = f'{hours}_{minutes}_{seconds}'
                            folder = 'drop' if time in drop_times else 'no_drop'
                            outs[time] = cv2.VideoWriter(os.path.join(args.output_folder, folder, f'{video_name[:-4]}_{time_str}.avi'),
                                                         cv2.VideoWriter_fourcc(*'XVID'), 25, (512, 288))
                        outs[time].write(image)
                    elif (time - timestamp) < -0.2 and outs[time] is not None:
                            outs[time].release()
                            outs[time] = None
                            #logging.info(f'Dumped video of time {time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split-file', type=str, default='dataset_split.json')
    parser.add_argument('--set', type=str, default='train')
    parser.add_argument('--video-length', type=int, default=5, help='Length of the each event video for dumping')
    parser.add_argument('--output-folder', type=str, default='/media/data/pose_data_3/train/', help='')
    parser.add_argument('--process-num', type=int, default=3)
    parser.add_argument('--dump-avis', action='store_true', default=False)
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')

    args = parser.parse_args()

    setup_logging()
    split_file = json.load(open(args.split_file, 'r'))
    chunks = [split_file[args.set][i::args.process_num] for i in range(args.process_num)]

    processes = []
    for i in range(args.process_num):
        t = Process(target=dump_videos, args=(chunks[i], i, args))
        t.start()
        processes.append(t)

    for p in processes:
        p.join()
