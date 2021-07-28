# The Master Thesis Project

The main project parts:  
1. Pose estimation:
- [Pose estimation dataset loading and processing](dataset/pose_dataset.py)
- [Pose estimation model with VGG19 backbone](model/pose_network.py)
- [Pose estimation model training script](utils/train_pose.py)
- [Pose estimation training and inference config](pose_config.json)
2. Action Recognition
- [Action recognition dataset loading and processing](dataset/video_dataset.py)
- [Action recognition model with VGG19 backbone](model/action_network.py)
- [Action recognition model training script](utils/train_action.py)
- [Action recognition training and inference config](action_config.json)
3. Expert System
- [Running expert system on the single video](expert_system/run_expert_system_on_video.py)
- [Running expert system on the video list](expert_system/run_expert_system_on_list.py)

Other features:
- [Running mAP evaluation of a pose estimation checkpoint](scripts/pose_eval.py)
- [Running evaluation on of a action recognition checkpoint](scripts/action_eval.py)
- [Running pose estimation inference on a video](scripts/run_video.py)
- [Skeletons class used for the easier inference manipulation](skeleton/skeletons.py)

Saved model checkpoint 
- [saved_checkpoints](saved_checkpoints)


