# The Master Thesis Project

The main project parts:
- [Dataset loading and processing](dataset/pose_dataset.py)
- [Pose estimation model with VGG19 backbone](model/pose_network.py)
- [Model training script](utils/train_pose.py)
- [Training and inference config](pose_config.json)

Other features:
- [Running mAP evaluation on a checkpoint](scripts/pose_eval.py)
- [Running inference on a video](scripts/run_video.py)
- [Running background subtraction to detect objects in hand in video](scripts/background_subtraction.py)
- [Skeletons class used for the easier inference manipulation](skeleton/skeletons.py)

Saved model checkpoint 
- [saved_checkpoints](saved_checkpoints)


