import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torchgeometry.image.gaussian import GaussianBlur
import cv2
import numpy as np
from skeleton.joint import Joint, JointType, LimbType
from skeleton.inference_skeletons import InferenceSkeletons
import json
from model.pose_network import PoseNet

class SkeletonEstimator(nn.Module):
    def __init__(self, model_path, config_path=None, connect_joints=False, clear_unpaired=False):
        super(SkeletonEstimator, self).__init__()
        if config_path is None:
            config_path = 'pose_config.json'
        config = json.load(open(config_path, 'r'))
        self.model = PoseNet(sigma=config['sigma'], paf_output=config['paf_output'],
                        output_shape=config['image_shape'], upscale=config['upscale'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.paf_output = config['paf_output']
        self.model_threshold = config['threshold']
        self.connect_joints = connect_joints
        self.clear_unpaired = clear_unpaired
        if config['upscale']:
            self.gaussian = GaussianBlur(kernel_size=(5,5), sigma=(config['sigma'],config['sigma']))
        else:
            self.gaussian = None
        self.max_pool = nn.MaxPool2d(3, 1, padding=1)
        self.output_shape = None
        self.input_shape = tuple(config['image_shape'])

    def get_output_images(self, image=None, paf=None, heat=None):
        """
        Returns paf and heat images for a given input image
        Args:
            image (np.array): raw image

        Returns: Grayscale images that represent paf and heat outputs of the network as np.arrays
        heat image is calculated as sum of all the heat channels
        paf image is calculated as sum of absolute values of all the paf channels
        """
        image, paf, heat = self.model.get_output_images(image_tensor=torch.FloatTensor(np.transpose(image, (2, 0, 1))),
                                                        paf=paf, heat=heat)
        paf = paf.detach().numpy()
        paf = np.transpose(paf, (1, 2, 0))
        heat = heat.detach().numpy()
        heat = np.transpose(heat, (1, 2, 0))
        return paf, heat

    def inference(self, image, output_shape=(1080, 1920)):
        """
        Do the inference and postprocessing of the model output on the given image
        Args:
            image (np.array): raw image
            output_shape (tuple): desired output shape to which all the keypoint coordinates should be scaled

        Returns: output of forward() method

        """
        if isinstance(image, np.ndarray) and tuple(image.shape[:2]) != tuple(self.input_shape):
            image = cv2.resize(image, self.input_shape)
        self.output_shape = output_shape
        return self(image)

    def forward(self, image):
        """
        Does the model inference on the image and postprocesses the output
        Args:
            image (np.array/FloatTensor): raw image

        Returns:
            list of Joint objects
            paf map image
            heat map image

        """
        """convert to the float tensor if image is a numpy array"""
        if isinstance(image, np.ndarray):
            image = image
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            image = torch.FloatTensor(np.transpose(image, (2, 0, 1))[np.newaxis, :, :, :]).cuda()
        outputs = self.model(image.cuda())
        """get the output from the last stages of paf and heat subnetworks"""
        original_paf = outputs[self.paf_output]
        original_heat = outputs[-1]
        '''gaussian blur'''
        if self.gaussian:
            heat = self.gaussian(original_heat)
        else:
            heat = original_heat
        '''finding local maxima'''
        maxes = self.max_pool(heat)
        '''non-max suppression'''
        peaks = torch.where(torch.eq(heat, maxes), heat, torch.zeros_like(heat))
        '''keep only the peaks with value greater than model threshold'''
        peaks_map = torch.where(torch.gt(peaks, self.model_threshold), peaks, torch.zeros_like(peaks))
        '''find the coordinates of the peaks'''
        peaks = torch.nonzero(peaks_map, as_tuple=False)
        '''convert to numpy'''
        peaks_t = peaks.t().cpu().numpy()

        peaks_idx = peaks.cpu().numpy()
        #for ind in range(len(peaks_idx)):
        #    peaks_idx[ind][2] = peaks_idx[ind][2]/self.input_shape[0] * self.output_shape[0]
        #    peaks_idx[ind][3] = peaks_idx[ind][3]/self.input_shape[1] * self.output_shape[1]
        confidences = maxes[peaks_t].detach().cpu().numpy()
        skeletons_batch = []
        paf_map = original_paf.cpu().detach().numpy()
        batch_size = image.shape[0]
        joints = []
        for i in range(batch_size):
            joints.append([])
        for j, joint_array in enumerate(peaks_idx):
            joints[joint_array[0]].append(Joint(joint_array[3], joint_array[2], JointType(joint_array[1]), confidences[j]))
        for j, joint_list in enumerate(joints):
            skeletons_batch.append(InferenceSkeletons(joint_list, paf_map[j], original_heat[0, 0, :, :].shape))
        for skeletons in skeletons_batch:
            skeletons.set_shape(original_heat[0, 0, :, :].shape)
            if self.connect_joints:
                skeletons.connect_joints()
            if self.connect_joints and self.clear_unpaired:
                skeletons.clear_unpaired_joints()

        return skeletons_batch, original_paf, original_heat, outputs[self.paf_output-1]


if __name__ == '__main__':
    from model.pose_network import PoseNet
    import json
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--image-path', type=str, default='', help='Path to the image for the inference')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')

    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    config = json.load(open(args.config_path, 'r'))

    estimator = SkeletonEstimator(args.checkpoint_path)
    skeletons, paf_map, heat_map, _ = estimator.inference(image / 255)

    plt.figure()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    image[:,:,0] = 0
    image[:, :, 1] = 255
    image[:, :, 2] = 0
    skeletons[0].scale(image.shape)
    skeletons[0].draw(image)
    plt.subplot(1, 2,1)
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.imshow(image)

    plt.figure()
    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.title(JointType(i))
        plt.imshow(heat_map[0][i, :, :].cpu().detach().numpy())
        plt.colorbar()
    plt.show()
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.title(LimbType(i))
        plt.imshow(paf_map[0][i*2, :, :].cpu().detach().numpy()**2 + paf_map[0][i*2+1, :, :].cpu().detach().numpy()**2)
        plt.colorbar()
    #plt.imshow(np.sum(heat_map[0].cpu().detach().numpy(), axis=0))
    plt.show()









