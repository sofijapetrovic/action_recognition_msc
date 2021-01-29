import torch
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torchgeometry.image.gaussian import GaussianBlur
class Estimator(nn.Module):
    def __init__(self, model, paf_stages=2, heat_stages=2, model_threshold=0.3):
        super(Estimator, self).__init__()
        self.model = model
        self.paf_stages = paf_stages
        self.heat_stages = heat_stages
        self.model_threshold = model_threshold
        self.gaussian = GaussianBlur(kernel_size=(5,5), sigma=(2,2))
        self.max_pool = nn.MaxPool2d(3, 1, padding=1)
        self.output_shape = None
        self.input_shape = (224, 398)

    def get_output_images(self, image):
        image, paf, heat = self.model.get_output_images(torch.FloatTensor(np.transpose(image, (2,0,1))))
        return paf, heat

    def raw_inference(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            image = torch.FloatTensor(np.transpose(image, (2, 0, 1))[np.newaxis, :, :, :]).cuda()
        outputs = self.model(image.cuda())
        paf = outputs[self.paf_stages-1]
        heat = outputs[-1]
        return paf, heat

    def inference(self, image, output_shape=(1080,1920)):
        self.output_shape = output_shape
        paf, heat = self.raw_inference(image)
        hands = heat[:,[0, -1], :, :]
        return self(hands)

    def forward(self, hands_batch):
        hands_batch = self.gaussian(hands_batch)
        maxes = self.max_pool(hands_batch)
        peaks = torch.where(torch.eq(hands_batch, maxes), hands_batch, torch.zeros_like(hands_batch))
        peaks_map = torch.where(torch.gt(peaks, self.model_threshold), peaks, torch.zeros_like(peaks))
        peaks = torch.nonzero(peaks_map, as_tuple=False)

        peaks_t = peaks.t().cpu().numpy()

        peaks_idx = peaks.cpu().numpy()
        confidences = maxes[peaks_t].detach().cpu().numpy()
        out_array = []
        confidence_array = []
        if len(peaks_idx) == 0:
            return [], []
        for i in range(peaks_idx[-1][0]+1):
            bundle = []
            cbundle = []
            for k in range(2):
                img_peaks = [(j,pk[1:]) for j,pk in enumerate(peaks_idx) if pk[0] == i]
                side = [(pk[0],pk[1][1:]) for pk in img_peaks if pk[1][0] == k]
                if len(side) == 0:
                    bundle.append(np.array([]))
                    cbundle.append(np.array([]))
                    continue
                idx, side = zip(*side)
                idx, side = list(idx), list(side)
                cfd = confidences[idx]
                side = [[int(elem[0] * self.output_shape[0]/self.input_shape[0]), int(elem[1] * self.output_shape[1]/self.input_shape[1])]
                        for elem in side]
                side = np.array(side)
                bundle.append(side)
                cbundle.append(cfd)

            out_array.append(bundle)
            confidence_array.append(cbundle)
        return out_array, confidence_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Path to the model checkpoint')
    parser.add_argument('--image-path', type=str, default='', help='Path to the image for the inference')

    args = parser.parse_args()
    img = cv2.imread(args.image_path)
    model = torch.load(args.checkpoint_path)

    estimator = Estimator(model)
    pred_peaks, confidences = estimator.inference(img)
    print(pred_peaks)
    plt.figure()
    plt.imshow(img)
    for k in range(len(pred_peaks[0][0])):
        plt.scatter(pred_peaks[0][0][k][1], pred_peaks[0][0][k][0], s=20,marker='o', c='r')
    for k in range(len(pred_peaks[0][1])):
        plt.scatter(pred_peaks[0][1][k][1], pred_peaks[0][1][k][0], s=20, marker='o', c='b')
    plt.show()









