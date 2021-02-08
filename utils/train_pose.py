import torch
import logging
import torch.nn as nn
import torch.optim as optim
from dataset.pose_dataset import PoseDataset
from model.pose_network import PoseNet

from utils.setup_logging import setup_logging
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
import argparse
import json

ANNOTATIONS_PATH = '/mnt/bgnas01-cold/JustDrop/ozark_sco/image_repository/annotations/' \
                   'hand_detection/part1_01_01_2020/v3_train_3clients_sco_part1_01_01_2020_175014.json'


def train(net, dataset, paf_output, paf_stages, heat_stages, epochs=1, initial_epoch=0,
          batch_size=18, paf_weight=1, heat_weight=1, log_every_iter=100,
          save_every_epoch=3, image_every_iter=2200):

    criterions = [nn.MSELoss(size_average=None, reduce=None, reduction='mean') for i in range(heat_stages+paf_stages)]
    optimizer = optim.Adam(params=net.parameters(), lr=0.0005)

    writer = SummaryWriter(log_dir='../logs/')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    logging.info('Starting training')
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        cur_epoch = 0
        iteration = 0
        logging.info(f'epoch: {epoch+1+initial_epoch}')
        for i_batch, batch in tqdm(enumerate(dataloader)):
            iteration += 1
            images, pafs, heats = batch[0], batch[1], batch[2]
            paf_gts = [pafs for i in range(paf_stages)]
            heat_gts = [heats for i in range(heat_stages)]
            gts = paf_gts + heat_gts
            # zero the parameter gradients
            optimizer.zero_grad()

            start = time.time()
            outputs = net(images.cuda())

            losses = None
            for j, criterion in enumerate(criterions):
                loss = criterion(outputs[j + paf_output].cuda(), gts[j].cuda())
                if losses:
                    k = paf_weight if j < paf_stages else heat_weight
                    losses += k*loss
                else:
                    k = paf_weight if j<paf_stages else heat_weight
                    losses = k*loss

            losses.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if iteration % log_every_iter == log_every_iter-1:  # print every 2000 mini-batche
                writer.add_scalar('Loss', running_loss/log_every_iter, iteration + (epoch+initial_epoch)*dataset.__len__()/batch_size)
                running_loss = 0.0
            if iteration % image_every_iter == image_every_iter-1:
                model.eval()
                image, paf, heat = model.get_output_images(images[0, :, :, :])
                model.train()
                grid = make_grid([image,paf,heat])
                writer.add_image(f'Input epoch: {epoch + initial_epoch + 1} iteration: {iteration}', grid, 0)
        if epoch % save_every_epoch == 0:
            torch.save(model.state_dict(), f'checkpoints/checkpoint_{epoch+1+initial_epoch}.pth')
        logging.info(f'epoch: {cur_epoch + 1}, loss: {running_loss / log_every_iter}')

    print('Finished Training')

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--config-path', type=str, default='pose_config.json', help='Path to the training config')

    args = parser.parse_args()
    config = json.load(open(args.config_path, 'r'))

    setup_logging()
    '''Load dateset'''
    data = PoseDataset(image_shape=config['image_shape'])
    data.load_annotations_from_json(ANNOTATIONS_PATH)
    logging.info('Loaded annotations')
    '''Create model'''
    initial_epoch = 0
    if not config['checkpoint_path']:
        model = PoseNet(sigma=config['sigma'], paf_stages=config['paf_stages'], keypoint_stages=config['heat_stages'],
                        output_shape=config['image_shape'])
    else:
        model = PoseNet(sigma=config['sigma'], paf_stages=config['paf_stages'], keypoint_stages=config['heat_stages'],
                        output_shape=config['image_shape'], upscale=config['upscale'])
        model.load_state_dict(torch.load(config['checkpoint_path']))
        initial_epoch = int(config['checkpoint_path'].split('_')[-1].split('.')[0])

    train(model, dataset=data, epochs=config["epochs"],
          paf_output=config['paf_output'],
          initial_epoch=initial_epoch,
          batch_size=config['batch_size'],
          paf_weight=config['paf_weight'],
          heat_weight=config['heat_weight'],
          log_every_iter=config['log_every_iter'],
          save_every_epoch=config['save_every_epoch'],
          image_every_iter=config['image_every_iter']
          )