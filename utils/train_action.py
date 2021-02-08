import torch
import logging
import torch.nn as nn
import torch.optim as optim
from dataset.video_dataset import VideoDataset
from model.action_network import ActionModel
import os
import sys

sys.path.append(os.getcwd())
from utils.setup_logging import setup_logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
from scripts.action_eval import eval, plot_confusion_matrix, calculate_metrics


def train(net, dataset, epochs=1, initial_epoch=0, batch_size=18, log_every_iter=100, save_every_epoch=3):
    criterion_pose = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
    criterion_app = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    optimizer_pose = optim.Adam(params=net.pose_action_network.parameters(), lr=0.0005)
    optimizer_app = optim.Adam(params=net.appearance_action_network.parameters(), lr=0.0005)
    torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(log_dir='logs/action_logs/')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    logging.info('Starting training')
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss_app, running_loss_pose = 0.0, 0.0
        iteration = 0

        logging.info(f'epoch: {epoch+1+initial_epoch}')
        gt_class_ids, pose_class_ids, app_class_ids, combined_class_ids = [], [], [], []
        for i_batch, batch in tqdm(enumerate(dataloader)):
            iteration += 1
            pose_features, app_features, act_class = batch[0], batch[1], batch[2]
            pose_features = torch.transpose(pose_features, -1, 1)
            app_features = torch.transpose(app_features, -1, 1)

            if epoch == 0 and i_batch == 0:
                writer.add_graph(net.appearance_action_network, app_features.cuda())
            gt_class_ids += list(act_class.cpu().detach())
            # zero the parameter gradients
            optimizer_pose.zero_grad()
            optimizer_app.zero_grad()

            app_out1, app_out2 = net.appearance_action_network(app_features.cuda())
            pose_out1, pose_out2 = net.pose_action_network(pose_features.cuda())

            app_out1 = torch.squeeze(app_out1)
            app_out2 = torch.squeeze(app_out2)
            app_loss = criterion_app[0](app_out1.cuda(), act_class.cuda()) + criterion_app[1](app_out2.cuda(), act_class.cuda())

            pose_out1 = torch.squeeze(pose_out1)
            pose_out2 = torch.squeeze(pose_out2)
            pose_loss = criterion_pose[0](pose_out1.cuda(), act_class.cuda()) + criterion_pose[1](pose_out2.cuda(), act_class.cuda())

            pose_class_ids += list(torch.argmax(pose_out2, dim=-1).cpu().detach())
            app_class_ids += list(torch.argmax(app_out2, dim=-1).cpu().detach())
            out = (pose_out2 + app_out2) / 2
            combined_class_ids += list(torch.argmax(out, dim=-1).cpu().detach())

            app_loss.backward()
            optimizer_app.step()

            pose_loss.backward()
            optimizer_pose.step()

            # print statistics
            running_loss_pose += pose_loss.item()
            running_loss_app += app_loss.item()

            if iteration % log_every_iter == log_every_iter-1:  # print every 2000 mini-batche
                '''losses'''
                writer.add_scalar('Train/Pose action/loss', running_loss_pose/log_every_iter, iteration +
                                  (epoch+initial_epoch)*dataset.__len__()/batch_size)
                writer.add_scalar('Train/Appearance action/loss', running_loss_app / log_every_iter, iteration +
                                  (epoch + initial_epoch) * dataset.__len__() / batch_size)
                running_loss_app, running_loss_pose = 0.0, 0.0
                '''accuracy'''
                pose_acc, _ = calculate_metrics(gt_class_ids, pose_class_ids)
                app_acc, _ = calculate_metrics(gt_class_ids, app_class_ids)
                combined_acc, confusion = calculate_metrics(gt_class_ids, combined_class_ids)
                gt_class_ids, pose_class_ids, app_class_ids, combined_class_ids = [], [], [], []
                writer.add_scalar('Train/Pose action/accuracy', pose_acc, iteration +
                                  (epoch + initial_epoch) * dataset.__len__() / batch_size)
                writer.add_scalar('Train/Appearance action/accuracy', app_acc, iteration +
                                  (epoch + initial_epoch) * dataset.__len__() / batch_size)
                writer.add_scalar('Train/Combined action/accuracy', combined_acc, iteration +
                                  (epoch + initial_epoch) * dataset.__len__() / batch_size)
                confusion_image = plot_confusion_matrix(confusion)
                writer.add_image(f'Train/Input epoch: {epoch + initial_epoch + 1} iteration: {iteration}',
                                 torch.tensor(torch.FloatTensor(confusion_image)), 0)

        if (epoch + initial_epoch) % save_every_epoch == 0:
            torch.save(net.pose_action_network.state_dict(),
                       f'checkpoints/action_network/checkpoint_pose_{epoch + 1 + initial_epoch}.pth')
            torch.save(net.appearance_action_network.state_dict(),
                       f'checkpoints/action_network/checkpoint_appearance_{epoch + 1 + initial_epoch}.pth')
            '''evaluate on validation'''
            logging.info('Validating')
            net.appearance_action_network.eval()
            net.pose_action_network.eval()
            pose_acc, app_acc, combined_acc, confusion = eval(batch_size=int(batch_size / 2), net=net,
                                                              print_logs=False)
            net.appearance_action_network.train()
            net.pose_action_network.train()
            writer.add_scalar('Validation/Pose action/accuracy', pose_acc, iteration +
                              (epoch + initial_epoch) * dataset.__len__() / batch_size)
            writer.add_scalar('Validation/Appearance action/accuracy', app_acc, iteration +
                              (epoch + initial_epoch) * dataset.__len__() / batch_size)
            writer.add_scalar('Validation/Combined action/accuracy', combined_acc, iteration +
                              (epoch + initial_epoch) * dataset.__len__() / batch_size)
            confusion_image = plot_confusion_matrix(confusion)
            writer.add_image(f'Validation/Input epoch: {epoch + initial_epoch + 1} iteration: {iteration}',
                             torch.tensor(torch.FloatTensor(confusion_image)), 0)
            logging.info(f'Checkpoint saved')

        logging.info(f'epoch: {initial_epoch + epoch + 1}, pose_loss: {running_loss_pose / log_every_iter}, appearance loss: '
                     f'{running_loss_app / log_every_iter}')

    print('Finished Training')


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Extract hand positions from the image')
    parser.add_argument('--config-path', type=str, default='action_config.json', help='Path to the training config')

    setup_logging()
    args = parser.parse_args()
    config = json.load(open(args.config_path, 'r'))
    '''Load dateset'''
    data = VideoDataset()
    logging.info('Loaded video features')
    '''Create model'''
    initial_epoch = 0

    model = ActionModel()
    if config['app_checkpoint_path'] and config['pose_checkpoint_path']:
        model.appearance_action_network.load_state_dict(torch.load(config['app_checkpoint_path']))
        model.pose_action_network.load_state_dict(torch.load(config['pose_checkpoint_path']))
        initial_epoch = int(config['app_checkpoint_path'].split('_')[-1].split('.')[0])

    train(model, dataset=data, epochs=config["epochs"],
          initial_epoch=initial_epoch,
          batch_size=config['batch_size'],
          log_every_iter=config['log_every_iter'],
          save_every_epoch=config['save_every_epoch'])