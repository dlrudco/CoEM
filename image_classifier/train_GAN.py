import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

from data_utils import *
from train_tools import *

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import math


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)



def save_checkpoint(state, epoch, checkpoint_dir):
    checkpoint_path = f'{checkpoint_dir}checkpoint_{epoch:04}.pth.tar'
    torch.save(state, checkpoint_path)


def accuracy(source, target):
    source = source.max(1)[1].long().cpu()
    target = target.cpu()
    correct = (source == target).sum().item()
    return correct / float(source.shape[0])

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def train_model(args):
    BATCH_SIZE = args.batch_size

    LR = args.lr
    MOMENTUM = args.momentum
    NUM_EPOCHS = args.num_epochs
    WD = args.wd
    PATIENCE = args.patience

    NUM_WORKERS = args.num_workers
    DEVICE = args.device
    
    root = '../Data/quickdraw'
    dataloaders, _ = quickdraw_setter(root=root, batch_size=256, num_workers=10)

    train_loader = dataloaders['train']
    val_loader = dataloaders['test']

    model_G = ImgVAE(z_dim=128, output_units=20)
    model_D = ImgDiscriminator()

    print(model_G.eval())
    criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCELoss(reduction='mean'), torch.nn.KLDivLoss(reduction='mean'),
                 torch.nn.MSELoss()]
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.num_epochs, 0, 10).step)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(args.num_epochs, 0, 10).step)



    experiment_id = str(time.strftime('%Y-%m-%d-%H_%M_%S', time.gmtime()))
    global model_folder
    model_folder = 'experiments/' + str(experiment_id) + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        os.makedirs(model_folder+'images')


    for epoch in range(NUM_EPOCHS):
        # Training iteration
        epoch_start_time = time.time()
        train_loss, acc = run_epoch(model_G, model_D,
                                 criterion,
                                 optimizer_G, optimizer_D,
                                 train_loader,
                                 DEVICE,
                                 is_training=True, epoch=epoch)
        _ = run_epoch(model_G, model_D,
                               criterion,
                               optimizer_G, optimizer_D,
                               val_loader,
                               DEVICE,
                               is_training=False, epoch=epoch)
        # Decrease the learning rate after not improving in the validation set
        scheduler_G.step()
        scheduler_D.step()

        time_stamp = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
        epoch_time = time.time() - epoch_start_time
        lr_G = optimizer_G.param_groups[0]['lr']
        lr_D = optimizer_D.param_groups[0]['lr']
        print(
            'Epoch %d, train loss %g, epoch-time %gs, lr_G %g, lr_D %g, time-stamp %s'
            % (epoch + 1, train_loss, epoch_time, lr_G, lr_D, time_stamp))
        print('')

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict_G': model_G.state_dict(),
            'state_dict_D': model_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }

        # save checkpoint in appropriate path (new or best)
        if (epoch +1) % 10 == 0:
            save_checkpoint(checkpoint, epoch, model_folder)


def run_epoch(model_G, model_D, criterion, optimizer_G, optimizer_D, data_loader, device, is_training, epoch):
    running_loss = {'GAN_LOSS' : 0.0, 'CLS_LOSS': 0.0, 'RECON_LOSS': 0.0, 'KLD_LOSS': 0.0, 'TPLT_LOSS': 0.0, 'TOTAL_LOSS': 0.0}
    if is_training:
        model_G.train()
        model_D.train()
    else:
        model_G.eval()
        model_D.eval()
    n_batches = 0
    model_G.to(device=device)
    model_D.to(device=device)
    pbar = tqdm(data_loader, desc=f'Train Epoch {epoch:02}' if is_training else 'Val')
    epoch_acc = 0
    acc_skip = 0
    triplet = TripletLoss(device)
    fake_buffer = ReplayBuffer()

    for features, labels in pbar:
        labels = labels.long()
        X = features.to(device=device)
        y = labels.to(device=device)

        if is_training:
            optimizer_G.zero_grad()
            y_onehot = torch.zeros((y.size(0), 20))
            for j in range(y_onehot.size(0)):
                y_onehot[j, y[j]] = 1.
            pred_x, pred_z, mu, logvar, _y = model_G(X, is_training=is_training, prior=y_onehot.to(device))

            cls_loss = criterion[0](_y, y)
            running_loss['CLS_LOSS'] += cls_loss.item()

            embed_tplt_loss = 0.001*triplet(pred_z, y)
            running_loss['TPLT_LOSS'] += embed_tplt_loss.item()

            recon_loss = criterion[1](torch.sigmoid(pred_x), torch.sigmoid(X))
            running_loss['RECON_LOSS'] += recon_loss.item()

            kld_loss = torch.mean((mu.pow(2) -1. + logvar.exp() - logvar)/2)
            running_loss['KLD_LOSS'] += kld_loss.item()

            real_gt = torch.ones((X.size(0), 1), requires_grad=False).to(device)
            fake_gt = torch.zeros((X.size(0), 1), requires_grad=False).to(device)
            gan_loss = criterion[3](model_D(pred_x), real_gt)
            running_loss['GAN_LOSS'] += gan_loss.item()

            loss_G = cls_loss + embed_tplt_loss + recon_loss + kld_loss + gan_loss

            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            dis_real_loss = criterion[3](model_D(X), real_gt)
            fake_ = fake_buffer.push_and_pop(pred_x)
            dis_fake_loss = criterion[3](model_D(fake_.detach()), fake_gt)

            loss_D = (dis_fake_loss + dis_real_loss)/2

            loss_D.backward()
            optimizer_D.step()

            if math.isnan(loss_D) or math.isnan(loss_G):
                print('nan!')

            acc = accuracy(_y, y)
        else:
            model_G.eval()
            real = features.to(device)
            y_onehot = torch.zeros((y.size(0), 20))
            for j in range(y_onehot.size(0)):
                y_onehot[j, y[j]] = 1.
            fake, _, _, _, _ = model_G(real, is_training=True, prior=y_onehot.to(device))
            real_img = np.empty((480, 640, 4))
            fake_img = np.empty((480, 640, 4))
            fake = fake.cpu().detach().numpy()
            real = real.cpu().detach().numpy()

            for i in range(real.shape[0]):
                t_real = real[i, 0, :, :]
                f1 = plt.figure()
                plt.imshow(t_real, interpolation='nearest')
                if i == 0:
                    real_img = figure_to_array(f1)
                else:
                    real_img = np.concatenate((real_img,figure_to_array(f1)), axis=1)
                plt.close(f1)

                t_fake = fake[i, 0, :, :]
                plt.imshow(t_fake, interpolation='nearest')
                f2 = plt.figure()
                if i == 0:
                    fake_img = figure_to_array(f2)
                else:
                    fake_img = np.concatenate((fake_img, figure_to_array(f2)), axis=1)
                plt.close(f2)
            real_img = np.concatenate((real_img, fake_img), axis=0)
            # Arange images along x-axis
            cv2.imwrite(f'{model_folder}images/{epoch:05}.png', real_img)
            break

        running_loss['TOTAL_LOSS'] += loss_D.item() + loss_G.item()

        n_batches += 1
        epoch_acc += 100 * acc
    if is_training:
        print(running_loss)
        print(f'\rloss: {running_loss["TOTAL_LOSS"] / n_batches}, acc: {epoch_acc / (n_batches-acc_skip)}')
        return running_loss["TOTAL_LOSS"] / n_batches, epoch_acc
    else:
        return 0


def parse():
    parser = argparse.ArgumentParser(description='GCT634 final project demo python')
    parser.add_argument('--num_workers', '--NUM_WORKERS', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num_epochs', '--NUM_EPOCHS', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', '--BATCH_SIZE', default=32, type=int,
                        metavar='BATCH_SIZE', help='mini-batch size per process (default: 24)')
    parser.add_argument('--lr', '--learning-rate', '--LR', default=0.001, type=float,
                        metavar='LR',
                        help='Initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='MOMENTUM',
                        help='momentum')
    parser.add_argument('--wd', '--wd', '--WD', default=1e-04, type=float,
                        metavar='WD', help='weight decay (default: 5e-4)')
    parser.add_argument('--patience', '-p', default=40, type=int,
                        metavar='PATIENCE', help='lr schedule frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--musicnn_dis', dest='musicnn_dis',action='store_true',
                        help='Use MusiCNN Front-End Mid-End architecture for Discriminator')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    torch.set_default_tensor_type(torch.FloatTensor)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(args)