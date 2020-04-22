import argparse
import math
import os
import random
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet34

from aum import AUMCalculator, DatasetWithIndex


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int):
    """
    Sets random, numpy, torch, and torch.cuda seeds
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_step(args, summary_writer, metrics, aum_calculator, log_interval, batch_step, num_batches,
               batch, epoch, num_epochs, global_step, model, optimizer, device):
    start = time.time()
    model.train()
    with torch.enable_grad():
        optimizer.zero_grad()

        input, target, sample_ids = batch
        input = input.to(device)
        target = target.to(device)

        # Compute output
        output = model(input)
        loss = F.cross_entropy(output, target)

        # Compute gradient and optimize
        loss.backward()
        optimizer.step()

        # Measure accuracy & record loss
        end = time.time()
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error = torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size

        metrics['error'].update(error, batch_size)
        metrics['loss'].update(loss.item(), batch_size)
        metrics['batch_time'].update(end - start)

        # Update AUM
        aum_calculator.update(output, target, sample_ids.tolist())

        # log to tensorboard
        summary_writer.add_scalar('train/error', metrics['error'].val, global_step)
        summary_writer.add_scalar('train/loss', metrics['loss'].val, global_step)
        summary_writer.add_scalar('train/batch_time', metrics['batch_time'].val, global_step)

        # log to console
        if (batch_step + 1) % log_interval == 0:
            results = '\t'.join([
                'TRAIN',
                f'Epoch: [{epoch}/{num_epochs}]',
                f'Batch: [{batch_step}/{num_batches}]',
                f'Time: {metrics["batch_time"].val:.3f} ({metrics["batch_time"].avg:.3f})',
                f'Loss: {metrics["loss"].val:.3f} ({metrics["loss"].avg:.3f})',
                f'Error: {metrics["error"].val:.3f} ({metrics["error"].avg:.3f})',
            ])
            print(results)


def eval_step(args, regime, metrics, log_interval, batch_step, num_batches, batch, epoch,
              num_epochs, model, device):
    start = time.time()
    model.eval()
    with torch.no_grad():
        input, target, sample_ids = batch
        input = input.to(device)
        target = target.to(device)

        # Compute output
        output = model(input)
        loss = F.cross_entropy(output, target)

        # Measure accuracy & record loss
        end = time.time()
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error = torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size

        metrics['error'].update(error, batch_size)
        metrics['loss'].update(loss.item(), batch_size)
        metrics['batch_time'].update(end - start)

        # log to console
        if (batch_step + 1) % log_interval == 0:
            results = '\t'.join([
                regime,
                f'Epoch: [{epoch}/{num_epochs}]',
                f'Batch: [{batch_step}/{num_batches}]',
                f'Time: {metrics["batch_time"].val:.3f} ({metrics["batch_time"].avg:.3f})',
                f'Loss: {metrics["loss"].val:.3f} ({metrics["loss"].avg:.3f})',
                f'Error: {metrics["error"].val:.3f} ({metrics["error"].avg:.3f})',
            ])
            print(results)


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data-dir', type=str, default='./', help='where to download dataset')
    parser.add_argument('--valid-size',
                        type=int,
                        default=5000,
                        help='num samples in validation set')

    # Output/logging file
    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        help='how many steps between logging to the console')
    parser.add_argument('--output-dir',
                        type=str,
                        default='./output',
                        help='where to save out the model, must be an existing directory.')

    parser.add_argument('--detailed-aum',
                        action='store_true',
                        help='if set, the AUM calculations will be done in non-compressed mode')

    # Optimizer params
    parser.add_argument('--learning-rate', type=float, default=0.1, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')

    # Training Regime params
    parser.add_argument('--num-epochs',
                        type=int,
                        default=150,
                        help='number of epochs to train over')
    parser.add_argument('--train-batch-size', type=int, default=64, help='size of training batch')

    # Validation Regime params
    parser.add_argument('--val-batch-size', type=int, default=64, help='size of val batch')

    args = parser.parse_args()
    return args


def main(args):
    pprint(vars(args))

    # Setup experiment folder structure
    # Create output folder if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # save out args
    with open(os.path.join(args.output_dir, 'args.txt'), 'w+') as f:
        pprint(vars(args), f)

    # Setup summary writer
    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tb_logs'))

    # Set seeds
    set_seed(42)

    # Load dataset
    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = datasets.CIFAR100(args.data_dir,
                                  train=True,
                                  transform=train_transforms,
                                  download=True)
    val_set = datasets.CIFAR100(args.data_dir, train=True, transform=test_transforms)
    test_set = datasets.CIFAR100(args.data_dir, train=False, transform=test_transforms)

    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - args.valid_size]
    valid_indices = indices[len(indices) - args.valid_size:]
    train_set = torch.utils.data.Subset(train_set, train_indices)
    val_set = torch.utils.data.Subset(val_set, valid_indices)

    train_set = DatasetWithIndex(train_set)
    val_set = DatasetWithIndex(val_set)
    test_set = DatasetWithIndex(test_set)

    val_loader = DataLoader(val_set,
                            batch_size=args.val_batch_size,
                            shuffle=False,
                            pin_memory=(torch.cuda.is_available()))
    test_loader = DataLoader(test_set,
                             batch_size=args.val_batch_size,
                             shuffle=False,
                             pin_memory=(torch.cuda.is_available()))

    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet34(num_classes=100)
    model = model.to(device)
    num_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(model)
    f'Number of parameters: {num_params}'

    # Create optimizer & lr scheduler
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters,
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True)
    milestones = [0.5 * args.num_epochs, 0.75 * args.num_epochs]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Keep track of AUM
    aum_calculator = AUMCalculator(args.output_dir, compressed=(not args.detailed_aum))

    # Keep track of things
    global_step = 0
    best_error = math.inf

    print('Beginning training')
    for epoch in range(args.num_epochs):

        train_loader = DataLoader(train_set,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  pin_memory=(torch.cuda.is_available()),
                                  num_workers=0)

        train_metrics = {
            'loss': AverageMeter(),
            'error': AverageMeter(),
            'batch_time': AverageMeter()
        }
        num_batches = len(train_loader)
        for batch_step, batch in enumerate(train_loader):
            train_step(args, summary_writer, train_metrics, aum_calculator, args.log_interval,
                       batch_step, num_batches, batch, epoch, args.num_epochs, global_step, model,
                       optimizer, device)

            global_step += 1

        scheduler.step()

        val_metrics = {
            'loss': AverageMeter(),
            'error': AverageMeter(),
            'batch_time': AverageMeter()
        }
        num_batches = len(val_loader)
        for batch_step, batch in enumerate(val_loader):
            eval_step(args, 'VAL', val_metrics, args.log_interval, batch_step, num_batches, batch,
                      epoch, args.num_epochs, model, device)

        # log eval metrics to tensorboard
        summary_writer.add_scalar('val/error', val_metrics['error'].avg, global_step)
        summary_writer.add_scalar('val/loss', val_metrics['loss'].avg, global_step)
        summary_writer.add_scalar('val/batch_time', val_metrics['batch_time'].avg, global_step)

        # Save best model
        if val_metrics['error'].avg < best_error:
            best_error = val_metrics['error'].avg
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))

    # Finalize aum calculator
    aum_calculator.finalize()

    # Eval best model on on test set
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pt')))
    test_metrics = {'loss': AverageMeter(), 'error': AverageMeter(), 'batch_time': AverageMeter()}
    num_batches = len(test_loader)
    for batch_step, batch in enumerate(test_loader):
        eval_step(args, 'TEST', test_metrics, args.log_interval, batch_step, num_batches, batch, -1,
                  -1, model, device)

    # log eval metrics to tensorboard
    summary_writer.add_scalar('test/error', test_metrics['error'].avg, global_step)
    summary_writer.add_scalar('test/loss', test_metrics['loss'].avg, global_step)
    summary_writer.add_scalar('test/batch_time', test_metrics['batch_time'].avg, global_step)

    # log test metrics to console
    results = '\t'.join([
        'FINAL TEST RESULTS',
        f'Loss: {test_metrics["loss"].avg:.3f}',
        f'Error: {test_metrics["error"].avg:.3f}',
    ])
    print(results)

"""
A demo to show how to calculate AUM while training a ResNet on CIFAR100.
"""
if __name__ == '__main__':
    args = parse_args()
    main(args)
