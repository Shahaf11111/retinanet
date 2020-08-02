import glob
import os
import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

import re

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def get_last_model_path(depth, search_dir='retinanet/runs'):
    last_list = glob.glob(f'{search_dir}/*.pt', recursive=True)
    last_list = [path for path in last_list if str(depth) in path]
    return max(last_list, key=os.path.getctime)


def get_last_run_dir(search_dir='retinanet/runs'):
    last_list = glob.glob(f'{search_dir}/*', recursive=True)
    last_list = [p for p in last_list if os.path.isdir(p)]
    return max(last_list, key=os.path.getctime) if len(last_list) > 0 else None


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--batch')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--resume', action='store_true', help='Resume training for an existing PT file (optional)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)
    if not os.path.exists('retinanet/runs'):
        os.mkdir('retinanet/runs')
    if parser.resume is False:
        run_dir = get_last_run_dir()
        if run_dir is None:
            run_dir = 'retinanet/runs/exp0'
        else:
            next_run_number = max([int(n) for n in re.findall(r'\d+', run_dir)]) + 1
            run_dir = run_dir[:-1] + str(next_run_number)
        os.mkdir(run_dir)
    else:
        run_dir = get_last_run_dir()
    print('Using {} directory'.format(run_dir))
    results_file = os.path.join(run_dir, 'retinanet-depth{}.txt'.format(parser.depth))
    checkpoint_file = os.path.join(run_dir, 'retinanet-depth{}.pt'.format(parser.depth))
    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
    batch = int(parser.batch)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    # NOT IN USE:
    # if dataset_val is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch, drop_last=False)
    #     dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True
    start_epoch = 0

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    if parser.resume:
        device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # checkpoint = torch.load(get_last_model_path(parser.depth), map_location=device)
        if checkpoint['model'] is not None:
            retinanet.load_state_dict(checkpoint['model'], strict=False)
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['training_results'] is not None:
            with open(results_file, 'w') as file:
                file.write(checkpoint['training_results'])  # write results.txt
        if checkpoint['epoch'] is not None:
            with open(results_file, 'w') as file:
                start_epoch = checkpoint['epoch'] + 1
        del checkpoint

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        reg_loss = []
        cls_loss = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for iter_num, data in progress_bar:
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                cls_loss.append(float(classification_loss))
                reg_loss.append(float(regression_loss))

                progress_bar.set_description(
                    'Epoch: {}/{} | clsLoss: {:1.5f} |  regLoss: {:1.5f} | runLoss: {:1.5f}'.format(
                        epoch_num, parser.epochs - 1, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            except Exception as e:
                print("Exception: " + e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset\n')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset\n')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

            with open(results_file, 'a') as f:
                f.write('Epoch: {}, mAP: {:1.5f}, clsLoss: {:1.5f}, regLoss: {:1.5f}, runLoss: {:1.5f}\n'.
                         format(epoch_num, mAP, np.mean(cls_loss), np.mean(reg_loss), np.mean(loss_hist)))
            with open(results_file, 'r') as f:
                checkpoint = {
                    'epoch': epoch_num,
                    'loss': np.mean(epoch_loss),
                    'training_results': f.read(),
                    'model': retinanet.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
            torch.save(checkpoint, os.path.join(run_dir, 'retinanet-depth{}.pt'.format(parser.depth)))

        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()


if __name__ == '__main__':
    main()
