import argparse
from utils import  Logger, getTimeStr
import os
from dataset import traverseData, formatData, MyDataset
import config as cfg
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.models as models

from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_data", help="Whether to create data files for training and testing.",action='store_true')
    parser.add_argument("-c", "--checkpoint_dir", help="Folder to save checkpoints", action='store', type=str, default='checkpoints')
    parser.add_argument("-l", "--logs", help="log dir", action='store', type=str, default='logs')
    
    args = parser.parse_args()
    
    #recreate data if necessary
    if args.create_data:
        dataset = traverseData(cfg.datapath, cfg.categories)
        formatData(dataset, 'train.txt', 'test.txt')

    #parse other arguments
    ckp_dir=args.checkpoint_dir
    log_dir=args.logs

    #Create logger, log files, and ckp folder
    if not os.path.exists(ckp_dir):
        os.mkdir(ckp_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    timestr = getTimeStr()
    mainlog = os.path.join(log_dir, 'mainlog_%s.txt'%timestr)
    tblog = os.path.join(log_dir, 'tblog_%s'%timestr)
    logger = Logger(mainlog)
    if not os.path.exists(tblog):
        os.mkdir(tblog)

    #create dataset and dataloader
    train_data=MyDataset('train.txt', transform=transforms.ToTensor())
    test_data=MyDataset('test.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size)
    device='cuda:0'
    net = models.vgg11(num_classes = cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9)
    lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    
    tboardWriter = SummaryWriter(tblog)
    trainer = Trainer(train_loader, test_loader, net, logger, optimizer, criterion, device='cuda:0', lr_scheduler=lr_scheduler)
    trainer.train(500, 1, ckp_dir, writer=tboardWriter)
    tboardWriter.close()
