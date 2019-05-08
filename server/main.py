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
    #create log file
    timestr = getTimeStr()
    logdir = 'logs_'+timestr
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logpath = os.path.join(logdir, 'mainLog.txt')
    logger = Logger(logpath)
    #-----
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_data", help="Whether to create data files for training and testing.",action='store_true')
    args = parser.parse_args()
    
    #recreate data if necessary
    if args.create_data:
        logger.record("Create train and test index files from %s" % cfg.datapath)
        dataset = traverseData(cfg.datapath, cfg.categories)
        formatData(dataset, 'train.txt', 'test.txt')
        
    #create dataset and dataloader
    train_data=MyDataset('train.txt', transform=transforms.ToTensor())
    test_data=MyDataset('test.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size)
    device='cuda:0'
    net = models.vgg11(num_classes = 6).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    tboardDir = "board_log_"+getTimeStr()
    if not os.path.exists(tboardDir):
        os.mkdir(tboardDir)
    tboardWriter = SummaryWriter(tboardDir)
    trainer = Trainer(train_loader, test_loader, net, logger, optimizer, criterion, device='cuda:0', lr_scheduler=lr_scheduler)
    trainer.train(500, 1, 'models', writer=tboardWriter)