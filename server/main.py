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

def getModel(modelname, num_classes, pretrained=True):
    if modelname=='resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif modelname=='resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif modelname=='resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError('Unknown model type')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def getTransform(reinforcement=True):
    if reinforcement:
        return {'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    else:
        tmp_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])
        return {'train': tmp_transform, 'test': tmp_transform}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_data", help="Whether to create data files for training and testing.",action='store_true')
    parser.add_argument("-c", "--checkpoint_dir", help="Folder to save checkpoints", action='store', type=str, default='checkpoints')
    parser.add_argument("-l", "--logs", help="log dir", action='store', type=str, default='logs')
    parser.add_argument("-n", "--net", help="structure of network", action='store', type = str, default='resnet18')
    parser.add_argument("--pretrained", help="whether to use pretrained model", action='store_true')
    parser.add_argument("--reinforcement", help="whether to do data reinforcement", action='store_true')
    args = parser.parse_args()
    
    #recreate data if necessary
    if args.create_data:
        dataset = traverseData(cfg.datapath, cfg.categories)
        formatData(dataset, 'train.txt', 'test.txt')

    #parse other arguments
    ckp_dir=args.checkpoint_dir
    log_dir=args.logs
    use_pretrained = args.pretrained
    modelname = args.net
    do_rein = args.reinforcement
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
    data_transform = getTransform(reinforcement=do_rein)
    logger.record("Create data loader")
    if do_rein:
        logger.record("Apply data reinforcement")
    else:
        logger.record("No data reinforcement")
    train_data=MyDataset('train.txt', data_transform['train'])
    test_data=MyDataset('test.txt', data_transform['test'])
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=cfg.batch_size)
    device='cuda:0'
    #net = models.vgg11(num_classes = cfg.num_classes).to(device)
    net = getModel(modelname=modelname, num_classes=cfg.num_classes, pretrained=use_pretrained).to(device)
    if use_pretrained:
        logger.record('Create %s model with pretrained'%modelname)
    else:
        logger.record('Create %s model from scratch'%modelname)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9)
    lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
    
    tboardWriter = SummaryWriter(tblog)
    trainer = Trainer(train_loader, test_loader, net, logger, optimizer, criterion, device='cuda:0', lr_scheduler=lr_scheduler)
    trainer.train(500, 1, ckp_dir, writer=tboardWriter)
    tboardWriter.close()
