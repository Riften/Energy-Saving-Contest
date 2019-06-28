'''
Several useful utils.
'''

import time
import logging
from logging import handlers
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms

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
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return {'train': tmp_transform, 'test': tmp_transform}

def getTimeStr():
    # return a string contining time info
    # used for file or log names
    # format: month_day_Hour_Min
    return time.strftime("%m_%d_%H_%M", time.localtime(time.time()))

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#mapping for log level

    def __init__(self, filename, level='info',when='D', interval=1, backCount=5):
        '''
        filename: Log file path
        level: Log level. Logs only contain information higher or equal to this level
        when:  Time Unit
            S: Seconds
            M: Minutes
            H: Hours
            D: Days
            W: Week day (0=Monday)
        interval: Number of time units that logger will split the log file.
        backCount: Number of log file matained.
        '''
        #fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        fmt_console = logging.Formatter('%(levelname)s: %(message)s') # Format for console
        fmt_file = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s') # Format for log file
        sh = logging.StreamHandler()
        sh.setFormatter(fmt_console)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when, interval = interval, backupCount=backCount,encoding='utf-8')
        th.setFormatter(fmt_file)
        
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level))
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
    
    def record(self, info):
        self.logger.info(info)
