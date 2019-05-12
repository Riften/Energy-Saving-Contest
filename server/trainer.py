#pytorch trainer and evaluator
from utils import  getTimeStr
import torch
import os

class Trainer():
    def __init__(self, train_loader, test_loader, net, logger, optimizer, criterion, device='cuda:0', lr_scheduler=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net = net
        self.logger = logger
        self.optimizer=optimizer
        self.criterion=criterion
        self.device=device
        if lr_scheduler is not None:
            self.decay=True
            self.lr_scheduler = lr_scheduler
        else:
            self.decay=False
        
    def train(self, epoches, saveepoch, savedir, writer=None, model=None):
        #epoches: number of epoches to train
        #saveepoch: number of epoches to save the model

        if model is not None:
            # load model
            self.logger.record('Train model from %s' % model)
        else:
            self.logger.record('Train model from scratch.')
        for epoch in range(epoches):
            if self.decay:
                self.lr_scheduler.step()
            train_loss = self.train_epoch(epoch)
            test_loss, test_accuracy=self.test_epoch()
            self.logger.record('Test: Loss: {:.4f},  Accuracy: {:.0f}%'.format(test_loss, 100 * test_accuracy))
            if writer is not None:
                self.logger.record('Draw tensorboard')
                writer.add_scalar('scalar/train loss', train_loss, epoch)
                writer.add_scalar('scalar/test loss', test_loss, epoch)
                writer.add_scalar('scalar/test acc', test_accuracy, epoch)

            if epoch % saveepoch==0:
                state = {'net':self.net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
                savepath = os.path.join(savedir, 'model_%s.pth' % getTimeStr())
                torch.save(state, savepath)
                self.logger.record('Save model to %s' % savepath)
    def train_epoch(self, epoch):
        self.net.train()
        running_loss=0.0
        total_loss=0.0
        total_data = len(self.train_loader.dataset)
        for i, data in enumerate(self.train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # print  and log statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 40 == 39:
                self.logger.record('[epoch:%d, %d/%d] loss: %.3f' %
                      ( epoch+1, i + 1, total_data+1, running_loss / 40))
                running_loss = 0.0
        return total_loss / total_data
    
    def test_epoch(self):
        self.net.eval()
        test_loss=0.0
        correct=0.0
        total_data = len(self.test_loader.dataset)
        for i, data in enumerate(self.test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # forward
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # print  and log statistics
            test_loss += loss.item()

        return test_loss / total_data, correct / total_data    
