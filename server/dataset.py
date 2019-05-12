'''
Utils for creating dataset.
Implementation of dataset.
'''

import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data.dataset import Dataset as torchDataset

def traverseData(datapath, categories, outpath=''):
    '''
    Traverse the dataset and return a dict contains the information of dataset.
    dict format:
        {label: [list of image path]}
    '''
    res = {}
    for label, c in enumerate(categories):
        res[label] = []
        imgdir = os.path.join(datapath, c)
        imglist = os.listdir(imgdir)
        for img in imglist:
            res[label].append(os.path.join(imgdir, img))
    return res

def formatData(imgDict, train='', test=''):
    '''
    Generate txt file for train and test data.
    imgDict is the return value from traverseData.
    Format of file:
        image path, label
    '''
    X, y = [], []
    for label, imglist in imgDict.items():
        for imgpath in imglist:
            X.append(imgpath)
            y.append(label)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    if train!='':
        f = open(train, 'w', encoding='utf-8')
        for i in range(len(X_train)):
            f.write(X_train[i]+'\t'+str(y_train[i])+'\n')
        f.close()
    if test!='':
        f = open(test, 'w', encoding='utf-8')
        for i in range(len(X_test)):
            f.write(X_test[i]+'\t'+str(y_test[i])+'\n')
        f.close()
    return X_train,X_test,y_train,y_test

#Dataset class for model training
class MyDataset(torchDataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        # datatxt: file path created by formatData function
        #       Format: image path \t label
        # transform: Data reinforcement method.
        super(MyDataset,self).__init__()        
        fh = open(datatxt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index): 
        fn, label = self.imgs[index] 
        img = Image.open(fn).convert('RGB')
 
        if self.transform is not None:
            img = self.transform(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)
    
def im2tensor(imgpath, data_transform):
    img=Image.open(imgpath).convert('RGB')
    img = data_transform(img)
    return img.unsqueeze(0)