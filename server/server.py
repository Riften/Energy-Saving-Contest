'''
Script running on server .
 - Listen the request from client.
 - Obtain and save imges.
 - Classify images using CNN model.
 - return classification result.
'''

from flask import request, Flask
import time
import os
import config as conf
import torch
from utils import getModel, getTransform, Logger, getTimeStr
from dataset import im2tensor
import argparse

#build net
net = getModel('resnet18', num_classes=conf.num_classes, pretrained=False)
transform = getTransform(reinforcement=True)

#Create Logger
log_dir= 'logs'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
timestr = getTimeStr()
serverlog = os.path.join(log_dir, 'serverlog_%s.txt'%timestr)
logger = Logger(serverlog)

#For Convenient, net and logger are set to be global variable
#In that case, you may need to change argument of getModel manually


def classification(imtensor):
    out = net(imtensor)
    _, predicted = torch.max(out.data, 1)
    return predicted.item()

app = Flask(__name__)

@app.route("/", methods=['POST'])
def get_frame():
    logger.record('Obtain an image from client.')
    start_time = time.time()
    upload_file = request.files['file']
    file_name = upload_file.filename
    if upload_file:
        file_path = os.path.join(conf.savepath, file_name)
        upload_file.save(file_path)
        logger.record('Image saved to %s' % file_path)
        
        #Doing classification
        logger.record('Classify the image using trained model')
        imtensor = im2tensor(file_path, transform['test'])
        predict = classification(imtensor)
        duration = time.time() - start_time
        logger.record('Prediction: %d, %s' % (predict, conf.categories[predict]))
        logger.record('duration:[%.0fms]' % (duration*1000))
        
        return str(predict)
    else:
        logger.record('No file obtained')
        return str(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path of the trained model", action='store', type=str)

    args = parser.parse_args()
    
    #parse arguments
    modelpath = args.model

    #Create logger, log files

    
    #Build Model
    logger.record('build model from %s' % modelpath)
    model_dict = torch.load(modelpath)
    
    net.load_state_dict(model_dict['net'])
    
    app.run("0.0.0.0", port=5000, debug=True)
    