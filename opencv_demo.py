import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import cv2
import matplotlib as plt
import numpy as np

# load the model
model = torch.load('vgg.pth',map_location=torch.device('cpu'))


face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'unmasked',1:'masked',2:'incorrectly_masked'}
color_dict={0:(255,0,0),1:(0,255,0),2:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(img,1.3,5)
#     plt.imshow(gray)



    for x,y,w,h in faces:
        face_img=img[y:y+w,x:x+w]

        # resize image and normalize it 
        resized=cv2.resize(face_img,(64,64))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = resized.transpose((2,0,1))
        resized = torch.as_tensor(resized,dtype = torch.float32)
        resized = (resized-0.5)/0.5
        reshaped=resized.view(1,3,64,64)

        # make prediction
        result=model(reshaped)
        pred = result.argmax(dim = 1,keepdim = True)

        label = int(pred[0][0])

        # draw the box with the predicted label
        cv2.rectangle(img,(x,y),(x+w,y+h),0,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(
          img, labels_dict[label], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        break
        
        
    cv2.imshow('LIVE_DEMO',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()

