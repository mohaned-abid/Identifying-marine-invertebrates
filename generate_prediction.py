import sys
#pt_models="../input/pretrained-models/pretrained-models.pytorch-master/"
#sys.path.insert(0,pt_models)
import pretrainedmodels

import glob
import torch 
import albumentations
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F
import cv2
TEST_BATCH_SIZE=1
MODEL_MEAN=(0.485,0.456,0.406)
MODEL_STD=(0.229,0.224,0.225)
IMG_HEIGHT=384
IMG_WIDTH=512
DEVICE="cuda"

import glob
class ResNet50(nn.Module):
    def __init__(self,pretrained):
        super(ResNet50,self).__init__()
        if pretrained is True:
            self.model=pretrainedmodels.__dict__['resnet50'](pretrained="imagenet")
        else:
            self.model=pretrainedmodels.__dict__['resnet50'](pretrained=None)
        self.l0=nn.Linear(2048,137)
    def forward(self,x):
        bs=x.shape[0]
        x=self.model.features(x)
        x=F.adaptive_avg_pool2d(x,1).reshape(bs, -1)
        l0=self.l0(x)
        return l0

class  marineDatasetTest:
    def __init__(self,img_height,img_width,mean,std):
        self.image_ids=np.array(glob.glob("../input/test_small/*"))
        self.aug=albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
        ])



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,item):
        image=cv2.imread(self.image_ids[item])
        img_id=self.image_ids[item]
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(384,512))
        image=image.reshape((384,512,3)).astype(float)
        image=self.aug(image=np.array(image))["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float)
        return{
            'image':torch.tensor(image,dtype=torch.float),
            'image_id':img_id
        }

list_model=[0]*5
for i in range(5):
    list_model[i]=ResNet50(pretrained=False)
    list_model[i].load_state_dict(torch.load(f"../input/models/resnet50_fold({i}).bin"))
    list_model[i].eval()
    list_model[i]=list_model[i].to(DEVICE)
predictions=[]
idds=[]
dataset=marineDatasetTest(img_height=IMG_HEIGHT,img_width=IMG_WIDTH,mean=MODEL_MEAN,std=MODEL_STD)
data_loader=torch.utils.data.DataLoader(dataset,batch_size=TEST_BATCH_SIZE,shuffle=True,num_workers=4)
dff=pd.read_csv('../input/ssm.csv')
ll=list(dff.columns)
dff=pd.DataFrame(columns=ll)
for bi, d in enumerate(data_loader):
        image=d["image"]
        img_id=d["image_id"][0].split('/')[-1]
        print(img_id)
        image=image.to(DEVICE,dtype=torch.float)
        c=list_model[0](image)
        for i in range(1,5):
            c1 =list_model[i](image)
            c=c+c1
        dff.loc[bi,'FILE']=img_id
        c=F.softmax(c)
        c=c.cpu().detach().numpy()
        c=list(c[0])
        dff.loc[bi,ll[1]:]=list(c)
dff.to_csv('../input/sub.csv')
        
