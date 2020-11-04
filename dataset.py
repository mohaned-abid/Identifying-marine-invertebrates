import pandas as pd
import joblib
import numpy as np
from PIL import Image
import albumentations
import torch 
import cv2
class  marineDatasetTrain:
    def __init__(self,folds,img_height,img_width,mean,std):
        df=pd.read_csv("../input/train_folds.csv")
        df=df[["direction","label","kfold"]]

        df=df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids=df.direction.values
        self.label=df.label.values
        if len(folds)==1:
                self.aug=albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
        ])
        else:
               self.aug=albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(mean,std,always_apply=True)
        ])        



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,item):
        image=joblib.load(f"{self.image_ids[item]}.pkl")
        image=image.reshape((384,512,3)).astype(float)
        #image=cv2.resize(image,(384,512)).astype(float)
        #image=Image.fromarray(image).convert("RGB")
        #image.shape
        image=self.aug(image=np.array(image))["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        return{
            'image':torch.tensor(image,dtype=torch.float),
            'label':torch.tensor(self.label[item],dtype=torch.long),
        }
