import pandas as  pd
import numpy as np
import joblib
import glob
from tqdm import tqdm
import cv2

if __name__=="__main__":
    df=pd.read_csv('../input/train_folds.csv')
    for imagedir in list(df.direction):
        print(imagedir)
        img = cv2.imread(imagedir)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(384,512))
        img=np.array(img).flatten()
        image_id=imagedir.split('/')[-1]
        joblib.dump(img,f"../input/image_pickles/{image_id}.pkl")
    '''
    files=glob.glob("../input/train_images/*.jpeg")
    print(files)
    for f in files:
        print(f)
        df=pd.read_parquet(f)
        image_ids=df.image_id.values
        df=df.drop("image_id",axis=1)
        image_array=df.values
        for j,img_id in tqdm(enumerate(image_ids),total=len(image_ids)):
            joblib.dump(image_array[j,:],f"../input/image_pickles/{img_id}.pkl")'''
