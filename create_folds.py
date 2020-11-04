import pandas  as pd
from sklearn.model_selection import StratifiedKFold
import glob
import numpy as np
if __name__=="__main__":

	df=pd.read_csv("../input/ssm.csv")
	l=list(df.columns)[1:]
	d={}
	for key,value in enumerate(l):
		d[value]=key
	folders=glob.glob("../input/train_small/*")
	dirr,label=[],[]
	for folder in folders:
		image_files=glob.glob(f"{folder}/*.jpeg")
		x=folder.split('/')[-1]
		for image_file in image_files:
			dirr.append(image_file)
			label.append(d[x])
	df1=pd.DataFrame({'direction':dirr,'label':label})
	df1.loc[:,'kfold']=-1
	df1=df1.sample(frac=1).reset_index(drop=True)
	skf=StratifiedKFold(n_splits=5)
	for fold, (trn_,val_) in enumerate(skf.split(np.array(dirr),np.array(label))):
		print("TRAIN: ",trn_,"VAL :",val_)
		df1.loc[val_,"kfold"]=fold	
	df1.to_csv('../input/train_folds.csv')

	