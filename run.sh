export CUDA_VISIBLE_DEVICES=0
export IMG_HEIGHT=384
export IMG_WIDTH=512
export EPOCHS=10
export TRAIN_BATCH_SIZE=3
export TEST_BATCH_SIZE=3
export MODEL_MEAN="(0.485,0.456,0.406)"
export MODEL_STD="(0.229,0.224,0.225)"
export BASE_MODEL="resnet50"
export  TRAINING_FOLDS_CSV="../input/train_folds.csv"

export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python3.6 train.py


'''
export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
python3.6 train.py

export TRAINING_FOLDS="(0,1,4,3)"
export VALIDATION_FOLDS="(2,)"
python3.6 train.py

export TRAINING_FOLDS="(0,4,2,3)"
export VALIDATION_FOLDS="(1,)"
python3.6 train.py

export TRAINING_FOLDS="(4,1,2,3)"
export VALIDATION_FOLDS="(0,)"
python3.6 train.py
'''