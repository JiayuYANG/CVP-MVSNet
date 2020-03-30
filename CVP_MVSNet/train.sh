# Shell script for training the CVP-MVSNet
# by: Jiayu Yang
# date: 2019-08-13

# Dataset
DATASET_ROOT="./dataset/dtu-train-128/"

# Logging
CKPT_DIR="./checkpoints/"
LOG_DIR="./logs/"

python3 train.py \
\
--info="train_dtu_128" \
--mode="train" \
\
--dataset_root=$DATASET_ROOT \
--imgsize=128 \
--nsrc=2 \
--nscale=2 \
\
--epochs=40 \
--lr=0.001 \
--lrepochs="10,12,14,20:2" \
--batch_size=16 \
\
--loadckpt='' \
--logckptdir=$CKPT_DIR \
--loggingdir=$LOG_DIR \
--resume=0 \
