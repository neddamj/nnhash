echo '[INFO] Beginning data loading...'
python data.py --dataset 'mnist' --train_len 10000 --val_len 100 --skip 0
echo '[INFO] Data loading complete, running training now...'
python train.py --rgb True --epochs 50 --batch_size 50 --learning_rate 0.0005
echo '[INFO] Training complete...'