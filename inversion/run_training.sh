echo '[INFO] Beginning data loading...'
python data.py --dataset 'celeba' --train_len 10000 --val_len 100 --skip 0
echo '[INFO] Data loading complete, running training now...'
python train.py --dataset 'celeba' --rgb 1 --epochs 50 --batch_size 64 --learning_rate 0.0005
echo '[INFO] Training complete...'  