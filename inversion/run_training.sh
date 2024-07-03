echo '[INFO] Beginning data loading...'
python data.py --dataset 'stl10' --train_len 40000 --val_len 500 --hash_func 'neuralhash' --skip 0
echo '[INFO] Data loading complete, running training now...'
python train.py --dataset 'stl10' --rgb 1 --epochs  50 --batch_size 64 --hash_func 'neuralhash' --perturbation 0.0 --learning_rate 0.0003
echo '[INFO] Training complete...'  