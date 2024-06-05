echo '[INFO] Beginning data loading...'
python data.py --dataset 'celeba' --train_len 20000 --val_len 100 --hash_func 'photodna' --skip 1
echo '[INFO] Data loading complete, running training now...'
python train.py --dataset 'celeba' --rgb 1 --epochs 1 --batch_size 64 --hash_func 'photodna' --learning_rate 0.0003
echo '[INFO] Training complete...'  