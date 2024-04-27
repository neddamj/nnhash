echo '[INFO] Beginning data loading...'
python data.py --dataset 'celeba' --train_len 200 --val_len 20 --skip 0
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 1 --display 1 --hash_func 'pdq' --path /your/file/path/here/