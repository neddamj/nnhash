echo '[INFO] Beginning data loading...'
python data.py --dataset 'mnist' --train_len 10000 --val_len 100 --hash_func 'neuralhash' --skip 1
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 0 --display 0 --hash_func 'neuralhash' --path /path/to/your/model