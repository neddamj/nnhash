echo '[INFO] Beginning data loading...'
python data.py --dataset 'celeba' --train_len 20000 --val_len 100 --hash_func 'neuralhash' --skip 1
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 1 --display 1 --hash_func 'photodna' --dataset 'stl10' --perturb 0.0 --path /path/to/inversion/model