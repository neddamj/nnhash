echo '[INFO] Beginning data loading...'
python data.py --dataset 'mnist' --train_len 10000 --val_len 100 --hash_func 'neuralhash' --skip 1
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 1 --display 1 --hash_func 'photodna' --path /Users/neddamj/Documents/BU/Research/2022PhotoDNA/nnhash/inversion/saved_models/photodna_celeba_model.pth