echo '[INFO] Beginning data loading...'
python data.py --dataset 'celeba' --train_len 10000 --val_len 1000 --skip 1
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 1 --display 1 --path /Users/neddamj/Documents/BU/Research/2022PhotoDNA/nnhash/inversion/saved_models/pdq_2024-04-03_11:15:35%_celeba_saved_model.pth
echo '[INFO] Inference complete...'