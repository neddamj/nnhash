echo '[INFO] Beginning data loading...'
python data.py --dataset 'mnist' --train_len 10000 --val_len 100 --skip 1
echo '[INFO] Data loading complete, running inference now...'
python generate.py --rgb 0 --display 0 --path /Users/neddamj/Documents/BU/Research/2022PhotoDNA/nnhash/inversion/saved_models/2024-03-01_12:41:11%_mnist_saved_model.pth
echo '[INFO] Inference complete...'