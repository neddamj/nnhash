# Assessing the Adversarial Security of Practical Perceptual Hashing Algorithms

This repo contais the official implementation for the paper [Assessing the Adversarial Security of Practical Perceptual Hashing Algorithms](https://arxiv.org/pdf/2406.00918).

## Setup

Refer to [this repository](https://github.com/KhaosT/nhcalc) to get the neural hash executable, and [this one](https://github.com/jankais3r/pyPhotoDNA) to get the PhotoDNA executable. To setup PDQ and install the rest of the dependencies, run the following command after cloning this repo.
``` 
pip install -r requirements.txt
```

## 1. Run the Adversarial Attacks
The implementation of the adversarial attacks is found in `attacks/` folder. To run the image editing attacks, run the `attacks/classical/run_classical_attack.sh` file. To run the untargeted adversarial attacks, run the `attacks/untargeted/attack.ipynb` file and to run the targeted adversarial attack, run the `attacks/targeted/attack.ipynb` file. Each of those runs will log their results to a csv file in the `metrics` folder in the same directory as the file that was ran. To analyze the results, paste the path to the csv file into the `analyze_results.ipynb` file and run it.

## 2. Run the Hash Inversion Attacks
There is a [bug](https://github.com/pytorch/vision/issues/1920) in the torchvision celeba dataloader. To work around the bug, follow the instructions [here](https://github.com/pytorch/vision/issues/1920#issuecomment-852237902).

To train the inversion models, navigate to `inversion/` then run:

``` bash
./run_training.sh
```

To generate images from their hash values, navigate to `inversion/` then run:

``` bash
./run_inference.sh
```

You can specify the model hyperparameters and general script parameters for training and generation in the script files above. Pretrained models for hash inversion can be found in `inversion/saved_models`.
