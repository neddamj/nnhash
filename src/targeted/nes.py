import sys
sys.path.append('..')

# Helper imports
import utils

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np

def nes_gradient_estimate(img, target_img, mean=0, std=0.1, sigma=0.5, num_samples=100):
    def direction_loss(img, perterbed_img):
        # Compute the hamming distance 
        orig_hash = utils.compute_hash(img)
        perterbed_hash = utils.compute_hash(perterbed_img)
        return np.sign(orig_hash-perterbed_hash)
    grads = []
    num_queries = 0
    print('Estimating gradients with NES...')
    for i in range(num_samples):
        noise = np.random.normal(mean, std, size=img.shape)
        new = img + sigma*noise*255
        g = noise*direction_loss(target_img, new)
        grads.append(g)
        num_queries += 2
    est_grad = np.mean(np.array(grads), axis=0)
    print('Gradient estimation complete...')
    return est_grad/(2*np.pi*sigma), num_queries

def nes_attack(orig_img, target_img, target_hamming_dist=39, nes_mean=0.0, nes_std=0.1, nes_sigma=0.5, nes_num_samples=100):
    # Estimate gradients
    est_grad, num_queries = nes_gradient_estimate(orig_img, 
                                                  target_img, 
                                                  mean=nes_mean, 
                                                  std=nes_std, 
                                                  sigma=nes_sigma,
                                                  num_samples=nes_num_samples)
    total_queries = num_queries
    hamming_dist = utils.distance(utils.compute_hash(orig_img), utils.compute_hash(target_img), "hamming")
    LR = 1
    img = orig_img
    while True:
        if hamming_dist <= target_hamming_dist:
            print(f'FINAL HAMMING DISTANCE: {hamming_dist}')
            return img, total_queries
        # Hamming distance when updated in positive direction
        plus_img = np.clip((img +  LR*np.sign(est_grad)), 0, 255).astype(np.uint8)
        plus_hamming_dist = utils.distance(utils.compute_hash(plus_img), utils.compute_hash(target_img), "hamming")
        # Hamming distance when updated in negative direction
        minus_img = np.clip((img -  LR*np.sign(est_grad)), 0, 255).astype(np.uint8)
        minus_hamming_dist = utils.distance(utils.compute_hash(minus_img), utils.compute_hash(target_img), "hamming")

        if ((hamming_dist - plus_hamming_dist) > (hamming_dist - minus_hamming_dist)) and (hamming_dist - plus_hamming_dist) > 0:
            # Plus hamming distance smaller than minus
            print(f'Updating in + direction. Hamming Dist: {hamming_dist}')
            img = plus_img
            hamming_dist = plus_hamming_dist
            LR = 1
            est_grad, num_queries = nes_gradient_estimate(img, 
                                                        target_img,
                                                        mean=nes_mean, 
                                                        std=nes_std, 
                                                        sigma=nes_sigma, 
                                                        num_samples=nes_num_samples)
            # Track the number of queries made to hashing function
            total_queries += num_queries
        elif ((hamming_dist - plus_hamming_dist) < (hamming_dist - minus_hamming_dist)) and (hamming_dist - minus_hamming_dist) > 0:
            # Plus hamming distance smaller than minus
            print(f'Updating in - direction. Hamming Dist: {hamming_dist}')
            img = minus_img
            hamming_dist = minus_hamming_dist
            LR = 1
            est_grad, num_queries = nes_gradient_estimate(img, 
                                                        target_img,
                                                        mean=nes_mean, 
                                                        std=nes_std, 
                                                        sigma=nes_sigma, 
                                                        num_samples=nes_num_samples)
            # Track the number of queries made to hashing function
            total_queries += num_queries
        else:
            LR += 0.5
            print(f'Hamming Dist: {hamming_dist}\t New LR: {LR}')
            if LR >= 150:
                # Recalculate gradients for the perturbed image
                LR = 1
                print('ESTIMATING NEW GRADIENTS')
                est_grad, num_queries = nes_gradient_estimate(img, 
                                                              target_img,
                                                              mean=nes_mean, 
                                                              std=nes_std, 
                                                              sigma=nes_sigma, 
                                                              num_samples=nes_num_samples)
                # Track the number of queries made to hashing function
                total_queries += num_queries
    

if __name__ == "__main__":
    idx = 1
    img_path = f'../../images/{idx}.jpeg' 
    img = utils.load_img(img_path)

    nes_mean = 0
    nes_std = 0.1
    nes_sigma = 0.7
    nes_eps = 0.005
    perturbed_img, nes_queries = nes_attack(img, mean=nes_mean, std=nes_std, sigma=nes_sigma, eps=nes_eps)

    plt.imshow(perturbed_img)