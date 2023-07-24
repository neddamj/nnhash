# %%
import sys
sys.path.append('..')

from data import CIFAR10, IMAGENETTE
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import utils
import copy

def decision_fn(orig_img, new_img, threshold=40):
    # Save the images and get their hashes
    orig_hash, new_hash = utils.compute_hash(orig_img), utils.compute_hash(new_img)
    # Make the decision based on the threshold
    hamming_dist = utils.distance(orig_hash, new_hash, 'hamming')
    print(f'Hamming Dist: {hamming_dist}')
    return hamming_dist >= threshold

def bin_boundary_search(orig_img, adv_img, l2_threshold, hamming_threshold, max_queries=20):
    print('[INFO] Starting Boundary Search...')
    num_queries = 0
    src_img = copy.deepcopy(orig_img)       # Copy of origin img that will be used for interpolation
    l2_dist = utils.distance(orig_img, adv_img, 'l2')
    while (l2_dist >= l2_threshold) and (num_queries < max_queries):
        num_queries += 1
        midpoint = (src_img + adv_img)/2
        if decision_fn(orig_img, midpoint, hamming_threshold):
            adv_img = midpoint
        else:
            src_img = midpoint
        print(f'L2 Dist: {l2_dist}')
        l2_dist = utils.distance(orig_img, adv_img, 'l2')
    print('[INFO] Boundary Search Complete...')
    return adv_img, num_queries

def estimate_grad_direction(img, sample_count, hamming_threshold):
    num_queries = 0
    # Create the noise unit vectors
    noise = np.random.randn(sample_count, img.shape[0], img.shape[1], img.shape[2])
    noise /= np.linalg.norm(noise)
    # Get the signs of each img with the additive noise
    directions = np.zeros((noise.shape[0], 1))
    print(f'[INFO] Estimating gradient direction...')
    for i in range(noise.shape[0]):
        print(f'Grad Direction Estimate # {i+1}/{noise.shape[0]}: ', end="")
        num_queries += 1
        new = img/255 + noise[i]
        new = np.asarray(255*new).astype(np.uint8)
        np.clip(new, 0, 255)
        directions[i] = -1 if decision_fn(img, new, threshold=hamming_threshold) else 1
    print(f'[INFO] Gradient direction estimation complete...')
    # Compute the gradient direction estimate
    direction_estimate = sum([noise[i]*directions[i] for i in range(noise.shape[0])])
    return direction_estimate, num_queries

def grad_based_update(orig_img, adv_img, grad_direction, stepsize, hamming_threshold):
    num_queries = 0
    print('[INFO] Starting gradient based update...')
    while (num_queries < 50):
        print(f'Gradient Update # {num_queries}: ', end="")
        num_queries += 1
        new_img = adv_img + stepsize*grad_direction
        if decision_fn(orig_img, new_img, hamming_threshold):
            break
        stepsize /= 2
    print('[INFO] Gradient based update complete...')
    return new_img, num_queries

def hop_skip_jump_attack(orig_img_path, 
         target_img_path,
         max_iters=10, 
         grad_queries=20, 
         l2_threshold=24, 
         hamming_threshold=10):
    orig_img, target_img = utils.load_img(orig_img_path), utils.load_img(target_img_path)
    num_queries = 0
    for idx in range(1, max_iters+1):
        print(f'HSJA Iteration: {idx}')
        # Find the img as close to the boundary as possible
        boundary_img, search_queries = bin_boundary_search(orig_img, target_img, l2_threshold, hamming_threshold)
        l2_dist = utils.distance(orig_img, boundary_img)
        if l2_dist < l2_threshold:
            print('[INFO] Stepsize too small...')
            return (boundary_img, num_queries+search_queries)
        # Estimate the gradient direction
        sample_count = min(int(grad_queries * np.sqrt(idx)), 100)
        grad_direction, grad_queries = estimate_grad_direction(boundary_img, sample_count, hamming_threshold)
        # Calculate the stepsize
        stepsize = utils.distance(orig_img, boundary_img, 'l2')/np.sqrt(idx)
        target_img, update_queries = grad_based_update(orig_img, boundary_img, grad_direction, stepsize, hamming_threshold)
        num_queries += (search_queries + grad_queries + update_queries)
    return (target_img, num_queries)

# %% 
if __name__ == '__main__':
    orig_img_path, target_img_path = '/Volumes/TempRAM/1.jpeg','/Volumes/TempRAM/1_new.jpeg'
    adv_img, steps = hop_skip_jump_attack(orig_img_path=orig_img_path, 
                                          target_img_path=target_img_path, 
                                          max_iters=2, 
                                          grad_queries=10, 
                                          l2_threshold=30, 
                                          hamming_threshold=10)
    im = Image.fromarray(adv_img.astype(np.uint8))
    im.show()
    #print(steps)
    orig_img = utils.load_img(orig_img_path)
    hash1, hash2 = utils.compute_hash(orig_img), utils.compute_hash(adv_img)
    print(f'Hamming Dist: {utils.distance(hash1, hash2, "hamming")}')
    print(f"l2 Dist: {utils.distance(orig_img, adv_img, 'l2')}")

