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

def bin_boundary_search(orig_img, adv_img, l2_threshold, hamming_threshold, max_iters=15):
    print('[INFO] Starting Boundary Search...')
    iter = 0
    src_img = copy.deepcopy(orig_img)       # Copy of origin img that will be used for interpolation
    l2_dist = utils.distance(orig_img, adv_img, 'l2')
    while l2_dist >= l2_threshold:
        if iter == max_iters:
            break
        midpoint = (src_img + adv_img)/2
        if decision_fn(orig_img, midpoint, hamming_threshold):
            adv_img = midpoint
        else:
            src_img = midpoint
        print(f'L2 Dist: {l2_dist}')
        l2_dist = utils.distance(orig_img, adv_img, 'l2')
        iter += 1
    print('[INFO] Boundary Search Complete...')
    return adv_img

def estimate_grad_direction(img, sample_count):
    # Create the noise unit vectors
    noise = np.random.randn(sample_count, img.shape[0], img.shape[1], img.shape[2])
    noise /= np.linalg.norm(noise)
    # Get the signs of each img with the additive noise
    directions = np.zeros((noise.shape[0], 1))
    for i in range(noise.shape[0]):
        new = img/255 + noise[i]
        new = np.asarray(255*new).astype(np.uint8)
        np.clip(new, 0, 255)
        directions[i] = 1 if decision_fn(img, new, threshold=2) else -1
    # Compute the gradient direction estimate
    direction_estimate = sum([noise[i]*directions[i] for i in range(noise.shape[0])])
    return direction_estimate

def grad_based_update(orig_img, adv_img, grad_direction, stepsize, hamming_threshold):
    while True:
        new_img = adv_img + stepsize*grad_direction
        if decision_fn(orig_img, new_img, hamming_threshold):
            break
        stepsize /= 2
    return new_img

def hop_skip_jump_attack(orig_img_path, 
         target_img_path,
         max_iters=10, 
         grad_queries=100, 
         l2_threshold=30, 
         hamming_threshold=5):
    orig_img, target_img = utils.load_img(orig_img_path), utils.load_img(target_img_path)
    steps = 0
    for idx in range(1, max_iters+1):
        steps = idx+1
        print(f'HSJA Iteration: {idx}')
        # Find the img as close to the boundary as possible
        boundary_img = bin_boundary_search(orig_img, target_img, l2_threshold, hamming_threshold)
        l2_dist = utils.distance(orig_img, boundary_img)
        if l2_dist < l2_threshold:
            print('[INFO] Stepsize too small...')
            return (boundary_img, steps)
        # Estimate the gradient direction
        sample_count = int(grad_queries * (idx)**0.5)
        grad_direction = estimate_grad_direction(boundary_img, sample_count)
        # Calculate the stepsize
        stepsize = utils.distance(orig_img, boundary_img, 'l2')/np.sqrt(idx)
        target_img = grad_based_update(orig_img, boundary_img, grad_direction, stepsize, hamming_threshold)
    return (target_img, steps)

# %% 
if __name__ == '__main__':
    orig_img_path, target_img_path = '/Volumes/TempRAM/1.jpeg','/Volumes/TempRAM/1_new.jpeg'
    adv_img, steps = hop_skip_jump_attack(orig_img_path=orig_img_path, target_img_path=target_img_path, max_iters=2, grad_queries=10, l2_threshold=30, hamming_threshold=10)
    im = Image.fromarray(adv_img.astype(np.uint8))
    im.show()
    #print(steps)
    orig_img = utils.load_img(orig_img_path)
    hash1, hash2 = utils.compute_hash(orig_img), utils.compute_hash(adv_img)
    print(f'Hamming Dist: {utils.distance(hash1, hash2, "hamming")}')
    print(f"l2 Dist: {utils.distance(orig_img, adv_img, 'l2')}")




