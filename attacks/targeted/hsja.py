 # %%
import sys
sys.path.append('..')

from data import CIFAR10, IMAGENETTE
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import utils
import copy

class HSJAttack:
    def __init__(self,
                max_iters, 
                grad_queries, 
                l2_threshold, 
                hamming_threshold):
        self.max_iters = max_iters
        self.grad_queries = grad_queries
        self.l2_threshold = l2_threshold
        self.hamming_threshold = hamming_threshold

    def decision_fn(self, orig_img, new_img, threshold=40):
        # Save the images and get their hashes
        orig_hash, new_hash = utils.compute_hash(orig_img), utils.compute_hash(new_img)
        # Make the decision based on the threshold
        hamming_dist = utils.distance(orig_hash, new_hash, 'hamming')
        print(f'Hamming Dist: {hamming_dist}')
        return (hamming_dist <= threshold)

    def bin_boundary_search(self, orig_img, adv_img, l2_threshold, hamming_threshold, max_queries=20):
        print('[INFO] Starting Boundary Search...')
        num_queries = 0
        src_img = copy.deepcopy(orig_img)       # Copy of origin img that will be used for interpolation
        l2_dist = utils.distance(orig_img, adv_img, 'l2')
        while (l2_dist >= l2_threshold) and (num_queries < max_queries):
            num_queries += 1
            midpoint = (src_img + adv_img)/2
            if self.decision_fn(orig_img, midpoint, hamming_threshold):
                adv_img = midpoint
            else:
                src_img = midpoint
            print(f'L2 Dist: {l2_dist}')
            l2_dist = utils.distance(orig_img, adv_img, 'l2')
        print('[INFO] Boundary Search Complete...')
        return adv_img, num_queries

    def estimate_grad(self, img, sample_count, hamming_threshold):
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
            directions[i] = -1 if self.decision_fn(img, new, threshold=hamming_threshold) else 1
        print(f'[INFO] Gradient direction estimation complete...')
        # Compute the gradient direction estimate
        direction_estimate = sum([noise[i]*directions[i] for i in range(noise.shape[0])])
        return direction_estimate, num_queries

    def grad_based_update(self, orig_img, adv_img, grad, stepsize, hamming_threshold):
        num_queries = 0
        print('[INFO] Starting gradient based update...')
        while (num_queries < 50):
            print(f'Gradient Update # {num_queries}: ', end="")
            num_queries += 1
            new_img = adv_img + stepsize*grad
            if self.decision_fn(orig_img, new_img, hamming_threshold):
                break
            stepsize /= 2
        print('[INFO] Gradient based update complete...')
        return new_img, num_queries

    def attack(self, orig_img_path, target_img_path):
        orig_img, target_img = utils.load_img(orig_img_path), utils.load_img(target_img_path)
        num_queries = 0
        for idx in range(1, self.max_iters+1):
            print(f'HSJA Iteration: {idx}')
            # Find the img as close to the boundary as possible
            boundary_img, search_queries = self.bin_boundary_search(orig_img, target_img, self.l2_threshold, self.hamming_threshold)
            boundary_hash, target_hash = utils.compute_hash(boundary_img),utils.compute_hash(target_img_path)
            hamming_dist = utils.distance(boundary_hash, target_hash, 'hamming')
            if hamming_dist < self.hamming_threshold:
                print('[INFO] Stepsize too small...')
                return (boundary_img, num_queries+search_queries+2)
            # Estimate the gradient direction
            sample_count = min(int(self.grad_queries * np.sqrt(idx)), 50)
            grad, grad_est_queries = self.estimate_grad(boundary_img, sample_count, self.hamming_threshold)
            # Calculate the stepsize
            stepsize = utils.distance(orig_img, boundary_img, 'l2')/np.sqrt(idx)
            target_img, update_queries = self.grad_based_update(orig_img, boundary_img, grad, stepsize, self.hamming_threshold)
            num_queries += (search_queries + grad_est_queries + update_queries + 2)
        return (target_img, num_queries)

# %% 
if __name__ == '__main__':
    orig_img_path, target_img_path = '/Volumes/TempRAM/1.jpeg','/Volumes/TempRAM/2.jpeg'
    hsja = HSJAttack(max_iters=2, grad_queries=10, l2_threshold=30, hamming_threshold=40)
    adv_img, steps = hsja.attack(orig_img_path=orig_img_path, target_img_path=target_img_path)
    im = Image.fromarray(adv_img.astype(np.uint8))
    im.show()

    """orig_img, target_img = utils.load_img(orig_img_path), utils.load_img(target_img_path)
    hash1, hash2 = utils.compute_hash(target_img), utils.compute_hash(adv_img)
    print(f'Hamming Dist: {utils.distance(hash1, hash2, "hamming")}')
    print(f"l2 Dist: {utils.distance(target_img, adv_img, 'l2')}")"""