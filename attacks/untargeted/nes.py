import sys
sys.path.append('..')

# Helper imports
import utils

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np
import os

class NESAttack:
    def __init__(self,
                 mean, 
                 std, 
                 sigma, 
                 eps, 
                 l2_threshold, 
                 l2_tolerance, 
                 num_samples, 
                 p):
        self.mean = mean
        self.std = std
        self.sigma = sigma
        self.eps = eps
        self.l2_threshold = l2_threshold
        self.l2_tolerance = l2_tolerance
        self.num_samples = num_samples
        self.p = p

    def nes_gradient_estimate(self, img, mean=0, std=0.1, sigma=0.5, num_samples=100, hash_func='neuralhash'):
        grads = []
        num_queries = 0
        print('Estimating gradients with NES...')
        for i in range(num_samples):
            noise = np.random.normal(mean, std, size=img.shape)
            new = img + sigma*noise*255
            # Find the hamming dist
            orig_hash = utils.perturb_hash(utils.compute_hash(img, hash_func=hash_func), p=self.p, hash_func=hash_func)
            perterbed_hash = utils.perturb_hash(utils.compute_hash(new, hash_func=hash_func), p=self.p, hash_func=hash_func)
            g = noise*utils.distance(orig_hash, perterbed_hash, 'hamming', hash_func=hash_func)
            grads.append(g)
            num_queries += 2
        est_grad = np.mean(np.array(grads), axis=0)
        return est_grad/(2*np.pi*sigma), num_queries

    def attack(self, img_path, hash_func):
        # Define the filepath
        path = img_path.split('/') 
        path[-1] = f'{img_path.split("/")[3].split(".")[0]}_nes.bmp'
        nes_filename = os.path.sep.join(path)
        # Initialize the image
        img = utils.load_img(img_path)
        num_queries, counter = 0, 0
        # Estimate gradients with NES and find the grad direction
        est_grad, num_queries = self.nes_gradient_estimate(img, mean=self.mean, std=self.std, sigma=self.sigma, num_samples=self.num_samples, hash_func=hash_func)
        grad_direction = np.sign(est_grad)
        while True:
            counter += 1
            # Update the image based on gradient direction 
            perturbed_img = img - 255*self.eps*grad_direction
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            # Re-run the update if there isnt enough distortion in the image
            l2_distance = utils.distance(img, perturbed_img)
            print(f'eps: {self.eps} Lower Bound: {self.l2_threshold-self.l2_tolerance} Upper Bound: {self.l2_threshold} L2 Dist: {l2_distance}')
            # Termination condition
            if counter < 20:
                if l2_distance > self.l2_threshold:
                    self.eps /= 2
                elif l2_distance < self.l2_threshold-self.l2_tolerance: 
                    self.eps *= 1.5
                else:
                    print("BREAK")
                    utils.save_img(nes_filename, perturbed_img)
                    break
            else:
                utils.save_img(nes_filename, perturbed_img)
                break
        return nes_filename, num_queries