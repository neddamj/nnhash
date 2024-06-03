import sys
sys.path.append('..')

# Helper imports
import utils
import os

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np

class ZOSignSGDttack:
    def __init__(self, max_queries, epsilon, l2_threshold, l2_tolerance, search_steps):
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.l2_threshold = l2_threshold
        #self.upper_tolerance = upper_tolerance
        #self.lower_tolerance = lower_tolerance
        self.l2_tolerance = l2_tolerance
        self.search_steps = search_steps

    def grad_estimate(self, img, hash_func):
        _grads = np.zeros_like(img).astype(np.float64)
        _shape = img.shape
        print('Estimating Gradients...')
        for _ in range(self.max_queries):
            exp_noise = np.random.randn(*_shape)
            pert_img = img + self.epsilon * (255 * exp_noise)
            hamm_dist = utils.distance(utils.compute_hash(pert_img, hash_func=hash_func), 
                                       utils.compute_hash(img, hash_func=hash_func), 
                                       'hamming',
                                       hash_func=hash_func)
            est_deriv = hamm_dist / self.epsilon
            _grads += est_deriv * exp_noise
        print('Gradient Estimation Complete...')
        return _grads, 2*self.max_queries

    def attack(self, img_path, hash_func):
        # Define the filepath
        path = img_path.split('/') 
        path[-1] = f'{img_path.split("/")[3].split(".")[0]}_zosignsgd.bmp'
        zosignsgd_filename = os.path.sep.join(path)
        # Initialize the image
        img = utils.load_img(img_path)
        grads, num_queries = self.grad_estimate(img, hash_func=hash_func)
        counter = 0
        while True:
            counter += 1
            print(f'Adding {self.epsilon} noise')
            perturbed_img = img - self.epsilon * np.sign(grads)
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            l2_distance = utils.distance(img, perturbed_img)
            print(f'Step: {counter} L2 Distance: {l2_distance}')
            if counter < self.search_steps:
                if l2_distance > self.l2_threshold:
                    self.epsilon /= 2
                elif l2_distance < self.l2_threshold + self.l2_tolerance: 
                    self.epsilon *= 1.5
                else:
                    # Save the image
                    utils.save_img(zosignsgd_filename, perturbed_img)
                    break
            else:
                # Save the image
                utils.save_img(zosignsgd_filename, perturbed_img)
                break
        return zosignsgd_filename, num_queries