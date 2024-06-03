import sys
sys.path.append('..')

import os
import utils
import numpy as np

class ZOSignSGDttack:
    def __init__(self, 
                 max_queries, 
                 epsilon, 
                 hamming_threshold, 
                 search_steps):
        self.max_queries = max_queries
        self.epsilon = epsilon
        self.hamming_threshold = hamming_threshold
        self.search_steps = search_steps

    def grad_estimate(self, img, target_img):
        _grads = np.zeros_like(img).astype(np.float64)
        _shape = img.shape
        print('Estimating Gradients...')
        for _ in range(self.max_queries):
            exp_noise = np.random.randn(*_shape)
            pert_img = img + self.epsilon * (255 * exp_noise)
            hamm_dist = utils.distance(utils.compute_hash(pert_img), 
                                       utils.compute_hash(target_img), 
                                       'hamming')
            est_deriv = hamm_dist / self.epsilon
            _grads += est_deriv * exp_noise
        print('Gradient Estimation Complete...')
        return _grads, 2*self.max_queries

    def attack(self, img_path, target_path):
        # Define the filepath
        path = img_path.split('/') 
        path[-1] = f'{img_path.split("/")[3].split(".")[0]}_simba.bmp'
        zosignsgd_filename = os.path.sep.join(path)
        img = utils.load_img(img_path)
        grads, num_queries = self.grad_estimate(img, target_path)
        counter = 0
        while True:
            counter += 1
            print(f'Adding {self.epsilon} noise')
            perturbed_img = img - self.epsilon * np.sign(grads)
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            l2_distance = utils.distance(img, perturbed_img)
            target_hash, perturbed_hash = utils.compute_hash(target_path), utils.compute_hash(perturbed_img)
            hamming_dist = utils.distance(target_hash, perturbed_hash, 'hamming')
            print(f'Step: {counter} L2 Distance: {l2_distance}')
            if counter < self.search_steps:
                if hamming_dist > self.hamming_threshold:
                    self.epsilon *= 1.5
                else:
                    utils.save_img(zosignsgd_filename, perturbed_img)
                    break
            else:
                utils.save_img(zosignsgd_filename, perturbed_img)
                break
        return zosignsgd_filename, num_queries