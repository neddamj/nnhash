import sys
sys.path.append('..')

import utils
import numpy as np

class NESAttack:
    def __init__(self,
                 mean, 
                 std, 
                 sigma, 
                 eps, 
                 hamming_threshold, 
                 num_samples):
        self.mean = mean
        self.std = std
        self.sigma = sigma
        self.eps = eps
        self.hamming_threshold = hamming_threshold
        self.num_samples = num_samples

    def nes_gradient_estimate(self, img, target_img):
        grads = []
        num_queries = 0
        print('Estimating gradients with NES...')
        for i in range(self.num_samples):
            noise = np.random.normal(self.mean, self.std, size=img.shape)
            new = img + self.sigma*noise*255
            # Find the hamming dist
            target_hash = utils.compute_hash(target_img)
            perterbed_hash = utils.compute_hash(new)
            g = noise*utils.distance(target_hash, perterbed_hash, 'hamming')
            grads.append(g)
            num_queries += 2
        est_grad = np.mean(np.array(grads), axis=0)
        return est_grad/(2*np.pi*self.sigma), num_queries

    def attack(self, img_path, target_path):
        # Initialize the image
        filename, filetype = img_path.split('.')
        img = utils.load_img(img_path)
        target_img = utils.load_img(target_path)
        num_queries, counter = 0, 0
        # Estimate gradients with NES and find the grad direction
        est_grad, num_queries = self.nes_gradient_estimate(img, target_path)
        grad_direction = np.sign(est_grad)
        while True:
            counter += 1
            # Update the image based on gradient direction 
            perturbed_img = img - 255*self.eps*grad_direction
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            # Re-run the update if there isnt enough distortion in the image
            l2_distance = utils.distance(img, perturbed_img)
            target_hash, perturbed_hash = utils.compute_hash(target_path), utils.compute_hash(perturbed_img)
            hamming_dist = utils.distance(target_hash, perturbed_hash, 'hamming')
            print(f'eps: {self.eps} Hamming Dist to Target: {hamming_dist} L2 Dist to Target: {l2_distance}')
            # Termination condition
            nes_filename = f'{filename}_nes.bmp'
            if counter < 20:
                if hamming_dist > self.hamming_threshold:
                    self.eps *= 1.5
                else:
                    utils.save_img(nes_filename, perturbed_img)
                    break
            else:
                utils.save_img(nes_filename, perturbed_img)
                break
        return nes_filename, (num_queries + counter * 2)