import sys
sys.path.append('..')

# Helper imports
import utils

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np

class NESAttack:
    def __init__(self,
                 mean, 
                 std, 
                 sigma, 
                 eps, 
                 l2_threshold, 
                 l2_tolerance, 
                 num_samples):
        self.mean = mean
        self.std = std
        self.sigma = sigma
        self.eps = eps
        self.l2_threshold = l2_threshold
        self.l2_tolerance = l2_tolerance
        self.num_samples = num_samples

    def nes_gradient_estimate(self, img, mean=0, std=0.1, sigma=0.5, num_samples=100):
        grads = []
        num_queries = 0
        print('Estimating gradients with NES...')
        for i in range(num_samples):
            noise = np.random.normal(mean, std, size=img.shape)
            new = img + sigma*noise*255
            # Find the hamming dist
            orig_hash = utils.compute_hash(img)
            perterbed_hash = utils.compute_hash(new)
            g = noise*utils.distance(orig_hash, perterbed_hash, 'hamming')
            grads.append(g)
            num_queries += 2
        est_grad = np.mean(np.array(grads), axis=0)
        return est_grad/(2*np.pi*sigma), num_queries

    def attack(self, img_path):
        # Initialize the image
        filename, filetype = img_path.split('.')
        img = utils.load_img(img_path)
        num_queries, counter = 0, 0
        # Estimate gradients with NES and find the grad direction
        est_grad, num_queries = self.nes_gradient_estimate(img, mean=self.mean, std=self.std, sigma=self.sigma, num_samples=self.num_samples)
        grad_direction = np.sign(est_grad)
        while True:
            num_queries += 2
            counter += 1
            # Update the image based on gradient direction 
            perturbed_img = img - 255*self.eps*grad_direction
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            # Re-run the update if there isnt enough distortion in the image
            l2_distance = utils.distance(img, perturbed_img)
            print(f'eps: {self.eps} Lower Bound: {self.l2_threshold-self.l2_tolerance} Upper Bound: {self.l2_threshold} L2 Dist: {l2_distance}')
            # Termination condition
            nes_filename = f'{filename}_nes.bmp'
            if counter < 20:
                if l2_distance > self.l2_threshold:
                    self.eps /= 2
                elif l2_distance < self.l2_threshold-self.l2_tolerance:
                    self.eps *= 1.5
                else:
                    print("BREAK")
                    utils.save_img(nes_filename, perturbed_img)
                    return nes_filename, num_queries
            else:
                utils.save_img(nes_filename, perturbed_img)
                return nes_filename, num_queries
    

if __name__ == "__main__":
    idx = 1
    img_path = f'../../images/{idx}.bmp' 
    nes_mean = 0
    nes_std = 0.1
    nes_sigma = 0.7
    nes_eps = 0.005
    nes = NESAttack(mean=nes_mean, std=nes_std, sigma=nes_sigma, eps=nes_eps)
    perturbed_img_path, nes_queries = nes.attack(img_path)
    perturbed_img = utils.load_img(perturbed_img_path)
    plt.imshow(perturbed_img)