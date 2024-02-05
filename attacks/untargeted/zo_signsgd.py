import sys
sys.path.append('..')

# Helper imports
import utils

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

    def grad_estimate(self, img):
        _grads = np.zeros_like(img).astype(np.float64)
        _shape = img.shape
        print('Estimating Gradients...')
        for _ in range(self.max_queries):
            exp_noise = np.random.randn(*_shape)
            pert_img = img + self.epsilon * (255 * exp_noise)
            hamm_dist = utils.distance(utils.compute_hash(pert_img), 
                                       utils.compute_hash(img), 
                                       'hamming')
            est_deriv = hamm_dist / self.epsilon
            _grads += est_deriv * exp_noise
        print('Gradient Estimation Complete...')
        return _grads, 2*self.max_queries

    def attack(self, img_path):
        # Initialize the image
        filename, filetype = img_path.split('.')
        zosignsgd_filename = f'{filename}_zosignsgd.bmp'
        img = utils.load_img(img_path)
        grads, num_queries = self.grad_estimate(img)
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


if __name__ == "__main__":
    idx = 0
    img_path = f'../../images/{idx+1}.bmp' 
    _, _, _, _, path, filetype = img_path.split('.')
    img_path = path.split('/')
    img_path = f'/Volumes/TempRAM/{img_path[2]}.{filetype}'
    max_queries = 100
    epsilon = 0.2
    hamming_threshold = int(0.1 * 127)
    zo_signsgd = ZOSignSGDttack(max_queries=max_queries, epsilon=epsilon, hamming_threshold=hamming_threshold)
    perturbed_img_path, zo_queries = zo_signsgd.attack(img_path)
    plt.imshow(utils.load_img(perturbed_img_path))
    plt.show()