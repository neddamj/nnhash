import sys
sys.path.append('..')

# Helper imports
import utils

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np

def nes_gradient_estimate(img, mean=0, std=0.1, sigma=0.5, num_samples=100):
    def hamming_dist_loss(img, perterbed_img):
        # Compute the hamming distance 
        orig_hash = utils.compute_hash(img)
        perterbed_hash = utils.compute_hash(perterbed_img)
        return utils.distance(orig_hash, perterbed_hash, 'hamming')
    grads = []
    num_queries = 0
    print('Estimating gradients with NES...')
    for i in range(num_samples):
        noise = np.random.normal(mean, std, size=img.shape)
        new = img + sigma*noise*255
        g = noise*hamming_dist_loss(img, new)
        grads.append(g)
        num_queries += 2
    est_grad = np.mean(np.array(grads), axis=0)
    return est_grad/(2*np.pi*sigma), num_queries

def nes_attack(img, mean=0, std=0.1, sigma=0.5, eps=0.05, l2_threshold=25, l2_tolerance=30, num_samples=50):
    num_queries, counter = 0, 0
    # Estimate gradients with NES and find the grad direction
    est_grad, num_queries = nes_gradient_estimate(img, mean=mean, std=std, sigma=sigma, num_samples=num_samples)
    grad_direction = est_grad
    while True:
        num_queries += 2
        counter += 1
        # Update the image based on gradient direction 
        perturbed_img = img - 255*eps*grad_direction
        perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
        # Re-run the update if there isnt enough distortion in the image
        l2_distance = utils.distance(img, perturbed_img)
        print(f'eps: {eps} Lower Bound: {l2_threshold-l2_tolerance} Upper Bound: {l2_threshold} L2 Dist: {l2_distance}')
        # Termination condition
        if counter < 20:
            if l2_distance > l2_threshold:
                eps /= 2
            elif l2_distance < l2_threshold-l2_tolerance:
                eps *= 1.5
            else:
                print("BREAK")
                return perturbed_img, num_queries
        else:
            eps = 0.01
            perturbed_img = img + 255*eps*grad_direction
            perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
            return perturbed_img, num_queries
    

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