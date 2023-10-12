import sys
sys.path.append('..')

# Helper imports
import utils

# Numerical computing and display imports
import matplotlib.pyplot as plt
import numpy as np

def nes_gradient_estimate(img, mean=0, std=0.1, sigma=0.5):
    def hamming_dist_loss(img, perterbed_img):
        # Compute the hamming distance 
        orig_hash = utils.compute_hash(img)
        perterbed_hash = utils.compute_hash(perterbed_img)
        return utils.distance(orig_hash, perterbed_hash, 'hamming')/96
    # Set random seed
    np.random.seed(42)
    noise_plus, noise_minus = np.random.normal(loc=mean-0.1, scale=std, size=img.shape), np.random.normal(loc=mean+0.1, scale=std, size=img.shape)
    img_plus = img + sigma*noise_plus*255
    img_minus = img - 255*sigma*noise_minus

    plus_loss, minus_loss = hamming_dist_loss(img, img_plus), hamming_dist_loss(img, img_minus)

    est_grads = np.zeros(img.shape)
    est_grads += plus_loss*noise_plus
    est_grads -= minus_loss*noise_minus

    return est_grads/(2*np.pi*sigma)

def nes_attack(img, mean=0, std=0.1, sigma=0.5, eps=0.05, l2_threshold=25, l2_tolerance=30):
    num_queries, counter = 0, 0
    # Estimate gradients with NES and find the grad direction
    est_grad = nes_gradient_estimate(img, mean=mean, std=std, sigma=sigma)
    num_queries += 4
    grad_direction = est_grad
    while True:
        num_queries += 2
        counter += 1
        # Update the image based on gradient direction 
        perturbed_img = img + 255*eps*grad_direction
        perturbed_img = np.clip(perturbed_img, 0, 255).astype(np.uint8)
        # Re-run the update if there isnt enough distortion in the image
        l2_distance = utils.distance(img, perturbed_img)
        print(f'eps: {eps} Lower Bound: {l2_threshold-l2_tolerance} Upper Bound: {l2_threshold} L2 Dist: {l2_distance}')
        # Termination condition
        if counter < 20:
            if l2_distance > l2_threshold:
                eps /= 3
            elif l2_distance < l2_threshold-l2_tolerance:
                eps *= 2
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