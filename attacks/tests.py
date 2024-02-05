from utils import compute_hash, load_img_paths, load_img

# Get all the image paths
img_paths = []
folder_paths = load_img_paths('../photos/')
for path in folder_paths:
    for img_path in load_img_paths(path):
        img_paths.append(img_path)

# Compute the image hashes
for i, path in enumerate(img_paths):
    if i%3 == 0:
        print('New IMG')
    print(f'Image: {path}\tHash Value:{hex(compute_hash(path))}')