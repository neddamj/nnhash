from hash import compute_hash
import cv2

def generate_test_images(image_path, downsize_factor=90, upsize_factor=10):
    # Generate a larger and a smaller version of the same image
    img =  cv2.imread(image_path)
    (H, W) = img.shape[0], img.shape[1]
    img_smaller = cv2.resize(img, (int(downsize_factor/100*W), int(downsize_factor/100*H)), 
                        interpolation=cv2.INTER_AREA)
    img_larger = cv2.resize(img, (int((100+upsize_factor)/100*W), int((100+upsize_factor)/100*H)), 
                        interpolation=cv2.INTER_AREA)
    
    # Save those images to disk and return the paths to the images
    filename, filetype = image_path.split('.')
    path_smaller = f'{filename}_smaller.{filetype}'
    path_larger = f'{filename}_larger.{filetype}'
    cv2.imwrite(path_smaller, img_smaller)
    cv2.imwrite(path_larger, img_larger)

    return (path_smaller, path_larger)


def verify_hash(path, path_smaller, path_larger):
    # Verify that the same has is computed regardless of the image size
    hash = compute_hash(path)
    hash_s = compute_hash(path_smaller)
    hash_l = compute_hash(path_larger)

    print(f'Hash Regular: {hash}\nHash Smaller: {hash_s}\nHash Larger: {hash_l}')

    if (hash[1] == hash_s[1]) and (hash_s[1] == hash_l[1]):
        return True
    else:
        return False

if __name__ == "__main__":
    img_path = 'images/c1.bmp'
    img_path_smaller, img_path_larger = generate_test_images(img_path)  
    result = verify_hash(img_path, img_path_smaller, img_path_larger)
    print(result)