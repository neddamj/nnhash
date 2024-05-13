import utils

class JointAttack:
    def __init__(self, soft_label, hard_label):
        self.soft_label = soft_label
        self.hard_label = hard_label

    def attack(self, img_path, hash_func='neuralhash'):
        # Soft-label attack
        (target_filename, sl_queries) = self.soft_label.attack(img_path=img_path,
                                                            hash_func=hash_func)
        # Hard-label attack
        (adv_img, hl_queries) = self.hard_label.attack(orig_img_path=img_path, 
                                    target_img_path=target_filename,
                                    hash_func=hash_func)
        # Load the original image as well as the noisy adv. image outputted by simba
        orig_img, sl_img = utils.load_img(img_path), utils.load_img(target_filename)
        return orig_img, sl_img, adv_img, sl_queries, hl_queries