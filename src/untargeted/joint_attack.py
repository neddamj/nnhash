import utils

class JointAttack:
    def __init__(self, sl_algo, hl_algo):
        self.sl_algo = sl_algo
        self.hl_algo = hl_algo

    def attack(self, img_path):
        # Soft-label attack
        (target_filename, sl_queries) = self.sl_algo.attack(img_path=img_path)
        # Hard-label attack
        (adv_img, hl_queries) = self.hl_algo.attack(orig_img_path=img_path, 
                                    target_img_path=target_filename)
        # Load the original image as well as the noisy adv. image outputted by simba
        orig_img, sl_img = utils.load_img(img_path), utils.load_img(target_filename)
        return orig_img, sl_img, adv_img, sl_queries, hl_queries