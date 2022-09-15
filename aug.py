import random
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)

class TwoCropsTransform:
    """Take two random crops of one image"""
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x1, x2):
        im1 = self.base_transform1(Image.fromarray(x1))
        im2 = self.base_transform2(Image.fromarray(x2))
        return im1, im2


def augmentation(aug_type):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(32, scale=(0.9, 1.)),
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(32, scale=(0.9, 1.)),
    ]

    augmentation3 = [
    ]

    if aug_type == "train":
    	aug_trans = TwoCropsTransform(transforms.Compose(augmentation1),
                         transforms.Compose(augmentation2))
    elif aug_type == "test":
        aug_trans = TwoCropsTransform(transforms.Compose(augmentation3),
                         transforms.Compose(augmentation3))
    elif aug_type == "single":
        aug_trans = transforms.Compose(augmentation3)
    else:
        print("There is something wrong with "+str(aug_type))
    return aug_trans

