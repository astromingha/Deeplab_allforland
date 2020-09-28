import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr

class LandcoverSegmentation(data.Dataset):
    def __init__(self, args, split="train"):

        self.root = args.dataset_path
        self.split = split
        self.args = args
        self.files = {}

        if self.args.dataset_cat == 'detail':
            self.images_base = os.path.join(self.root,self.split, 'image', 'image')
            if self.split == "train":
                self.annotations_base = os.path.join(self.root, self.split, 'mask_detail', 'mask')
            else:
                self.annotations_base = os.path.join(self.root, self.split, 'mask', 'mask')

            self.NUM_CLASSES = 41 + 2
            self.void_classes = [42, 43]
            self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
            self.class_names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16', '17','18','19',
                                '20','21','22', '23','24','25','26','27','28','29','30','31','32','33', '34', '35','36',
                                '37','38','39','40','41','42','43']

        elif self.args.dataset_cat == 'middle':
            self.images_base = os.path.join(self.root, self.split, 'image', 'image')
            self.annotations_base = os.path.join(self.root, self.split, 'mask_middle', 'mask')

            self.NUM_CLASSES = 22 + 2
            self.void_classes = [23, 24]
            self.valid_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            self.class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                                '17', '18', '19', '20', '21', '22', '23', '24']

        elif self.args.dataset_cat == 'main':
            self.images_base = os.path.join(self.root, self.split, 'image', 'image')
            self.annotations_base = os.path.join(self.root, self.split, 'mask_main', 'mask')

            self.NUM_CLASSES = 7 + 2
            self.void_classes = [8, 9]
            self.valid_classes = [1, 2, 3, 4, 5, 6, 7]
            self.class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            raise NotImplementedError("args.dataset_cat is required.")

        self.files[split] = self.recursive_glob(rootdir=self.images_base)

        self.mean = (0.27, 0.306, 0.294)
        self.std = (0.227, 0.211, 0.198)
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                os.path.basename(img_path)[:-4] + '.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp += 1
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.mean, std= self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = LandcoverSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=10, shuffle=True, num_workers=1)

    nimages = 0
    mean = 0.0
    var = 0.0
    for i_batch, batch_target in enumerate(dataloader):
        batch = batch_target['image']
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print(mean)
    print(std)