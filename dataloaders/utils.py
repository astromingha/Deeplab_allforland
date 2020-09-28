import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_seg_map_sequence(label_masks, dataset='detail'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'detail':
        n_classes = 43
        label_colours = get_detail_labels()
    elif dataset == 'middle':
        n_classes = 24
        label_colours = get_mid_labels()
    elif dataset == 'main':
        n_classes = 9
        label_colours = get_main_labels()
    else:
        raise NotImplementedError("args.dataset_cat is required.")

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 2]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 0]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_detail_labels():
    return np.array([
        [194, 230, 254], [111, 193, 223], [132, 132, 192], [184, 131, 237], [164, 176, 223], [138, 113, 246], [254, 38, 229],
        [81, 50, 197], [78, 4, 252], [42, 65, 247], [0, 0, 115], [18, 177, 246], [0, 122, 255], [27, 88, 199], [191, 255, 255],
        [168, 230, 244], [102, 249, 247], [10, 228, 245], [115, 220, 223], [44, 177, 184], [18, 145, 184], [0, 100, 170],
        [44, 160, 51], [64, 79, 10], [51, 102, 51], [148, 213, 161], [90, 228, 128], [90, 176, 113], [51, 126, 96], [208, 167, 180],
        [153, 116, 153], [162, 30, 124], [236, 219, 193], [202, 197, 171], [165, 182, 171],[138, 90, 88], [172, 181, 123],
        [255, 242, 159], [255, 167, 62], [255, 109, 93], [255, 57, 23], [0, 0, 0], [255, 255, 255]
        ])

def get_mid_labels():
    return np.array([
        [255, 100, 255], [70, 50, 250], [80, 100, 180], [100, 80, 190], [50, 140, 240], [110, 130, 250], [90, 240, 220],
        [50, 240, 140], [90, 230, 190], [0, 240, 240], [20, 170, 150], [80, 210, 30], [210, 210, 70], [230, 250, 140],
        [70, 130, 70], [50, 150, 100], [200, 80, 200], [200, 90, 100], [160, 150, 130], [130, 180, 160], [250, 10, 10],
        [200, 120, 40], [0, 0, 0], [255, 255, 255]
    ])

def get_main_labels():
    return np.array([
        [100, 0, 255], [0, 200, 200], [100, 255, 0], [10, 100, 0], [100, 10, 80], [90, 90, 100], [200, 100, 10], [0, 0, 0], [255, 255, 255]
    ])



def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])