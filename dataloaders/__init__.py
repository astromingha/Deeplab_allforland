from dataloaders.datasets import Landcover
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    train_set = Landcover.LandcoverSegmentation(args, split='train')
    val_set = Landcover.LandcoverSegmentation(args, split='valid')
    test_set = Landcover.LandcoverSegmentation(args, split='valid')

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class


