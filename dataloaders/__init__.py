# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, Landcover_detail, Landcover_mid, Landcover_main
from dataloaders.datasets import Landcover
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    train_set = Landcover.LandcoverSegmentation(args, split='train')
    val_set = Landcover.LandcoverSegmentation(args, split='test')
    test_set = Landcover.LandcoverSegmentation(args, split='test')

    num_class = train_set.NUM_CLASSES
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class
    #
    # if args.dataset == 'pascal':
    #     train_set = pascal.VOCSegmentation(args, split='train')
    #     val_set = pascal.VOCSegmentation(args, split='val')
    #     if args.use_sbd:
    #         sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #         train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
    #
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #
    #     return train_loader, val_loader, test_loader, num_class
    #
    # elif args.dataset == 'cityscapes':
    #     if args.dataset_cat == 'detail':
    #         train_set = Landcover_detail.LandcoverSegmentation(args, split='train')
    #         val_set = Landcover_detail.LandcoverSegmentation(args, split='val')
    #         test_set = Landcover_detail.LandcoverSegmentation(args, split='test')
    #     elif args.dataset_cat == 'middle':
    #         train_set = Landcover_mid.LandcoverSegmentation(args, split='train')
    #         val_set = Landcover_mid.LandcoverSegmentation(args, split='val')
    #         test_set = Landcover_mid.LandcoverSegmentation(args, split='test')
    #     elif args.dataset_cat == 'main':
    #         train_set = Landcover_main.LandcoverSegmentation(args, split='train')
    #         val_set = Landcover_main.LandcoverSegmentation(args, split='val')
    #         test_set = Landcover_main.LandcoverSegmentation(args, split='test')
    #     else:
    #         raise print("choose dataset category!")
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #
    #     return train_loader, val_loader, test_loader, num_class
    #
    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class
    #
    # else:
    #     raise NotImplementedError

