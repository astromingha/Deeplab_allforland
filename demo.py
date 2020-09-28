#
# demo.py
#
import argparse
import os
import numpy as np

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset_cat', type=str, default='detail',                # 추론할 토지피복도 유형 선택 (detail:세분류, middle:중분류, main:대분류)
                        choices=['detail', 'middle', 'main'], help='category')
    parser.add_argument('--in_path', type=str,  default='/home/user/')              # 추론할 이미지가 들어있는 폴더 경로 입력
    parser.add_argument('--out_path', type=str,  default='/home/user/pytest/refer')             # 추론된 이미지 결과가 저장될 폴더 경로 입
    parser.add_argument('--backbone', type=str, default='xception',                 # 추론 네트워크 백본 선택(checkpoint 와 동일해야 함)
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='run/cityscapes/deeplab-xception/experiment_0/checkpoint.pth.tar',  #checkpoint 경로 입력(checkpoint이름 포함)
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,                       #out stride 값(checkpoint와 동일해야 함)
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=True,            #True 시 GPU 사용안
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',                        #사용할 GPU id
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--crop-size', type=int, default=513,                       #사진 입력 사이즈
                        help='crop image size')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.dataset_cat == 'detail':
        num_class = 43
    elif args.dataset_cat == 'middle':
        num_class = 24
    elif args.dataset_cat == 'main':
        num_class = 9
    else:
        raise NotImplementedError

    model = DeepLab(num_classes=43,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=False,
                    freeze_bn=False)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    img_list = [i for i in os.listdir(args.in_path) if '.jpg' in i.lower() or '.png' in i.lower()]

    if not len(img_list):
        raise NotImplementedError("No image to infer")

    for imgfile in img_list:
        image = Image.open(args.in_path + '/'+imgfile).convert('RGB')
        target = Image.open(args.in_path+ '/'+imgfile).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))
        save_image(grid_image, args.out_path+'/'+imgfile)
    print("type(grid) is: ", type(grid_image))
    print("grid_image.shape is: ", grid_image.shape)


if __name__ == "__main__":
   main()