import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator




class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader : Train, Test 데이터를 파이토치 모델입력에 맞게 처리하는 Dataloader 호출
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define Tensorboard Summary : Tensorboard 사용에 대한 정의
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer : Optimizer  호출
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # whether to use class balanced weights : weight balancing 적용 여부 확인 후 반영
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # Define Criterion : loss function 정의
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator : Evaluator 정의
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler : learning rate scheduler 정의
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda : Nvidia GPU사용 설정
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint : 학습 재개를 원하는 checkpoint설정이 있을시 호출하여 학습재개
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            self.val_loss_final = test_loss / (i + 1)
            tbar.set_description('Test loss: %.3f' % (self.val_loss_final))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training : Acc, Acc_class, mIoU 등 평가지표 계산 진행
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, iou, confusion_matrix = self.evaluator.Mean_Intersection_over_Union_IOU()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        
        # checkpoint 저장위치에 class별 iou와 confusion_matrix csv로 저장
        self.saver.save_metrics(iou, confusion_matrix)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        print('Loss: %.3f' % test_loss)

def main():
    # 학습 옵션 및 하이퍼파라미터 사용자 입력부분. default=에 원하는 값 입력
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset_cat', type=str, default='detail',    # 학습할 토지피복도 유형 선택 (detail:세분류, middle:중분류, main:대분류, 현재 데이터셋은 세분류만 지원)
                        choices=['detail', 'middle', 'main'], help='category')
    parser.add_argument('--dataset_path', type=str, help='category', default=='../Landcover_dataset') # 학습데이터 경로의 root 경로 입력(train과 test 폴더가 들어있는 Landcover_dataset 경로 입력)
    parser.add_argument('--backbone', type=str, default='xception', # Backbone 선택 (xception, resnet, drn, mobilenet 중 선택하여 입력)
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16, # out-stride 입력
                        help='network output stride (default: 16)')
    parser.add_argument('--workers', type=int, default=1, # 1이상일 경우 멀티 프로세싱으로 진행(추천값 1)
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=513, # 학습 입력데이터 scaling시 기본 사이즈
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=513, # 위 base size에서 scaling된 사진을 잘라내어 입력으로 들어갈 최종 사이즈
                        help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None, # Synchronized BatchNorm 적용 여부 결정 (default=None - 그냥 BatchNorm 사용)
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False, # 위에서 설정된 BatchNorm 관련 파라미터 동결 여부 결정
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce', # loss function 선택(ce - cross entropy, focal - focal loss)
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: auto)') # 학습 epoch 결정
    parser.add_argument('--start_epoch', type=int, default=0, # resume 옵션에 학습 재개할 checkpoint 입력이 없을 경우 start_epoch를 정해줄 수 있음
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None, # 1 step(epoch가 아님)당 학습시킬 사진 수(default=None일시 gpu당 4장씩 학습)
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=None, # 1 step(epoch가 아님)당 평가시킬 사진 수(default=None일시 위 batch_size와 동일값)
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False, # Balaced weight 사용 결정 여부(True일시 학습 전 학습데이터에 대한 balance 계산이 진행됨)
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, metavar='LR', default=None, # learning rate 설정(default=None : 0.01 / (4 * len(args.gpu_ids)) * args.batch_size)
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly', # learning rate scheduler 설정(poly함수, step함수, cosine함수 중 선택)
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9, # Gradient Descent에서 적용할 Momentum 값 입력
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, # Weight decay 값 입력
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, #Nesterov momentum 적용 여부 결정
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=        # GPU 사용을 안 할경우 True
                        False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',     # 사용을 원하는 GPU id 입력
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', # 학습에 사용되는 random값에 대한 시드 결정
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default='run/cityscapes/deeplab-xception/experiment_0/checkpoint.pth.tar',    #학습 재개할 '.pth.tar' 파일 경로(없을시 None입력)
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, # 학습 결과 저장시 폴더 이름 결정(None 일시 'run/deeplab-(backbone이름)'으로 자동저장)
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False, # fine tuning 적용 여부 결정
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,    # 몇 Epoch당 평가를 진행할 것인지 결정(default=1 : 매회)
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False, # 평가 없이 학습만 진행을 원할 때 True 입력(비권장)
                        help='skip validation during training')

    args = parser.parse_args()

    # 위 옵션에서 선택한 값들이 None일시 default 세팅
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = 0.01
        args.lr = lrs / (4 * len(args.gpu_ids)) * args.batch_size

    # checkpoint 가 저장될 폴더 이름 설정(None일시 deeplab_xception으로 자동 저장)
    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    # 랜덤 생성을 위한 시드 설정 후 위에서 정의된 학습과정(Trainer)을 호출하여 epoch 당 학습 시작
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)

    trainer.validation(0)


if __name__ == "__main__":
    main()
