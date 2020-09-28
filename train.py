import argparse
import os
import numpy as np
from tqdm import tqdm
# from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver : 학습결과물 저장에 대한 설정(저장위치, 파라미터 로그)
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary : Tensorboard 사용에 대한 정의
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader : Train, Test 데이터를 파이토치 모델입력에 맞게 처리하는 Dataloader 호출
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network : 설정한 파라미터 적용하여 딥랩 모델 호출
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
            classes_weights_path = os.path.join(args.dataset_path, 'train', 'mask_'+args.dataset_cat, 'classes_weights.npy')
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

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
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

        # Clear start epoch if fine-tuning : fine-tuning을 원하는 설정이 있을시 epoch를 0으로 초기화하여 학습진행
        if args.ft:
            args.start_epoch = 0

    # Training 과정 정의 (image와 타켓 입력 -> 추론 -> loss 계산 -> 역전파(back propagation) -> 매개변수 갱신 -> tensorboard 로그 업데이트)
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.train_loss_final = train_loss / (i + 1)
            tbar.set_description('Train loss: %.3f' % (self.train_loss_final))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            ##  Show 10 * 3 inference results each epoch : 학습도중 tensorboard 에서 추론된 사진을 특정 epoch간격마다 확인하고 싶을 때 주석 해제
            # if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
            #    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)


        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # True 일 경우 매 epoch에 대한 checkpoint를 덮어쓰면서 저장하고 Best miou를 달성한 checkpoint는 따로 저장,
        # False일 경우 Best miou를 달성한 checkpoint만 저장
        if True:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    # Validatino 과정 정의(image와 타켓 입력 -> 추론 -> loss 계산 -> tensorboard 로그 업데이트)
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
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        # 평과 결과 tensorboard 로그 업데이트
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        self.writer.add_scalars('loss',{'train_loss': self.train_loss_final,'val_loss': self.val_loss_final}, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

        print('Loss: %.3f' % test_loss)
        # mIou가 비교 후 Best 값일시 저장
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    # 학습 옵션 및 하이퍼파라미터 사용자 입력부분. default=에 원하는 값 입력
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--dataset_cat', type=str, default='detail',    # 학습할 토지피복도 유형 선택 (detail:세분류, middle:중분류, main:대분류)
                        choices=['detail', 'middle', 'main'], help='category')
    parser.add_argument('--dataset_path', type=str, help='category', default='/home/user/NAS/Internal/Dataset/Dataset_allfor') # 학습데이터 경로의 root 경로 입력(train과 test 폴더가 들어있는 폴더 경로 입력)
    parser.add_argument('--backbone', type=str, default='xception', # Backbone 선택 (xception, resnet, drn, mobilenet 중 선택하여 입력)
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16, # out-stride 입력
                        help='network output stride (default: 16)')
    # parser.add_argument('--dataset', type=str, default='cityscapes',
    #                     choices=['pascal', 'coco', 'cityscapes'],
    #                     help='dataset name (default: pascal)')

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
    parser.add_argument('--resume', type=str, default='/home/user/Work/pytorch-deeplab-xception/run/cityscapes/deeplab-xception/experiment_0/checkpoint.pth.tar',    #학습 재개할 '.pth.tar' 파일 경로(없을시 None입력)
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
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # epoch당 학습이 끝날때마다 평가 진행
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
    main()
