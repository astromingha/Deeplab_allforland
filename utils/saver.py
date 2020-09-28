import os
import shutil
import torch
from collections import OrderedDict
import glob
from dataloaders.datasets import Landcover
import json
import pandas as pd

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.cityscapes_train = Landcover.LandcoverSegmentation(args, split='train')

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['category'] = self.args.dataset_cat
        p['backbone'] = self.args.backbone
        p['valid_class_num'] = len(self.cityscapes_train.valid_classes)
        p['mean'] = json.dumps(self.cityscapes_train.mean)
        p['std'] = json.dumps(self.cityscapes_train.std)
        p['out_stride'] = self.args.out_stride
        p['batch_size'] = self.args.batch_size
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size
        p['gpu_num'] = self.args.gpu_ids
        p['resume'] = self.args.resume
        p['use_balanced_weights'] = self.args.use_balanced_weights

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def save_metrics(self,iou, confusion_matrix):
        ioufile = os.path.join(self.experiment_dir,'iou.csv')
        conf_matrixfile = os.path.join(self.experiment_dir,'confustion_matrix.csv')

        dataframe1 = pd.DataFrame(np.transpose(iou))
        dataframe2 = pd.DataFrame(np.transpose(confusion_matrix))
        dataframe1.to_csv(ioufile, header=False, index=False)
        dataframe2.to_csv(conf_matrixfile, header=False, index=False)