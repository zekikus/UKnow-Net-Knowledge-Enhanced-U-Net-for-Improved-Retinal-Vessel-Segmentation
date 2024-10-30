import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component


class Tester():
    def __init__(self, model, test_loader, dataset_path, path, seed, show=False):

        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show
        self.path = path
        self.seed = seed
        if self.show:
            dir_exists(f"{self.path}/save_picture_{self.seed}")
        cudnn.benchmark = True

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.pre = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()

    def _metrics_update(self, auc, f1, acc, sen, spe, pre, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.pre.update(pre)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "pre": self.pre.average,
            "IOU": self.iou.average
        }

    def test(self, result_df, data_flag, encoder_name, seed):
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img)
                self.batch_time.update(time.time() - tic)

                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)
                img = img[0,0,...]
                gt = gt[0,0,...]
                pre = pre[0,0,...]
                if self.show:
                    predict = torch.sigmoid(pre).cpu().detach().numpy()
                    predict_b = np.where(predict >= 0.5, 1, 0)
                    cv2.imwrite(
                        f"{self.path}/save_picture_{self.seed}/img{i}.png", np.uint8(img.cpu().numpy()*255))
                    cv2.imwrite(
                        f"{self.path}/save_picture_{self.seed}/gt{i}.png", np.uint8(gt.cpu().numpy()*255))
                    cv2.imwrite(
                        f"{self.path}/save_picture_{self.seed}/pre{i}.png", np.uint8(predict*255))
                    cv2.imwrite(
                        f"{self.path}/save_picture_{self.seed}/pre_b{i}.png", np.uint8(predict_b*255))

                self._metrics_update(*get_metrics(pre, gt, 0.5).values())

                '''
                tbar.set_description(
                    'TEST ({}) | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                '''
                tic = time.time()
        
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')

        # Data Frame
        new_row = dict()
        new_row.update({"DATASET": data_flag, "ENCODER_NAME": encoder_name, "SEED": seed})
        new_row.update({f'{str(k)}' : v for k, v in self._metrics_ave().items()})
        result_df.loc[len(result_df.index)] = new_row

        return result_df
        """
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')
        """