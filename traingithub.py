import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm
import argparse
from typing import Any, Tuple
from PIL import Image
import numpy as np
from models import DNnet_train
from torch.optim.lr_scheduler import StepLR
import os
import os.path as osp
import time
import datetime
import sys
from utils.misc import print_args, Logger
from utils.seed_utils import init_seeds
from utils.metric_utils import MetricLogger, SmoothedValue
from torchsummary import summary
from utils.combined_loss import combinedloss
from utils import metric_utils as utils
from torchvision.utils import save_image
from glob import glob
from os.path import join
from ntpath import basename
from utils.ssim_psnr_utils import getSSIM, getPSNR
from mmengine.config import Config

best_psnr = 0
best_ssim = 0
best_mse = 10000

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))

    ssims, psnrs, mses = [], [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        if (gtr_f == gen_f):
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L");
            g_im = g_im.convert("L")
            psnr, mse = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
            mses.append(mse)
    return np.array(ssims), np.array(psnrs), np.array(mses)


class UIE_Dataset(VisionDataset):
    def __init__(self, rawname: str, gtname: str, data_name: str, cache_dir,
                 root: str, train: bool = True, size: int = 256, ) -> None:
        super().__init__()
        self.train = train
        self.size = size
        self.rawcache = self.find_npy_file(cache_dir+data_name+'/', rawname)
        self.gtcache = self.find_npy_file(cache_dir + data_name + '/', gtname)

        if self.rawcache is not None and self.gtcache is not None:
            self.raw_image = np.load(cache_dir+data_name+'/'+rawname+'.npy')
            self.GT_image = np.load(cache_dir + data_name + '/' + gtname + '.npy')
            print(self.raw_image.shape)
            print(self.GT_image.shape)
            print("成功读取缓存文件")

        else:
            folder_raw = root + rawname + '/'
            folder_GT = root + gtname + '/'
            if self.train:
                text1 = "Caching train raw data"
                text2 = "Caching train GT data"
            else:
                text1 = "Caching test raw data"
                text2 = "Caching test GT data"

            self.raw_image = self.load_images_and_flip(folder_raw, text1)
            self.GT_image = self.load_images_and_flip(folder_GT, text2)
            if not os.path.exists(cache_dir+data_name+'/'):
                os.makedirs(cache_dir+data_name+'/')
            np.save(cache_dir+data_name+'/'+rawname+'.npy', np.transpose(self.raw_image, (0, 3, 1, 2)))
            np.save(cache_dir + data_name + '/' + gtname + '.npy', np.transpose(self.GT_image, (0, 3, 1, 2)))
            print("成功在如下位置创建缓存: " + cache_dir+data_name+'/')

        if len(self.raw_image) == len(self.GT_image):
            self.len = len(self.raw_image) // 2
        else:
            print("原始数据集和真值数据集数量不一致，请检查数据集")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.train:
            random_nums = np.random.random(size=1)
            random_nums = np.where(random_nums > 0.5, 1, 0)
            index_ = int(2 * index + random_nums)
        else:
            index_ = int(2 * index)
        raw, GT = self.raw_image[index_], self.GT_image[index_]
        return raw, GT

    def find_npy_file(self, given_path, cache_name):
        file_name = f"{cache_name}.npy"
        for root, dirs, files in os.walk(given_path):
            if file_name in files:
                file_path = os.path.join(root, file_name)
                return file_path
        return None

    def load_images_and_flip(self, folder_path, text):
        images = []
        image_list = [folder_path + i for i in os.listdir(folder_path)]
        for i, (file_path) in enumerate(tqdm(image_list, desc=text)):
            if file_path.endswith('.png'):  # jpg格式会损失精度，因此只支持png
                img = Image.open(file_path)
                img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)
                images.append(img)
                img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                images.append(img_flip)

        if len(images) == 0:
            print("未读取到数据集，请检查数据集格式或路径，注意只推荐png格式")
        return np.array(images)/255.0

    def __len__(self) -> int:
        return self.len


class UIE_PrefetchLoader:
    def __init__(
            self, loader, fp16=False):
        self.loader = loader
        self.fp16 = fp16

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_raw, next_GT in self.loader:
            with torch.cuda.stream(stream):
                next_raw = next_raw.cuda(non_blocking=True)
                next_GT = next_GT.cuda(non_blocking=True)
                if self.fp16:
                    next_raw = next_raw.half()
                    next_GT = next_GT.half()
                else:
                    next_raw = next_raw.float()
                    next_GT = next_GT.float()

            if not first:
                yield raw, GT
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            raw = next_raw
            GT = next_GT
        yield raw, GT

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, args, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return metric_logger.loss.global_avg


def evaluate(model, criterion, data_loader, args):
    model.eval()
    GT_savemark = False
    save_generate = args.output_dir + '/generate_samples/'
    if not osp.exists(save_generate):
        os.makedirs(save_generate)
    save_GT = args.output_dir + '/GT_samples/'
    if not osp.exists(save_GT):
        os.makedirs(save_GT)
        GT_savemark = True
    with torch.inference_mode():
        for i, (image, target) in enumerate(data_loader):
            output = model(image)
            loss = criterion(output, target)

            save_image(output.data, save_generate + ' (' + str(i + 1) + ').png')
            if GT_savemark:
                save_image(target.data, save_GT + ' (' + str(i + 1) + ').png')

    SSIM_measures, PSNR_measures, MSE_measures = SSIMs_PSNRs(save_GT, save_generate)

    ssim = np.mean(SSIM_measures)
    psnr = np.mean(PSNR_measures)
    mse = np.mean(MSE_measures)
    print("SSIM >> Mean: {0:.6f} std: {1:.6f}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
    print("PSNR >> Mean: {0:.6f} std: {1:.6f}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
    print("MSE >> Total: {0:.6f} std: {1:.6f}".format(np.mean(MSE_measures), np.std(MSE_measures)))
    return psnr, ssim, mse


def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, scaler, config):
    global best_psnr
    global best_ssim
    global best_mse

    if config.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, test_dataloader, args=config)
        return

    # 开始训练
    print("[INFO] Start training")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.num_epochs):

        train_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, epoch, config, scaler)
        lr_scheduler.step()
        psnr, ssim, mse = evaluate(model, criterion, test_dataloader, args=config)

        is_best_psnr = psnr > best_psnr
        is_best_ssim = ssim > best_ssim
        is_best_mse = mse < best_mse

        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        best_mse = min(mse, best_mse)

        if config.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": config,
                "best_psnr": best_psnr,
            }
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()

            # save best checkpoint
            if is_best_psnr:
                print(f"\n[FEAT] Best PSNR: {best_psnr:.3f}\n")
                utils.save_on_master(checkpoint, osp.join(config.output_dir, "best_psnr_model.pth"))
            if is_best_ssim:
                print(f"\n[FEAT] Best SSIM: {best_ssim:.3f}\n")
                utils.save_on_master(checkpoint, osp.join(config.output_dir, "best_ssim_model.pth"))
            if is_best_mse:
                print(f"\n[FEAT] Best MSE: {best_mse:.3f}\n")
                utils.save_on_master(checkpoint, osp.join(config.output_dir, "best_mse_model.pth"))
            utils.save_on_master(checkpoint, osp.join(config.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\n[INFO] Best PSNR: {best_psnr:.3f}, Best SSIM: {best_ssim:.3f}, Best MSE: {best_mse:.3f}")
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument('--data_path', type=str, default="./datasets/UIEB/", help='Path of dataset')
    # 数据集名称
    parser.add_argument('--data_name', type=str, default="UIEB", help='Name of dataset')
    # 原始水下图像训练集路径
    parser.add_argument('--train_raw_dir_name', type=str, default="train_raw",
                        help='Folder of raw underwater images for training')
    # 真值图像训练集路径
    parser.add_argument('--train_GT_dir_name', type=str, default="train_GT",
                        help='Folder of GT underwater images for training')
    # 原始水下图像测试集路径
    parser.add_argument('--test_raw_dir_name', type=str, default="test_raw",
                        help='Folder of raw underwater images for testing')
    # 真值图像测试集路径
    parser.add_argument('--test_GT_dir_name', type=str, default="test_GT",
                        help='Folder of GT underwater images for testing')
    # 是否读取数据集缓存
    parser.add_argument('--read_cache', type=bool, default=True,
                        help='Recommended when conducting multiple experiments in the same dataset')
    # 读取数据集缓存路径
    parser.add_argument('--cache_dir', type=str, default="./cache_dir/",
                        help='Recommended when conducting multiple experiments in the same dataset')
    # 模型保存路径
    parser.add_argument("--output-dir", default="./work_dir", type=str, help="Path to save outputs")
    # 否：训练 是：测试
    parser.add_argument("--test_only", default=False, help="Only test the model if True")
    # 将test_only置为True后，此处填写需要加载的权重
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint to load")
    # 混合精度训练
    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")
    # 模型名称
    parser.add_argument("--model_name", default="DNnet",
                        help="The speed setting of the model corresponds to the parameter n in our paper")
    # 模型的速度设置，对应论文中的参数n
    parser.add_argument("--model_speed", default=1,
                        help="The speed setting of the model corresponds to the parameter n in our paper")
    # 运行的设备
    parser.add_argument('--device', default="cuda", help="default cuda device")
    # 学习率设置
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    # steplr的步长间隔
    parser.add_argument('--lr_step_size', type=int, default=6, help="step_size of steplr")
    # steplr的gama值
    parser.add_argument('--lr_gamma', type=float, default=0.7, help="gamma of steplr")
    # 总训练轮数,由于调度原因总轮次增加一
    parser.add_argument('--num_epochs', type=int, default=31)
    # 总训练轮数
    parser.add_argument('--start_epoch', type=int, default=0)
    # 训练批次大小
    parser.add_argument('--train_batch_size', type=int, default=8)
    # 测试批次大小
    parser.add_argument('--test_batch_size', type=int, default=1)
    # 训练和测试的尺寸，n=3时需要修改为255
    parser.add_argument('--size', type=int, default=256,
                        help="resize images, default:resize images to 256*256")
    # 优化器weight decay设置
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay")
    # 打印训练信息的频率
    parser.add_argument('--print_freq', type=int, default=100)
    # 设置训练种子
    parser.add_argument('--seed', type=int, default=0)
    # 设置训练workers
    parser.add_argument('--workers', type=int, default=2)
    config = parser.parse_args()
    config = Config(vars(config))

    if RANK in {-1, 0}:
        timestamp = datetime.datetime.now()
        # 创建输出结果保存路径
        config.output_dir = osp.join(
            config.output_dir,
            config.model_name,
            config.data_name,
            timestamp.strftime('%Y%m%d/%H%M%S'),
        )
        if not osp.exists(config.output_dir):
            os.makedirs(config.output_dir)
        # 日志文件名
        log_file_name = f"{config.data_name}-{config.model_name}.log"
        # 将日志在控制台和文件都打印
        sys.stdout = Logger(osp.join(config.output_dir, log_file_name))
        print(f"[INFO] rank: {RANK}")
        print(f"[INFO] result path: {osp.abspath(config.output_dir)}\n", flush=True)

    print_args(config)
    init_seeds(seed=config.seed)

    print("[INFO] Loading data")

    train_dataset = UIE_Dataset(root=config.data_path, train=True, size=config.size,
                                rawname=config.train_raw_dir_name, gtname=config.train_GT_dir_name,
                                cache_dir=config.cache_dir, data_name=config.data_name)
    test_dataset = UIE_Dataset(root=config.data_path, train=False, size=config.size,
                               rawname=config.test_raw_dir_name, gtname=config.test_GT_dir_name,
                               cache_dir=config.cache_dir, data_name=config.data_name)

    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = UIE_PrefetchLoader(
        DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=train_sampler,
                   num_workers=config.workers, pin_memory=True))

    test_dataloader = UIE_PrefetchLoader(
        DataLoader(test_dataset, batch_size=config.test_batch_size, sampler=test_sampler,
                   num_workers=config.workers, pin_memory=True))

    print("[INFO] Creating model")

    model = DNnet_train.DNnet(n=config.model_speed)
    model.to(config.device)

    if config.resume and config.test_only:
        checkpoint = torch.load(config.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    summary(model, (3, 256, 256), batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    lr_scheduler = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    criterion = combinedloss()

    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    train_model(model, criterion, optimizer, train_dataloader, test_dataloader, scaler, config)
