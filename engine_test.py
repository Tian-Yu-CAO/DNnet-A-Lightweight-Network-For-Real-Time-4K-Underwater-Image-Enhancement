import os
import time
import torch
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from PIL import Image
from glob import glob
import pycuda.autoinit
import tensorrt as trt
from os.path import join
from ntpath import basename
import pycuda.driver as cuda
from tabulate import tabulate
from torchvision import transforms
from utils.uiqm_utils import getUIQM
from utils.dataloader import myDataSet
from torch.utils.data import DataLoader
from utils.ssim_psnr_utils import getSSIM, getPSNR


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = [
    "tes",
    "setup",]


def ssims_psnrs(gtr_dir, gen_dir, im_res=(256, 256)):
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
        if gtr_f == gen_f:
            # assumes same filenames
            r_im = Image.open(gtr_path).resize(im_res)
            g_im = Image.open(gen_path).resize(im_res)
            # get ssim on RGB channels
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            # get psnt on L channel (SOTA norm)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr, mse = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
            mses.append(mse)
    return np.array(ssims), np.array(psnrs), np.array(mses)


def measure_uiqms(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)


@torch.no_grad()
def tes(config, test_dataloader):
    # Deserialize the CUDA engine from file
    engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(open(config.trt_engine_path, "rb").read())
    context = engine.create_execution_context()

    # Prepare input data
    input_shape = (1, 3, 256, 256)
    d_input = cuda.mem_alloc(int(1 * np.prod(input_shape) * np.float32().itemsize))
    d_output = cuda.mem_alloc(int(1 * np.prod(input_shape) * np.float32().itemsize))

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    assert (len(tensor_names) == 2)

    context.set_tensor_address(tensor_names[0], int(d_input))
    context.set_tensor_address(tensor_names[1], int(d_output))
    stream = cuda.Stream()

    print("-------------------Inferencing----------------------")
    t0 = 0
    # Process each image in the dataloader
    for i, (img, _, name) in enumerate(tqdm(test_dataloader, desc="Processing images")):
        start_time = time.time()
        # Ensure image data is of correct shape and type
        img = img.numpy() if hasattr(img, 'numpy') else img
        img = img.astype(np.float32)
        img = np.ascontiguousarray(img)

        cuda.memcpy_htod_async(d_input, img, stream)

        context.execute_async_v3(stream.handle)  # Execute model

        output = np.empty(input_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)

        tensor = torch.from_numpy(output)
        torchvision.utils.save_image(tensor.data, config.output_images_path + name[0])

        t0 += time.time() - start_time

    d_input.free()
    d_output.free()
    return t0


def setup(config):
    transform = transforms.Compose(
        [transforms.Resize([config.resize, config.resize]),
         transforms.ToTensor()])  # Preprocessing

    test_dataset = myDataSet(config.test_images_path, None, transform, False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print("Test Dataset Reading Completed.\n")
    return test_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_engine_path', type=str, default="./Engine/dnnet_n1_256.engine",
                        help='path to the TensorRT engine')
    parser.add_argument('--test_images_path', type=str, default="./Datasets/UIEB/input/", help='path of input images')
    parser.add_argument('--output_images_path', type=str, default='./TOutput/',
                        help='path to save test generated images')
    parser.add_argument('--batch_size', type=int, default=1, help="default : 1")
    parser.add_argument('--resize', type=int, default=256, help="resize images, default:256*256")
    parser.add_argument('--calculate_metrics', type=bool, default=True,
                        help="calculate PSNR, SSIM MSE and UIQM on test images")
    parser.add_argument('--label_images_path', type=str, default="./Datasets/UIEB/GT/", help='path to label images')
    parser.add_argument('--device', type=str, default="cuda", help='device to process')

    print("-------------------Initializing---------------------")
    configs = parser.parse_args()
    if not os.path.exists(configs.output_images_path):
        os.mkdir(configs.output_images_path)

    test_loader = setup(configs)
    process_time = tes(configs, test_loader)

    if configs.calculate_metrics:
        print("\nCalculating metrics may take some time...\n")
        UIQM_measures = measure_uiqms(configs.output_images_path)
        SSIM_measures, PSNR_measures, MSE_measures = ssims_psnrs(configs.label_images_path, configs.output_images_path)

        data = [
            ["UQIM", "{:.3f}".format(np.mean(UIQM_measures)), "{:.3f}".format(np.std(UIQM_measures))],
            ["SSIM", "{:.3f}".format(np.mean(SSIM_measures)), "{:.3f}".format(np.std(SSIM_measures))],
            ["PSNR", "{:.3f}".format(np.mean(PSNR_measures)), "{:.3f}".format(np.std(PSNR_measures))],
            ["MSE", "{:.3f}".format(np.mean(MSE_measures)/1000), "{:.3f}".format(np.std(MSE_measures)/1000)]
        ]

        headers = ["Metric", "Mean", "Std"]
        print(tabulate(data, headers=headers, tablefmt="grid", stralign="left"))
