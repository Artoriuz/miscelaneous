from argparse import ArgumentParser
from tqdm import tqdm
from shutil import rmtree
from scipy.ndimage import convolve
import numpy as np
import cv2
import glob
import os
import skimage


def msssim(im1, im2, data_range = 255, channel_axis = None):
    level = 5
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    msssim = []
    for _ in range(level):
        ssim_res = skimage.metrics.structural_similarity(im1 = im1, im2 = im2, data_range = data_range, channel_axis = channel_axis)
        msssim.append(ssim_res)
        if channel_axis:
            filtered_im1 = np.zeros_like(im1)
            filtered_im2 = np.zeros_like(im2)
            for channel in range(channel_axis):
                filtered_im1[:, :, channel] = convolve(im1[:, :, channel], downsample_filter, mode='reflect')
                filtered_im2[:, :, channel] = convolve(im2[:, :, channel], downsample_filter, mode='reflect')
            im1 = filtered_im1[::2, ::2, :]
            im2 = filtered_im2[::2, ::2, :]
        else:    
            filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
            filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
            im1 = filtered_im1[::2, ::2]
            im2 = filtered_im2[::2, ::2]
    return np.average(np.array(msssim), weights=weights)


parser = ArgumentParser(description='IM benchmarking')
parser.add_argument('-p','--path', help='Directory', default="./inputs/")
args = parser.parse_args()
output_file = open("benchmark_result.txt", "w")

EWA = True

resampling_filters = ['Bartlett',
                      'Blackman',
                      'Bohman',
                      'Box',
                      'Catrom',
                      'Cosine',
                      'Cubic',
                      'Gaussian',
                      'Hamming',
                      'Hann',
                      'Hermite',
                      'Jinc',
                      'Kaiser',
                      'Lagrange',
                      'Lanczos',
                      'Lanczos2',
                      'Lanczos2Sharp',
                      'LanczosRadius',
                      'LanczosSharp',
                      'Mitchell',
                      'Parzen',
                      'Point',
                      'Quadratic',
                      'Robidoux',
                      'RobidouxSharp',
                      'Sinc',
                      'SincFast',
                      'Spline',
                      'CubicSpline',
                      'Triangle',
                      'Welch']


print("Downsampling images")
os.mkdir("box_ref")
os.system(f"magick mogrify -colorspace rgb -filter box -resize 50% -colorspace srgb -path box_ref {args.path}*.png")

print("Starting benchmarks")
for resampling_filter in tqdm(resampling_filters):
    os.mkdir("./low_res")
    mae_list = []
    psnr_list = []
    ssim_list = []
    msssim_list = []

    if EWA:
        os.system(f"magick mogrify -colorspace RGB -filter {resampling_filter} -distort Resize 50% -colorspace sRGB -path low_res {args.path}*.png")
    else:
        os.system(f"magick mogrify -colorspace RGB -filter {resampling_filter} -resize 50% -colorspace sRGB -path low_res {args.path}*.png")
    
    ref_filelist = sorted(glob.glob("box_ref/*.png"))
    test_filelist = sorted(glob.glob("low_res/*.png"))
    for ref_file, test_file in zip(ref_filelist, test_filelist):
        ref_image = cv2.imread(ref_file, cv2.IMREAD_COLOR).astype(float) / 255.0
        test_image = cv2.imread(test_file, cv2.IMREAD_COLOR).astype(float) / 255.0
        # ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)       

        mae_score = np.mean(np.absolute(ref_image - test_image))
        psnr0 = skimage.metrics.peak_signal_noise_ratio(ref_image, test_image)
        ssim0 = skimage.metrics.structural_similarity(ref_image, test_image, data_range = 1, channel_axis = 2)
        msssim0 = msssim(ref_image, test_image, data_range = 1, channel_axis = 2)

        mae_list.append(mae_score)
        psnr_list.append(psnr0)
        ssim_list.append(ssim0)
        msssim_list.append(msssim0)

    mae_result = np.mean(np.array(mae_list))
    psnr_result = np.mean(np.array(psnr_list))
    ssim_result = np.mean(np.array(ssim_list))
    msssim_result = np.mean(np.array(msssim_list))

    rmtree("./low_res")

    if EWA:
        print(f"Polar_{resampling_filter} - MAE: {mae_result}, PSNR: {psnr_result}, SSIM: {ssim_result}, MSSSIM: {msssim_result}\n")
        output_file.write(f"Polar_{resampling_filter},{mae_result},{psnr_result},{ssim_result},{msssim_result}\n")
    else:
        print(f"{resampling_filter} - MAE: {mae_result}, PSNR: {psnr_result}, SSIM: {ssim_result}, MSSSIM: {msssim_result}\n")
        output_file.write(f"{resampling_filter},{mae_result},{psnr_result},{ssim_result},{msssim_result}\n")

rmtree("./box_ref")
output_file.close()
