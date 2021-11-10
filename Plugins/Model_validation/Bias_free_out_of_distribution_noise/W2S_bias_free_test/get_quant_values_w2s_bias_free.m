clc 
close all
clear variables

%date: 21st September 2021
%VM: test bias_free FMD image 

font = 14;
format long;

addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Sep21/2109/w2s_bias_free_test/');

target = imread('target_avg400_010_0.tif');
%target = read(target);
target = double(target);


target1= imread('outputImage_Noise2Noise_Denoised_5dB.tif'); 
target1 = double(target1);

target2 = imread('outputImage_Noise2Noise_Denoised_10dB.tif'); 
target2 = double(target2);

target3 = imread('outputImage_Noise2Noise_Denoised_15dB.tif');
target3 = double(target3);

target4 = imread('outputImage_Noise2Noise_Denoised_20dB.tif');
target4 = double(target4);

target5 = imread('outputImage_Noise2Noise_Denoised_25dB.tif');
target5 = double(target5);

target6 = imread('outputImage_Noise2Noise_Denoised_30dB.tif');
target6 = double(target6);

target7 = imread('outputImage_Noise2Noise_Denoised_35dB.tif');
target7 = double(target7);

target8 = imread('outputImage_Noise2Noise_Denoised_40dB.tif');
target8 = double(target8);

target9 = imread('outputImage_Noise2Noise_Denoised_45dB.tif');
target9 = double(target9);

target10 = imread('outputImage_Noise2Noise_Denoised_50dB.tif');
target10 = double(target10);


%psnr calculations
psnr_1  = calculate_PSNR_new(target1, target);
psnr_2  = calculate_PSNR_new(target2, target);
psnr_3  = calculate_PSNR_new(target3, target);
psnr_4  = calculate_PSNR_new(target4, target);
psnr_5  = calculate_PSNR_new(target5, target);
psnr_6  = calculate_PSNR_new(target6, target);
psnr_7  = calculate_PSNR_new(target7, target);
psnr_8  = calculate_PSNR_new(target8, target);
psnr_9  = calculate_PSNR_new(target9, target);
psnr_10  = calculate_PSNR_new(target10, target);

psnr_vals = [psnr_1 psnr_2  psnr_3 psnr_4 psnr_5 psnr_6 psnr_7 psnr_8 psnr_9 psnr_10]';
psnr_names = ["psnr_1" "psnr_2"  "psnr_3" "psnr_4" "psnr_5" "psnr_6" "psnr_7" "psnr_8" "psnr_9" "psnr_10"]';


ssim_1 = ssim(target, target1);
ssim_2 = ssim(target, target2);
ssim_3 = ssim(target, target3);
ssim_4 = ssim(target, target4);
ssim_5 = ssim(target, target5);
ssim_6 = ssim(target, target6);
ssim_7 = ssim(target, target7);
ssim_8 = ssim(target, target8);
ssim_9 = ssim(target, target9);
ssim_10 = ssim(target, target10);


SSIM_vals = [ ssim_1 ssim_2  ssim_3 ssim_4 ssim_5 ssim_6 ssim_7 ssim_8 ssim_9 ssim_10]';
SSIM_names = ["ssim_1" "ssim_2"  "ssim_3" "ssim_4" "ssim_5" "ssim_6" "ssim_7" "ssim_8" "ssim_9" "ssim_10"]';