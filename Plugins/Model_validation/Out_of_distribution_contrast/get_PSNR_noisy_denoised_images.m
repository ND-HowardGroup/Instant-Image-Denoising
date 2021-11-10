clc 
close all
clear variables

%date: 4th October 2021
%VM: test bias_free bateria image 

font = 14;
format long;

addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0410/noisy_images/');

target = imread('AVG_2mw_OD=0_Myxo_1mM Ferri in 240mM KNO3_CV_9.png');
%target = read(target);
target = double(target);

target1= imread('noisy_images/2mw_OD=0_Myxo_1mM_90000.tif'); 
target1 = double(target1);

target2 = imread('noisy_images/Noisy_images_avg2.png'); 
target2 = double(target2);

target3 = imread('noisy_images/Noisy_images_avg4.png');
target3 = double(target3);

target4 = imread('noisy_images/Noisy_images_avg8.png');
target4 = double(target4);

target5 = imread('noisy_images/Noisy_images_avg16.png');
target5 = double(target5);


target11= imread('denoised_images/outputImage_Noise2Noise_Denoised_00.tif'); 
target11 = double(target11);

target12 = imread('denoised_images/outputImage_Noise2Noise_Denoised_avg2.tif'); 
target12 = double(target12);

target13 = imread('denoised_images/outputImage_Noise2Noise_Denoised_avg4.tif');
target13 = double(target13);

target14 = imread('denoised_images/outputImage_Noise2Noise_Denoised_avg8.tif');
target14 = double(target14);

target15 = imread('denoised_images/outputImage_Noise2Noise_Denoised_avg16.tif');
target15 = double(target15);


%psnr calculations
psnr_1  = calculate_PSNR_new(target1, target);
psnr_2  = calculate_PSNR_new(target2, target);
psnr_3  = calculate_PSNR_new(target3, target);
psnr_4  = calculate_PSNR_new(target4, target);
psnr_5  = calculate_PSNR_new(target5, target);
psnr_6  = calculate_PSNR_new(target11, target);
psnr_7  = calculate_PSNR_new(target12, target);
psnr_8  = calculate_PSNR_new(target13, target);
psnr_9  = calculate_PSNR_new(target14, target);
psnr_10  = calculate_PSNR_new(target15, target);


psnr_vals = [psnr_1 psnr_2  psnr_3 psnr_4 psnr_5 psnr_6 psnr_7 psnr_8 psnr_9 psnr_10]';
psnr_names = ["psnr_1" "psnr_2"  "psnr_3" "psnr_4" "psnr_5" "psnr_6" "psnr_7" "psnr_8" "psnr_9" "psnr_10" ]';


ssim_1 = ssim(target, target1);
ssim_2 = ssim(target, target2);
ssim_3 = ssim(target, target3);
ssim_4 = ssim(target, target4);
ssim_5 = ssim(target, target5);
ssim_6 = ssim(target, target11);
ssim_7 = ssim(target, target12);
ssim_8 = ssim(target, target13);
ssim_9 = ssim(target, target14);
ssim_10 = ssim(target, target15);

SSIM_vals = [ ssim_1 ssim_2  ssim_3 ssim_4 ssim_5 ssim_6 ssim_7 ssim_8 ssim_9 ssim_10]';
SSIM_names = ["ssim_1" "ssim_2"  "ssim_3" "ssim_4" "ssim_5" "ssim_6" "ssim_7" "ssim_8" "ssim_9" "ssim_10" ]';