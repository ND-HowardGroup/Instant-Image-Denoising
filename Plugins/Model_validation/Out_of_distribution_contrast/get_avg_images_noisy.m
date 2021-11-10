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

target2 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90001.tif'); 
target2 = double(target2);

target3 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90002.tif');
target3 = double(target3);

target4 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90003.tif');
target4 = double(target4);

target5 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90004.tif');
target5 = double(target5);

target6 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90005.tif');
target6 = double(target6);

target7 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90006.tif');
target7 = double(target7);

target8 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90007.tif');
target8 = double(target8);

target11= imread('noisy_images/2mw_OD=0_Myxo_1mM_90008.tif'); 
target11 = double(target11);

target12 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90009.tif'); 
target12 = double(target12);

target13 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90010.tif');
target13 = double(target13);

target14 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90011.tif');
target14 = double(target14);

target15 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90012.tif');
target15 = double(target15);

target16 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90013.tif');
target16 = double(target16);

target17 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90014.tif');
target17 = double(target17);

target18 = imread('noisy_images/2mw_OD=0_Myxo_1mM_90015.tif');
target18 = double(target18);

Avg1 = target1;
Avg2 = target1+target2;
Avg2 = Avg2/2;

Avg4 = target1+target2+target3+target4;
Avg4 = Avg4/4;

Avg8 = target1+target2+target3+target4+target5+target6+target7+target8;
Avg8 = Avg8/8;

Avg16 = target1+target2+target3+target4+target5+target6+target7+target8+target11+target12+target13+target14+target15+target16+target17+target18;
Avg16 = Avg16/16;

%psnr calculations
psnr_1  = calculate_PSNR_new(target1, target);
psnr_2  = calculate_PSNR_new(target2, target);
psnr_3  = calculate_PSNR_new(target3, target);
psnr_4  = calculate_PSNR_new(target4, target);
psnr_5  = calculate_PSNR_new(target5, target);
psnr_6  = calculate_PSNR_new(target6, target);
psnr_7  = calculate_PSNR_new(target7, target);
psnr_8  = calculate_PSNR_new(target8, target);

psnr_91 = calculate_PSNR_new(Avg1, target);
psnr_92 = calculate_PSNR_new(Avg2, target);
psnr_93 = calculate_PSNR_new(Avg4, target);
psnr_94 = calculate_PSNR_new(Avg8, target);
psnr_95 = calculate_PSNR_new(Avg16, target);

psnr_vals = [psnr_1 psnr_2  psnr_3 psnr_4 psnr_5 psnr_6 psnr_7 psnr_8 ]';
psnr_names = ["psnr_1" "psnr_2"  "psnr_3" "psnr_4" "psnr_5" "psnr_6" "psnr_7" "psnr_8" ]';

% ssim_1 = ssim(target, target1);
% ssim_2 = ssim(target, target2);
% ssim_3 = ssim(target, target3);
% ssim_4 = ssim(target, target4);
% ssim_5 = ssim(target, target5);
% ssim_6 = ssim(target, target6);
% ssim_7 = ssim(target, target7);
% ssim_8 = ssim(target, target8);
% 
% SSIM_vals = [ ssim_1 ssim_2  ssim_3 ssim_4 ssim_5 ssim_6 ssim_7 ssim_8 ]';
% SSIM_names = ["ssim_1" "ssim_2"  "ssim_3" "ssim_4" "ssim_5" "ssim_6" "ssim_7" "ssim_8" ]';
imwrite(uint8(Avg2), 'noisy_images/Noisy_images_avg2.png'); 
imwrite(uint8(Avg4), 'noisy_images/Noisy_images_avg4.png'); 
imwrite(uint8(Avg8), 'noisy_images/Noisy_images_avg8.png'); 
imwrite(uint8(Avg16), 'noisy_images/Noisy_images_avg16.png'); 