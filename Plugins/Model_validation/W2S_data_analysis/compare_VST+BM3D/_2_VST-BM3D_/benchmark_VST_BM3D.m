clc
clear
close all

data_name = 'W2S';

addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Sep21/1409/check_BM3D_W2S/_2_VST-BM3D_/code');
dir_avg1 = '/Users/varunmannam/Desktop/Fall21/Research_Fall21/Sep21/1409/check_BM3D_W2S/n2/'; % 360 images images_avg1_8bit
dir_gt = '/Users/varunmannam/Desktop/Fall21/Research_Fall21/Sep21/1409/check_BM3D_W2S/images_avg400_8bit/'; % 360 image


[avg1_array] = import_img_array(dir_avg1);
[img_gt] = import_img_array(dir_gt);

%%%%%%%%%%%%%%%%%%%%%%%%%%% noise realizations %%%%%%%%%%%%%%%%%%%%%%%%%%%
n_repeat = 360; 
% n_repeat = 1;

diary([data_name, '.log'])

PSNR_avg1 = zeros(n_repeat, 1);
SSIM_avg1 = zeros(n_repeat, 1);
TIME_avg = zeros(n_repeat, 1);



for i_repeat = 1:1
    
    fprintf('\n ----- The %d / %d realization -----\n', i_repeat,n_repeat)
    
    % extract the images and average them
    img_avg1 = avg1_array(:,:,i_repeat);
       
    
   
    % perform the denoising algorith
    [denoise_avg1, time_avg1] = denoise_VST_BM3D(img_avg1);
    
    TIME_avg(i_repeat) = time_avg1;
    
    % compute the PSNR
    PSNR_avg1(i_repeat) = 10*log10(1/mean((img_gt(:)-denoise_avg1(:)).^2));
        
    % compute the SSIM
    SSIM_avg1(i_repeat) = ssim(denoise_avg1, img_gt);
end

fprintf('Average = 1 (PSNR: %.2f dB, SSIM: %.4f)\n', mean(PSNR_avg1), mean(SSIM_avg1))
fprintf('Average time for each denoising operation = %.2f\n', mean(TIME_avg))


% figure;
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 0.8, 1]);
% diary off













