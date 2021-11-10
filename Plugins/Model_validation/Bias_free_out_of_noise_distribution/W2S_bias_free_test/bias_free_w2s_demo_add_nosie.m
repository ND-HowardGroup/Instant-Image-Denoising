clc 
close all
clear variables

%date: 21st Sep 2021
%VM withg real image of image_10_001

font = 14;
format long; 


% load image
I = imread('target_avg400_010_0.tif');	
I = double(I); 
I = I/255;

% add noise
noiseSD1 = 0.01; 
noiseI1 = I + noiseSD1*randn(size(I));
imwrite(uint8(noiseI1*255),'w2s_noisy_image_Std_40dB.png');

noiseSD2 = 0.0178;
noiseI2 = I + noiseSD2*randn(size(I));
imwrite(uint8(noiseI2*255),'w2s_noisy_image_Std_35dB.png');

noiseSD3 = 0.0316;
noiseI3 = I + noiseSD3*randn(size(I));
imwrite(uint8(noiseI3*255),'w2s_noisy_image_Std_30dB.png');

noiseSD4 = 0.0562;
noiseI4 = I + noiseSD4*randn(size(I));
imwrite(uint8(noiseI4*255),'w2s_noisy_image_Std_25dB.png');

noiseSD5 = 0.1;
noiseI5 = I + noiseSD5*randn(size(I));
imwrite(uint8(noiseI5*255),'w2s_noisy_image_Std_20dB.png');

noiseSD6 = 0.1778;
noiseI6 = I + noiseSD6*randn(size(I));
imwrite(uint8(noiseI6*255),'w2s_noisy_image_Std_15dB.png');

noiseSD7 = 0.316;
noiseI7 = I + noiseSD7*randn(size(I));
imwrite(uint8(noiseI7*255),'w2s_noisy_image_Std_10dB.png');

noiseSD8 = 0.562;
noiseI8 = I + noiseSD8*randn(size(I));
imwrite(uint8(noiseI8*255),'w2s_noisy_image_Std_5dB.png');

noiseSD01 = 0.0056;
noiseI01 = I + noiseSD01*randn(size(I));
imwrite(uint8(noiseI01*255),'w2s_noisy_image_Std_45dB.png');

noiseSD02 = 0.00316;
noiseI02 = I + noiseSD02*randn(size(I));
imwrite(uint8(noiseI02*255),'w2s_noisy_image_Std_50dB.png');

% % output result
% figure(1);
% imshow(I); title('Original');
% figure(2);
% imshow(noiseI); title('Corrupted Image');
% fprintf('PSNR is:%f\n',20*log10(1/std2(noiseI-I)));


