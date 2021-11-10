clc 
close all
clear variables

%date: 5th October 2021
%VM: test contrast of the bacteria dataset imagess

font = 14;
format long;
% 
% input1 =  imread('image_R_1_001.png'); 
% input1 = double(input1);
% 
% min1 = min(input1(:));
% max1 = max(input1(:));
% 
% 
% target1 =  imread('image_R_1_avg50.png'); 
% target1 = double(target1);
% 
% min2 = min(target1(:));
% max2 = max(target1(:));
% 
% contrast_input = max1-min1;
% contrast_target = max2-min2;


addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/contrast_bacteria_images/');

Num_imgs = 600;
contrast_raw = zeros(Num_imgs,1);

pwd1 = '/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/contrast_bacteria_images/';
files1 = dir(fullfile(pwd1, '*.tif'));
min_int_input = zeros(Num_imgs,1);
max_int_input = zeros(Num_imgs,1);

for i = 1:Num_imgs
    input1 =  imread(files1(i).name);
    input1 = double(input1);
    input1 = input1/65535;   
    
    [min1,max1] = calculate_hist_values(input1, 65536);
    min_int_input(i) = min1;
    max_int_input(i) = max1;
    
    contrast_input = max1-min1;
    contrast_raw(i) = contrast_input;
end

