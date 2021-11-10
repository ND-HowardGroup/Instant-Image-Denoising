clc 
close all
clear variables

%date: 5th October 2021
%VM: test contrast of the W2S dataset image 

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


addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/w2s_dataset/w2s_raw_data/');
addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/w2s_dataset/w2s_gt_data/');
Num_imgs = 360;
contrast_raw = zeros(Num_imgs,1);
contrast_gt = zeros(Num_imgs,1);

pwd1 = '/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/w2s_dataset/w2s_raw_data/';
pwd2 = '/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/w2s_dataset/w2s_gt_data/';
files1 = dir(fullfile(pwd1, '*.tif'));
files2 = dir(fullfile(pwd2, '*.tif'));

min_int_input = zeros(Num_imgs,1);
max_int_input = zeros(Num_imgs,1);
min_int_target = zeros(Num_imgs,1);
max_int_target = zeros(Num_imgs,1);
for i = 1:Num_imgs
    input1 =  imread(files1(i).name);
    input1 = double(input1);
       
    min1 = min(input1(:));
    max1 = max(input1(:));
    
    min_int_input(i) = min1;
    max_int_input(i) = max1;
    
    contrast_input = max1-min1;
    contrast_raw(i) = contrast_input;
end

for i = 1:Num_imgs
    
    target1 = imread(files2(i).name);
    target1 = double(target1);
    
    min2 = min(target1(:));
    max2 = max(target1(:));
    
    min_int_target(i) = min2;
    max_int_target(i) = max2;
    contrast_target = max2-min2;
    contrast_gt(i) = contrast_target;
end

