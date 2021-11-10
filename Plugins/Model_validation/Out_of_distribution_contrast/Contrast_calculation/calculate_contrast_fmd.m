clc 
close all
clear variables

%date: 5th October 2021
%VM: test contrast of the FMD dataset image 

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

l1={'TwoPhoton_BPAE_R_2.png','Confocal_BPAE_R_4.png','TwoPhoton_BPAE_R_3.png','TwoPhoton_BPAE_R_1.png','Confocal_FISH_4.png','WideField_BPAE_B_4.png','TwoPhoton_BPAE_R_4.png','Confocal_BPAE_R_2.png','Confocal_BPAE_R_3.png','Confocal_FISH_1.png','WideField_BPAE_B_1.png','Confocal_FISH_3.png','WideField_BPAE_B_3.png','Confocal_BPAE_R_1.png','WideField_BPAE_B_2.png','Confocal_FISH_2.png','WideField_BPAE_G_1.png','TwoPhoton_MICE_2.png','TwoPhoton_MICE_3.png','WideField_BPAE_G_2.png','TwoPhoton_MICE_1.png','WideField_BPAE_G_3.png','TwoPhoton_MICE_4.png','WideField_BPAE_G_4.png','TwoPhoton_BPAE_B_3.png','Confocal_BPAE_B_4.png','TwoPhoton_BPAE_B_2.png','WideField_BPAE_R_4.png','TwoPhoton_BPAE_B_1.png','WideField_BPAE_R_1.png','Confocal_BPAE_B_3.png','Confocal_BPAE_B_2.png','TwoPhoton_BPAE_B_4.png','WideField_BPAE_R_2.png','Confocal_BPAE_B_1.png','WideField_BPAE_R_3.png','TwoPhoton_BPAE_G_4.png','Confocal_MICE_4.png','Confocal_BPAE_G_2.png','Confocal_BPAE_G_3.png','Confocal_BPAE_G_1.png','TwoPhoton_BPAE_G_2.png','Confocal_MICE_2.png','Confocal_BPAE_G_4.png','Confocal_MICE_3.png','TwoPhoton_BPAE_G_3.png','TwoPhoton_BPAE_G_1.png','Confocal_MICE_1.png'};
l1 = l1';

[l2, index] = sort(l1);

addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/test_mix');
addpath('/Users/varunmannam/Desktop/Fall21/Research_Fall21/Oct21/0510/contrast_calculations/test_mix');
Num_imgs = 48;
contrast_raw = zeros(Num_imgs,1);
contrast_gt = zeros(Num_imgs,1);
min_int_input = zeros(Num_imgs,1);
max_int_input = zeros(Num_imgs,1);
min_int_target = zeros(Num_imgs,1);
max_int_target = zeros(Num_imgs,1);

for i = 1:Num_imgs
    input1 =  imread(strcat(pwd, '/test_mix/raw/', cell2mat(l2(i))));
    input1 = double(input1);
    target1 = imread(strcat(pwd, '/test_mix/gt/', cell2mat(l2(i))));
    target1 = double(target1);
    
    min1 = min(input1(:));
    max1 = max(input1(:));
    
    min2 = min(target1(:));
    max2 = max(target1(:));
    
    contrast_input = max1-min1;
    contrast_target = max2-min2;
    
    min_int_input(i) = min1;
    max_int_input(i) = max1;
    min_int_target(i) = min2;
    max_int_target(i) = max2;
    
    contrast_raw(i) = contrast_input;
    contrast_gt(i) = contrast_target;
end

