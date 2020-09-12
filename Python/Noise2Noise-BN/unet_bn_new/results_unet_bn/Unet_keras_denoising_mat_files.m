clc 
close all
clear variables

%date: 25th May 2020
%VM

font = 14;
linewidth = 2;
format long;
addpath('/Users/varunmannam/Desktop/Spring20/Research_S20/May20/2505/dncnn_keras/dataset/');
est_lr1 = load('Estimated_result_c1_lr1.mat');
est_lr1 = est_lr1.Estimated_result_c1;
est_lr1 = double(est_lr1);
est_lr1 = (est_lr1+0.5)*255;

N_images = 48;
N_slices = 48*4;
imsize = 512;
im_size = 256;
slices = 4;

denoising_images = zeros(N_images, imsize, imsize);
for i=1:N_images
    %for j=1:slices
    denoising_images(i,1:im_size,1:im_size) = est_lr1((i-1)*4+1,:,:);
    denoising_images(i,1:im_size,im_size+1:imsize) = est_lr1((i-1)*4+2,:,:);
    denoising_images(i,im_size+1:imsize,1:im_size) = est_lr1((i-1)*4+3,:,:);
    denoising_images(i,im_size+1:imsize,im_size+1:imsize) = est_lr1((i-1)*4+4,:,:);
    %end
end
