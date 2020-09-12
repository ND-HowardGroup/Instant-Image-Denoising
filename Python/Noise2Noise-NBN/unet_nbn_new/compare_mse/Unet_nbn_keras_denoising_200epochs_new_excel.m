clc 
close all
clear variables

%date: 10th June 2020
%VM
%with new excel that has true target for Unet bn model different lr values
%lr1 = 1e-3, lr2 = 1e-4, lr3 = 5e-4, lr4 = 5e-5, lr5 = 1e-5
font = 14;
linewidth = 2;
format long;
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/June20/1006/unet_nbn_new/lr1/');
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/June20/1006/unet_nbn_new/lr2/');
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/June20/1006/unet_nbn_new/lr3/');
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/June20/1006/unet_nbn_new/lr4/');
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/June20/1006/unet_nbn_new/lr5/');

lr1 = load('train_loss_unet_denoising_20200610_231313.txt'); %new result with lr=1e-3 and diff LR values and weights initalization
lr2 = load('train_loss_unet_denoising_20200606_060205.txt'); %new result with lr=1e-4 and diff LR values and weights initalization
lr3 = load('train_loss_unet_denoising_20200605_232554.txt'); %new result with lr=5e-4 and diff LR values and weights initalization
lr4 = load('train_loss_unet_denoising_20200603_055706.txt'); %new result with lr=5e-5 and diff LR values and weights initalization
lr5 = load('train_loss_unet_denoising_20200604_020746.txt'); %new result with lr=1e-5 and diff LR values and weights initalization

epochs = 200;
x=[1:epochs]';

figure, 
plot(x,lr1,'b--', 'Linewidth', linewidth);
hold on
plot(x,lr2,'r--', 'Linewidth', linewidth);
plot(x,lr3,'g--', 'Linewidth', linewidth);
plot(x,lr4,'m--', 'Linewidth', linewidth);
plot(x,lr5,'k--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE loss');
title('training loss Unet nbn Keras');
legend('lr1= 1e-3', 'lr2= 1e-4', 'lr3= 5e-4','lr4= 5e-5','lr5= 1e-5', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);


figure, 
plot(x(epochs-10:epochs),lr1(epochs-10:epochs),'b-', 'Linewidth', linewidth);
hold on
plot(x(epochs-10:epochs),lr2(epochs-10:epochs),'r-', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr3(epochs-10:epochs),'g-', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr4(epochs-10:epochs),'m--', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr5(epochs-10:epochs),'k--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE loss');
title('training loss Unet nbn Keras');
%legend('lr3= 1e-3', 'lr4= 1e-4 diff LR', 'lr41= 1e-4 diff LR weights', 'Location', 'best');
%legend('lr3= 1e-3', 'lr4= 1e-4 diff LR', 'lr41= 1e-4 diff LR weights','lr51= 1e-3 diff LR weights','lr61= 5e-5 diff LR weights', 'Location', 'best');
legend('lr1= 1e-3', 'lr2= 1e-4', 'lr3= 5e-4','lr4= 5e-5','lr5= 1e-5', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);