close all
clear variables
clc

%Varun Mannam
%date: taken from 29th Sep 2019
%date: 27th July 2020
%DnCNN model with diff lr in keras
%calculate the PSNR on test-mix data raw nbn
format long
font = 14;

%data
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/May20/2505/dncnn_keras/dataset/');
%lr files
%addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/July20/2707/denoising_dncnn_results/lr1_1e-3_100epochs/');
%addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/July20/2707/denoising_dncnn_results/lr2_5e-4_200epochs/');
%addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/July20/2707/denoising_dncnn_results/lr3_1e-4_200epochs/');
addpath('/Users/varunmannam/Desktop/Summer20/Research_SU20/July20/2707/denoising_dncnn_results/lr4_5e-5_200epochs/');



est21 = load('Estimated_result_c1_lr9_weights_epochs.mat');
est21 = est21.Estimated_result_c1_lr9_weights_epochs;
est21 = double(est21);

test_samples = 48;
slices = 4;
imsize = 256;
im_size = 512;
test_index = int8(rand(1)*test_samples);

%input and target
l1={'TwoPhoton_BPAE_R_2.png','Confocal_BPAE_R_4.png','TwoPhoton_BPAE_R_3.png','TwoPhoton_BPAE_R_1.png','Confocal_FISH_4.png','WideField_BPAE_B_4.png','TwoPhoton_BPAE_R_4.png','Confocal_BPAE_R_2.png','Confocal_BPAE_R_3.png','Confocal_FISH_1.png','WideField_BPAE_B_1.png','Confocal_FISH_3.png','WideField_BPAE_B_3.png','Confocal_BPAE_R_1.png','WideField_BPAE_B_2.png','Confocal_FISH_2.png','WideField_BPAE_G_1.png','TwoPhoton_MICE_2.png','TwoPhoton_MICE_3.png','WideField_BPAE_G_2.png','TwoPhoton_MICE_1.png','WideField_BPAE_G_3.png','TwoPhoton_MICE_4.png','WideField_BPAE_G_4.png','TwoPhoton_BPAE_B_3.png','Confocal_BPAE_B_4.png','TwoPhoton_BPAE_B_2.png','WideField_BPAE_R_4.png','TwoPhoton_BPAE_B_1.png','WideField_BPAE_R_1.png','Confocal_BPAE_B_3.png','Confocal_BPAE_B_2.png','TwoPhoton_BPAE_B_4.png','WideField_BPAE_R_2.png','Confocal_BPAE_B_1.png','WideField_BPAE_R_3.png','TwoPhoton_BPAE_G_4.png','Confocal_MICE_4.png','Confocal_BPAE_G_2.png','Confocal_BPAE_G_3.png','Confocal_BPAE_G_1.png','TwoPhoton_BPAE_G_2.png','Confocal_MICE_2.png','Confocal_BPAE_G_4.png','Confocal_MICE_3.png','TwoPhoton_BPAE_G_3.png','TwoPhoton_BPAE_G_1.png','Confocal_MICE_1.png'};
input = zeros(test_samples,im_size,im_size,'uint8');
target = zeros(test_samples,im_size,im_size,'uint8');

est_u21 = zeros(test_samples,im_size,im_size);
est_u21_c = zeros(test_samples,im_size,im_size,'uint8');
path = '/Users/varunmannam/Desktop/Summer20/Research_SU20/May20/2505/dncnn_keras/dataset/test_mix/';
pnsr_results = zeros(test_samples,2);
for i=1:test_samples
    if i<=48 
        str1 = strcat(path,'raw/');
    elseif i>=49 && i<=96
        str1 = strcat(path,'avg2/');
    elseif i>=97 && i<=144
        str1 = strcat(path,'avg4/');
    elseif i>=145 && i<=192
        str1 = strcat(path,'avg8/');
    else
        str1 = strcat(path,'avg16/');
    end
    str11 = strcat(path,'gt/');
    indx1 = mod(i-1,48)+1;
    str2=l1(indx1);
    str3 = cell2mat(strcat(str1,str2));
    str4 = cell2mat(strcat(str11,str2));
    
    ipx1 = imread(str3);
    input(i,:,:) = ipx1;
    tarx1 = imread(str4);
    target(i,:,:) = tarx1;
    
     x00 = est21((i-1)*4+1:i*4,:,:);
     x11=  combine_slices(x00);
     est_u21(i,:,:) = x11;
     %mx11 = min(x11(:));
     %mx12 = max(x11(:));
     %x12 = (x11-mx11)/(mx12-mx11);
     x12 = x11; %no normalization
     estx1 = uint8((x12+0.5)*255); %0-1 range and convert to 0-255 
     est_u21_c(i,:,:) = estx1;
     
     ipx2 = double(ipx1);
     tarx2 = double(tarx1);
     estx2 = double(estx1);
     pnsr_results(i,:)  = calcualte_PSNR(ipx2,tarx2,estx2);
     
end

for i=1:1
   mean_ip(i) = mean(pnsr_results((i-1)*48+1:i*48,1));
   mean_est(i) = mean(pnsr_results((i-1)*48+1:i*48,2));
end
display(mean_ip);
display(mean_est);

% mean_ip = mean(pnsr_results(:,1));
% mean_est = mean(pnsr_results(:,2));
% display(mean_ip);
% display(mean_est);

% %test index
% display('_________________________________________');
% ip1 = input(test_index,:,:);
% ip1 = reshape(ip1,im_size,im_size);
% 
% tar1 = target(test_index,:,:);
% tar1 = reshape(tar1,im_size,im_size);
% 
% est1 = est_u21_c(test_index,:,:);
% est1 = reshape(est1,im_size,im_size);
% 
% %psnr
% smax = 255;
% ip11 = double(ip1);
% tar11 = double(tar1);
% mse_op = power((ip11-tar11),2);
% mse2_op= sum(mse_op(:))/(512*512);
% snr_op = smax*smax/mse2_op;
% psnr_op = 10*log10(snr_op);
% display('psnr_ip');
% display(psnr_op);
% 
% est11 = double(est1);
% mse_op2 = power((est11-tar11),2);
% mse2_op2= sum(mse_op2(:))/(512*512);
% snr_op2 = smax*smax/mse2_op2;
% psnr_op2 = 10*log10(snr_op2);
% display('psnr_est');
% display(psnr_op2);
% 
% figure(1);
% subplot(2,2,1),imagesc(ip1,[min(ip1(:)),max(ip1(:))]);
% title('input')
% colorbar
% set(gca,'FontSize',font)
% %colormap(map)
% subplot(2,2,2),imagesc(tar1,[min(ip1(:)),max(ip1(:))])
% title('clean image')
% colorbar
% set(gca,'FontSize',font)
% subplot(2,2,3),imagesc(est1,[min(ip1(:)),max(ip1(:))])
% title('estimated image')
% colorbar
% set(gca,'FontSize',font)
% diffx1 = est11-tar11;
% subplot(2,2,4),imagesc(diffx1,[min(diffx1(:)),max(diffx1(:))])
% title('difference (est-clean) image')
% colorbar
% set(gca,'FontSize',font)
% 
% figure(2);
% ip_snr = pnsr_results(:,1);
% op_snr = pnsr_results(:,2);
% scatter(ip_snr,op_snr,'b');
% hold on
% xm1 = min(ip_snr(:));
% xm2 = max(ip_snr(:));
% xm = [xm1-0.1*xm1:0.1:xm2+0.1*xm2]';
% plot(xm,xm,'r');
% xlabel('input SNR');
% ylabel('output SNR');
% title('Scatter psnr (input and estimated)')
% legend('nbn2 estimated','y=x line','Location','best')
% set(gca,'FontSize',font)
