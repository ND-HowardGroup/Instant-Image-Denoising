clc 
close all
clear variables

%date: 6th October 2021
%VM: test contrast of the clean bacteria dataset image 

font = 14;
format long;


target1 =  imread('AVG_2mw_OD=0_Myxo_1mM Ferri in 240mM KNO3_CV_9.tif'); 
target1 = double(target1);

min2 = min(target1(:));
max2 = max(target1(:));

contrast_target1 = max2-min2;

target1 = target1/65535; 
[min1,max1] = calculate_hist_values(target1, 65536);

contrast_target2 = max1-min1;