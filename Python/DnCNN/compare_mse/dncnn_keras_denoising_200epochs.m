clc 
close all
clear variables

%date: 07th June 2020
%VM
%with new excel that has true target for Unet bn model different lr values
%lr1 = 1e-3, lr2 = 1e-4, lr3 = 5e-4, lr4 = 5e-5, lr5 = 1e-5
font = 14;
linewidth = 2;
format long;

lr1 = load('lr1.txt'); %new result with lr=1e-3 and diff LR values and weights initalization
lr2 = load('lr2.txt'); %new result with lr=5e-4 and diff LR values and weights initalization
lr3 = load('lr3.txt'); %new result with lr=1e-4 and diff LR values and weights initalization
lr4 = load('lr4.txt'); %new result with lr=5e-5 and diff LR values and weights initalization

epochs = 200;
x=[1:epochs]';

figure, 
plot(x(1:100),lr1,'b--', 'Linewidth', linewidth);
hold on
plot(x,lr2,'r--', 'Linewidth', linewidth);
plot(x,lr3,'g--', 'Linewidth', linewidth);
plot(x,lr4,'m--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE loss');
title('training loss DnCNN bn Keras');
legend('lr1= 1e-3', 'lr2= 1e-4', 'lr3= 5e-4','lr4= 5e-5', 'Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);


figure, 
plot(x(190:200),lr1(90:100),'b-', 'Linewidth', linewidth);
hold on
plot(x(epochs-10:epochs),lr2(epochs-10:epochs),'r-', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr3(epochs-10:epochs),'g-', 'Linewidth', linewidth);
plot(x(epochs-10:epochs),lr4(epochs-10:epochs),'m--', 'Linewidth', linewidth);
xlabel('Epochs');
ylabel('MSE loss');
title('training loss DnCNN bn Keras');
%legend('lr3= 1e-3', 'lr4= 1e-4 diff LR', 'lr41= 1e-4 diff LR weights', 'Location', 'best');
%legend('lr3= 1e-3', 'lr4= 1e-4 diff LR', 'lr41= 1e-4 diff LR weights','lr51= 1e-3 diff LR weights','lr61= 5e-5 diff LR weights', 'Location', 'best');
legend('lr1= 1e-3', 'lr2= 1e-4', 'lr3= 5e-4','lr4= 5e-5','Location', 'best');
%colormap(gray); colorbar;
set(gca,'FontSize',font);