function results=calculate_PSNR_new(tar1,ip1)
%psnr
size1 = size(ip1); %input size is here instead of 512x512
smax = 255;
ip11 = double(ip1);
tar11 = double(tar1);
mse_op = power((ip11-tar11),2);
mse2_op= sum(mse_op(:))/(size1(1)*size1(2));
snr_op = smax*smax/mse2_op;
psnr_op = 10*log10(snr_op);
%disp('psnr_ip');
%display(psnr_op);
results = [psnr_op];
end