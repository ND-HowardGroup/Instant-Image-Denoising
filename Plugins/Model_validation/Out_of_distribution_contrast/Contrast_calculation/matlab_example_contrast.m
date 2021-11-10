pout = imread("pout.tif");
pout_imadjust = imadjust(pout);
pout_histeq = histeq(pout);
pout_adapthisteq = adapthisteq(pout);


montage({pout,pout_imadjust,pout_histeq,pout_adapthisteq},"Size",[1 4])
title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")