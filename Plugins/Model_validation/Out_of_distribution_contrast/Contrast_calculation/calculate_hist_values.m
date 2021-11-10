function [ilow, ihigh] = calculate_hist_values(img, nbins)
tol_low = 0.01;
tol_high = 0.99;
N = imhist(img,nbins);
cdf = cumsum(N)/sum(N); %cumulative distribution function
ilow = find(cdf > tol_low, 1, 'first');
ihigh = find(cdf >= tol_high, 1, 'first');

end
