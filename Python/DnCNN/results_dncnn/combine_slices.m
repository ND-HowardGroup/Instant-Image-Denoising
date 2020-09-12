function result = combine_slices(input)
format long
font = 14;
im_size = 512;
imsize = 256;
result = zeros(im_size,im_size);
slices = 4;
for j=1:slices
    if j==1
        result(1:imsize,1:imsize) = input(j,:,:);
    end
    if j==2
        result(1:imsize,imsize+1:im_size) = input(j,:,:);
    end
    if j==3
        result(imsize+1:im_size,1:imsize) = input(j,:,:);
    end
    if j==4
        result(imsize+1:im_size,imsize+1:im_size) = input(j,:,:);
    end
end

end