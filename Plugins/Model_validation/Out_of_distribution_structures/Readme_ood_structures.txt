Readme for out-of-distribution structures: 
https://docs.google.com/spreadsheets/d/1aGkMKP9XdRKXuepxzLW_PhzDs6tq1dFQ/edit#gid=213636369


Since the file size is more than 100MB, most of the out-of-distribution structures are placed in the CurateND (University of Notre Dame) server. 
Curate ND link: https://curate.nd.edu/show/und:5h73pv66h5f

Datasets: Name, Reference paper, Github link, description of the paper and dataset 
1. CARE dataset [link: https://www.nature.com/articles/s41592-018-0216-7 , https://github.com/CSBDeep/CSBDeep] --> supervised training method (need input/noisy and target/clean image pairs)
2. Noise2Void dataset [Link: https://arxiv.org/abs/1811.10980 ,  https://github.com/juglab/n2v] --> self-supervised training method (need only noisy inputs)
3. Candle-J dataset [Link: https://github.com/haiderriazkhan/CANDLE-J in MATLAB, https://github.com/haiderriazkhan/CANDLE-J/blob/master/Sample%20Images/NoisyImage.tif] (only noisy dataset is available)
4. GigaDB dataset [Link: https://academic.oup.com/gigascience/article/10/5/giab032/6269106 , http://gigadb.org/dataset/100888] (low SNR and high SNR images are available with different photon counts) (BPAE samples of 3 different channels (mitochondria, F-actin, nucleus as multiple channels), membrane images)
5. 3D RCAN dataset: [Link: https://www.nature.com/articles/s41592-021-01155-x.pdf , https://zenodo.org/record/4624364#.YeLvx1jMJmB] --> low SNR, diffraction limited as input and high SNR with deconvolution as target (RCAN model provides the joint image denoising and super-resolution) (supervised training method)
6. ACsN dataset: [Link: https://www.nature.com/articles/s41467-019-13841-8.pdf, https://github.com/ShuJiaLab/ACsN] --> for fixed pattern noise (self-supervised image denoising to remove the fixed pattern noise) (samples: Microtubules of HeLa cells, Mitochondria in live human embryonic kidney (HEK) cells, GFP-stained calcein in live Adipocytes (lipocytes), Fluorescently labeled adult brine shrimp, Live human lung cancer cells )
7. CBMI dataset: [Link: https://www.nature.com/articles/s41592-021-01285-2.pdf , https://zenodo.org/record/5212734#.YeLus1jMJmA] --> Removing independent noise in systems neuroscience data using DeepInterpolation (segmentation images (images, masks are available))
8. Convallaria dataset: [Link: https://arxiv.org/pdf/1906.00651.pdf , https://zenodo.org/record/5156913#.Ydb9QRPMJmB] --> PN2V dataset (self-supervised training method)
9. Online Dataset:  [Link: https://www.nature.com/articles/s41592-021-01167-7.pdf, https://github.com/qnano/simnoise] Structured illumination microscopy with noise-controlled image reconstructions (only noisy images are available: "nano test structures, Fixed cells, GFP cells")

Fluoresence microscopy sample details are explained in the reference paper/github links. Microscopy details are also provided in the given reference papers. 
Quantitative metrics such as PSNR and SSIM are calculated using the CARE paper method [https://www.nature.com/articles/s41592-018-0216-7] using percentiles for normalization and compute the PSNR and SSIM values. 


Copyright
© 2019 Varun Mannam, University of Notre Dame
License
Licensed under the GPL
