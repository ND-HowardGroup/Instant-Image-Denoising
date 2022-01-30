# Instant-Image-Denoising
This repository includes the code used for the image denoising of our Noise2Noise and DnCNN ML models trained on the FMD dataset microscopy images. This repository also includes the ImageJ plugin (contains pre-trained ML model and computationally much faster with GPU configuration) for image denoising on the 2D and 3D data. 

#Images: The test images can be downloaded from here https://curate.nd.edu/show/f4752f78z6t

#Citation for dataset: Please cite the FMD dataset using the following format: Mannam, Varun, Yide Zhang, and Scott Howard. “Fluorescence Microscopy Denoising (FMD) Dataset.” Notre Dame, April 21, 2019. https://doi.org/10.7274/r0-ed2r-4052. #DOI: 10.7274/r0-ed2r-4052

# Results shown in the Journal paper

Input Noisy Image          | Noise2Noise Plugin        | Target Image 		         |	
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Test_images/2D_images/Raw_Images/Confocal_FISH_3.png" width="200" height="200" />   |  <img src="Plugins/Test_images/2D_images/Image_Denoising_results/denoised_confocal_fish3.png" width="200" height="200" />| <img src="Plugins/Test_images/2D_images/Target(ground_truth)_Images/gt_Confocal_FISH_3.png" width="200" height="200" /> |


Details: 
Input: 2D single channel image from FMD dataset: Confocal_FISH_3.png (from a confocal microscopy and sample: Zebrafish)
Denoised: Image denoising using our ImageJ plugin (from trained Noise2Noise ML model): (time: 80 ms in GPU, image size: 512x512)
Target: Target image generated by taking average of 50 noisy images in the same FOV: 


# Comparison of Noise2Void image denoising method: (test image from the W2S dataset)

Input Noisy Image          | Noise2Void           	   | Noise2Noise Plugin        | Target Image 		         |	
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/N2V_Comparison/W2S_dataset/W2S_noisy_input_avg1_010_0.png" width="175" height="175" />  | <img src="Plugins/Model_validation/N2V_Comparison/W2S_dataset/W2S_denosied_Noise2Void_010_0.png" width="175" height="175" /> | <img src="Plugins/Model_validation/N2V_Comparison/W2S_dataset/W2S_denosied_Noise2Noise(Ours)_010_0.png" width="175" height="175" />  | <img src="Plugins/Model_validation/N2V_Comparison/W2S_dataset/W2S_target_avg400_010_0.png" width="175" height="175" />  | 
PSNR: 17.93 dB			       | PSNR: 22.29 dB			       | PSNR: 25.44 dB	           | 


# Results on the out-of-distribution structures using the Noise2Noise plugin image denoising method: 

Input Noisy Image          | Noise2Noise Plugin        |       	  
:-------------------------:|:-------------------------:|
:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/N2V_dataset/N2v/FA - 0488 Evolve.png" width="225" height="225" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/N2V_dataset/N2v/denoised_FA - 0488 Evolve.png" width="225" height="225" /> |
:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/PN2V_sample_images/20190520_tl_25um_50msec_05pc_488_130EM_Conv0000_crop.png" width="225" height="225" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/PN2V_sample_images/denoised_20190520_tl_25um_50msec_05pc_488_130EM_Conv0000_crop.png" width="225" height="225" /> |
:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/CBMI_dataset/Selected_CBMI_images/low_snr_mito_img7.png" width="225" height="225" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/CBMI_dataset/Selected_CBMI_images/denoised_mito_img7.png" width="225" height="225" /> |
:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/CBMI_dataset/Selected_CBMI_images/low_snr_er_img13.png" width="225" height="225" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/CBMI_dataset/Selected_CBMI_images/denoised_er_img13.png" width="225" height="225" /> |
:-------------------------:|:-------------------------:|
:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/Online_dataset/gain50_1_z0_t0_r0_h0.png" width="225" height="225" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/Online_dataset/gain50_1_z0_t0_r0_h0_denoised.png" width="225" height="225" /> |
:-------------------------:|:-------------------------:|

Details: 
Input: 2D single channel image from CARE dataset, PN2V Convallaria dataset, CBMI dataset (Mito, ER images), other online datasets.

# Results on the out-of-distribution structures from the 3D RCAN dataset (includes Actin, ER, Golgi, Lysosome, Matrix-mitochondria, Microtubules and Tomm20-mitochondria samples) using the Noise2Noise plugin image denoising results: 
Input Noisy Image          | Noise2Noise Plugin        | High SNR (ground truth)   |       	  
:-------------------------:|:-------------------------:|:-------------------------:|
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Actin_crop2/noisy_17_decon0027_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Actin_crop2/denoised_17_decon0027_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Actin_crop2/17_decon0027_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/ER_crop2/noisy_image000001_decon0007_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/ER_crop2/denoised_image000001_decon0007_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/ER_crop2/image000001_decon0007_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Golgi_crop2/noisy_01_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Golgi_crop2/denoised_01_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Golgi_crop2/01_decon0005_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Lysosome_crop2/noisy_80_image000003_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Lysosome_crop2/denoised_80_image000003_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Lysosome_crop2/80_image000003_decon0005_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Matrix_mito_crop2/noisy_C1-image000011_decon0007_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Matrix_mito_crop2/denoised_C1-image000011_decon0007_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Matrix_mito_crop2/C1-image000011_decon0007_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Microtubules_crop2/noisy_image000002gt_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Microtubules_crop2/denoised_image000002gt_decon0005_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Microtubules_crop2/image000002gt_decon0005_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Tomm20_mito_crop2/noisy_80_image000027_decon0011_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Tomm20_mito_crop2/denoised_80_image000027_decon0011_crop.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/3D_RCAN_dataset_sample_images/small_fov_images/Tomm20_mito_crop2/80_image000027_decon0011_crop.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|



Details: 
Input: 2D single channel image from 3D RCAN dataset includes Actin, ER, Golgi, Lysosome, Matrix-mitochondria, Microtubules and Tomm20-mitochondria samples.

# Results on the GigaDB dataset (of BPAE samples, membrane structures) using the Noise2Noise plugin image denoising method: 
Input Noisy Image          | Noise2Noise Plugin        | High SNR (ground truth)   |       	  
:-------------------------:|:-------------------------:|:-------------------------:|
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/nucleus-lowsnr-sample.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/nucleus-lowsnr-sample_denoised.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/nucleus-highsnr-sample.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/membrane-lowsnr-sample.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/membrane-lowsnr-sample_denoised.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/membrane-highsnr-sample.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-confocal-lowsnr-sample.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-confocal-lowsnr-sample_denoised.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-confocal-highsnr-sample.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-confocal-lowsnr22.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-confocal-lowsnr_denoised22.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-confocal-highsnr-sample22.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-20x-noise1-lowsnr-sample.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-20x-noise1-lowsnr-sample_denoised.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-20x-noise1-highsnr-sample.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-20x-noise1-lowsnr-sample.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-20x-noise1-lowsnr-sample_denoised.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-20x-noise1-highsnr-sample.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-60x-noise1-lowsnr-sample1.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-60x-noise1-lowsnr-sample_denosied1.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/mito-60x-noise1-highsnr-sample1.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|
<img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-60x-noise1-lowsnr-sample1.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-60x-noise1-lowsnr-sample_denoised1.png" width="200" height="200" /> | <img src="Plugins/Model_validation/Out_of_distribution_structures/GigadB_dataset_sample_images/cropped_images/actin-60x-noise1-highsnr-sample1.png" width="200" height="200" /> |
:-------------------------:|:-------------------------:|:-------------------------:|

Details: 
Input: 2D single channel image from GigaDB dataset (Nucleus, membrane, Mitochondria (confocal), Actin (WF), Mitochondria (WF) using 2 different objectives (20x and 60x) respectively ).

# Results of Noise2Noise plugin for datasets: 
The image denoising dataset size is large to upload to Github that includes the noisy, denoised images are provided in the CurateND folder by University of Notre Dame. [Link](https://curate.nd.edu/show/5h73pv66h5f) 


## **Frequently Asked Questions (FAQs)**
Some of the faq like image conversions, 2D, 3D image denoising, denoising images in a folder using macros, etc..
https://github.com/ND-HowardGroup/Instant-Image-Denoising/blob/master/FAQ_Instant_Image_Denoising.docx


## **Copyright**

© 2019 Varun Mannam, University of Notre Dame  

## **License**

Licensed under the [GPL](https://github.com/ND-HowardGroup/Instant_image_denoising/blob/master/LICENSE)

