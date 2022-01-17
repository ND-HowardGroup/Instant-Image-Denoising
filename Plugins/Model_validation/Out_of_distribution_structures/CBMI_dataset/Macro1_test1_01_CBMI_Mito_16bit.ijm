input_path = getDirectory("image"); //current image location
print("\\Clear"); //clear log window
print("input_path: ",input_path);
filelist = getFileList(input_path); //file list
for (i=0;i<filelist.length;i++)
{
	print(filelist[i]); //present file name
	run("Close All"); //close all open files/results
	run("Clear Results"); //clear all results
	open(input_path + filelist[i]); //open ith image 
	//run("Enhance Contrast", "saturated=0.35 normalized"); //reduce max value and set between 0-1
	//setOption("ScaleConversions", true);
	//run("8-bit"); //to 8bit 0-255
	//run("Apply LUT");
	
	//saveAs("Tiff","C:\\Users\\David\\Downloads\\W2S_dataset_denoising\\images_avg1_8bit\\input_"+filelist[i]); //save the denoised image
	run("Noise2Noise Denoising"); // run noise2noise model
	setOption("ScaleConversions", false);
	run("16-bit");
	saveAs("Tiff","C:\\Users\\David\\Downloads\\image_denoising_validation_PC2\\Revision_R2\\CBMI\\CBMI_Mito\\denoised_mito\\denoised_"+filelist[i]); //save the denoised image
	run("Close All"); //close input and denoised images
	wait(2000); //wai 2sec (2000 ms)
}
