package net.image.denoising.demo;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.command.CommandModule;

import java.util.concurrent.ExecutionException;
//import net.imglib2.img.Img;

import static org.junit.Assert.assertNotNull;
//import org.scijava.ItemIO;
//import org.scijava.plugin.Parameter;

public class Noise2Noise_DenoisingTest 
{
        
	@Test
	public void runLabelImageCommand() throws ExecutionException, InterruptedException 
        {
		final ImageJ ij = new ImageJ();
		Dataset img = ij.dataset().create(new FloatType(), new long[]{256, 256}, "", new AxisType[]{Axes.X, Axes.Y});
		CommandModule module = ij.command().run(Noise2Noise_Denoising.class, false, "inputImage", img).get();
		assertNotNull(module);
		//denoised_outputImage = module.getOutput("denoised_outputImage");
		
	}

}
