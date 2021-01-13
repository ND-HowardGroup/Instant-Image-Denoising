/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 Board of Regents of the University of Notre Dame (Electrical Engineering).
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package net.image.denoising.demo;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.image.denoising.GraphBuilder;
import net.image.denoising.TensorFlowService;
import net.image.denoising.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.http.HTTPLocation;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;


// added by Varun Mannam
import java.nio.FloatBuffer;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import org.tensorflow.DataType;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.util.Intervals;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;

/**
 * Command to use an Image denoising model to Denoise Poisson + Gaussian noise with in an image.
 * Author: Varun Mannam
 * Department: Department of Electrical Engineering
 * University: University of Notre Dame
 * South Bend, Indiana, USA, Zip: 46656
 */
@Plugin(type = Command.class, menuPath = "Plugins> Noise2Noise Denoising",headless = true)
public class Noise2Noise_Denoising implements Command 
{

	private static final String DENOISED_MODEL_URL = "https://storage.googleapis.com/noise2noise_ml_model_vm_01/Noise2Noise_ML_Model_VM_01.zip";

	private static final String MODEL_NAME = "Noise2Noise_ML_Model_VM_01";

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private static LogService log;

	@Parameter
	private Dataset inputImage;
        
    @Parameter(type = ItemIO.OUTPUT)
	private Img<FloatType> outputImage_Noise2Noise_Denoised;
	
	private static float max_im;
	public static String image_type; // image type: 8-bit uint, 16-bit uint, 32-bit float	
	public static long[] dims; // this is the vector	
	public static int ndims;  // n-dimensions

	@Override
	public void run() 
	{

		tensorFlowService.loadLibrary();
		if(!tensorFlowService.getStatus().isLoaded()) return;
		log.info("Version of TensorFlow: " + tensorFlowService.getTensorFlowVersion());

		try 
                {
                        
                        final HTTPLocation source = new HTTPLocation(DENOISED_MODEL_URL);
                        final Graph graph = tensorFlowService.loadGraph(source, MODEL_NAME,"Noise2Noise_ML_Model_VM_01.pb");
                        
                        try 
                        (
                                final Tensor<Float> inputTensor = loadFromImgLib(inputImage);
                        )

                        {
                                final long startMs = System.currentTimeMillis();
                                //log.info("max_value is " + max_im);
                                final Tensor<Float> outputTensor = process_image(inputTensor, graph, max_im);
                                final long endMs = System.currentTimeMillis();
                                log.info("Model inference time is: " + (endMs - startMs) + "ms, TensorFlow version: " + tensorFlowService.getTensorFlowVersion());
                                outputImage_Noise2Noise_Denoised = Tensors.imgFloat(outputTensor, new int[]{ 2, 1, 0, 3 });        
                        }
		}
		catch (final Exception exc) 
                {
			log.error(exc);
		}
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private static Tensor<Float> loadFromImgLib(final Dataset d) throws IOException
        {
			
			log.info("Input image type is " + d.getTypeLabelShort()+ "."); //added 1306 to find the type of image d.getType() gives the first element, like 3, 771, 3.0 for 8-bit,16-bit ad 32-bit images	
					image_type  = d.getTypeLabelShort();
		final RandomAccessibleInterval image = (RandomAccessibleInterval) d.getImgPlus(); //get image in randomaccessibleinterval
                final int ndims = image.numDimensions(); //check dimensions
		if (ndims == 1 || ndims == 4) 
                {
			dims = new long[ndims]; //convert to array final long[] dims = new long[ndims];
			image.dimensions(dims);
			throw new IOException("Can only process 2D/3D images, not an image with " + ndims + " dimensions (" + Arrays.toString(dims) + ")");
                }
                return loadFromImgLib(image); //call function to convert to tensor
	}

	private static <T extends RealType<T>> Tensor<Float> loadFromImgLib(final RandomAccessibleInterval<T> image)
	{
		// NB: Assumes XYC ordering. TensorFlow wants YXC.
		RealFloatConverter<T> converter = new RealFloatConverter<>();       
                RandomAccessibleInterval<FloatType> ix1 = Converters.convert(image,converter,new FloatType());
                
                max_im = Tensors.max_image(ix1); //maximum of input image (255 or 65535 for 8-bit or 16-bit)
                //log.info("max_value is " + max_im);
                final long dims = ix1.numDimensions();
                if (dims==2)
                {
                    ix1 = Views.addDimension(ix1, 0, 0);
                } //added extra dimension
                if (dims==3)
                {
                    ix1 = ix1; //no additional view to add extra dims
                }
                if (dims>3)
                {
                	log.info("Can only process 2D/3D images, but not an image with " + dims + " dimensions");
                }
                return Tensors.tensorFloat(ix1, new int[]{ 2, 1, 0 });
	}

	// -----------------------------------------------------------------------------------------------------------------
	// All the code below was essentially copied verbatim from:
	// https://github.com/tensorflow/tensorflow/blob/e8f2aad0c0502fde74fc629f5b13f04d5d206700/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
	// -----------------------------------------------------------------------------------------------------------------
	private static Tensor<Float> normalizeImage(final Tensor<Float> t, final float max_im) 
        {
		try (Graph g = new Graph()) 
                {
			final long[] x1shape = t.shape();
			
                        final GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			//   float using (value - Mean)/Scale.
                        final int rem1 = (int) x1shape[1] % 32;
                        final int H;
                        final int W;
                        if (rem1==0)
                        {
                           H = (int) x1shape[1];
                        }
                        else 
                        {
                            H = (((int) x1shape[1]/32)+1)*32; //added extra size of resize image 
                        }
			//final int H = (int) x1shape[1];  
                        final int rem2 = (int) x1shape[2] % 32;
                        if (rem2==0)
                        {
                            W = (int) x1shape[2];
                        }
                        else 
                        {
                            W = (((int) x1shape[2]/32)+1)*32; //added extra size of resize image 
                        }
			//final int W = (int) x1shape[2];           
			final float mean = (max_im/2); //127.5f;
			final float scale = max_im; //255f;

			// Since the graph is being constructed once per execution here, we can
			// use a constant for the input image. If the graph were to be re-used for
			// multiple input images, a placeholder would have been more appropriate.
			final Output<Float> img = g.opBuilder("Const", "img").setAttr("dtype", t.dataType()).setAttr("value", t).build().output(0);       
			final Output<Float> output = b.div(b.sub(b.resizeBilinear(b.expandDims(//
				img, //
				b.constant("make_batch", 3)), //
				b.constant("size", new int[] { H, W })), //
				b.constant("mean", mean)), //
				b.constant("scale", scale));
			
			try (Session s = new Session(g)) 
                        {
				@SuppressWarnings("unchecked")
				Tensor<Float> result = (Tensor<Float>) s.runner().fetch(output.op().name()).run().get(0);
				return result;
			}
		}
	}
        
        private static Tensor<Float> process_image(final Tensor<Float> tin, final Graph graph, final float max_im) //, final int dims)
        {
            final long[] dims_tensor = tin.shape();
            final Tensor<Float> image = normalizeImage(tin, max_im);
            final long startMs1 = System.currentTimeMillis();
            final Tensor<Float> outputImage1 = executeNoise2NoiseDenoisingGraph(graph, image);
            final long endMs1 = System.currentTimeMillis();
            //log.info("Graph inference time is: " + (endMs1 - startMs1) + "ms");
            final Tensor<Float> post_img = post_process(outputImage1, dims_tensor, max_im);
            return post_img;
        }
               
        private static Tensor<Float> post_process(final Tensor<Float> top, final long[] dims_tensor, final float max_im) 
        {
                try (Graph g = new Graph()) 
                {
                        final long[] x2shape = dims_tensor; //top.shape();
                        final GraphBuilder b = new GraphBuilder(g);
                        // Some constants specific to the pre-trained model at:
                        // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
                        //
                        // - The model was trained with images scaled to 224x224 pixels.
                        // - The colors, represented as R, G, B in 1-byte each were converted to
                        //   float using (value - Mean)/Scale.
                        final int H_post = (int) x2shape[1];
                        final int W_post = (int) x2shape[2];
                        final float mean = -0.5f; // -0.5f;
                        final float scale = (1/max_im); //0.0039215f;

                        // Since the graph is being constructed once per execution here, we can
                        // use a constant for the input image. If the graph were to be re-used for
                        // multiple input images, a placeholder would have been more appropriate.
                        final Output<Float> img = g.opBuilder("Const", "img").setAttr("dtype", top.dataType()).setAttr("value", top).build().output(0);       
                        final Output<Float> output = b.div(b.sub(b.resizeBilinear(
                                img,//
                                b.constant("size", new int[] { H_post, W_post })),//
                                b.constant("mean", mean)), //
                                b.constant("scale", scale));
                        
                        try (Session s = new Session(g)) 
                        {
                                @SuppressWarnings("unchecked")
                                Tensor<Float> result = (Tensor<Float>) s.runner().fetch(output.op().name()).run().get(0);
                                return result;
                        }
                }
        }

        private static final Tensor<Float> executeNoise2NoiseDenoisingGraph(final Graph g, final Tensor<Float> image)
        {
                final Session s = new Session(g);
                @SuppressWarnings("unchecked")
                final Tensor<Float> result = (Tensor<Float>) s.runner().feed("img", image).fetch("activation/Tanh").run().get(0);
                return result; 
        }

        public static void main(String[] args) throws IOException 
        {
                final ImageJ ij = new ImageJ();
                ij.launch(args);

                // Open an image and display it.
                final String imagePath = "/Users/varunmannam/Desktop/images_to_code/gray_image.png";
                final Object dataset = ij.io().open(imagePath);//ij.scifio().datasetIO().Open(imagePath); //ij.io().open(imagePath);
                //Dataset image = ij.scifio().datasetIO().Open(imagePath);
                ij.ui().show(dataset);
                
                // Launch the "output denoising Images" command with some sensible defaults.
                ij.command().run(Noise2Noise_Denoising.class, true, "inputImage", dataset);
        }
}