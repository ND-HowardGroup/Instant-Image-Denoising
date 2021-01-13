/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 Board of Regents of the University of Notre Dame Electrical Engineering
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

package net.image.denoising;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgView;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

/**
 * Utility class for working with TensorFlow {@link Tensor} objects. In
 * particular, this class provides methods for converting between ImgLib2 data
 * structures and TensorFlow {@link Tensor}s.
 *
 * @author Varun Mannam
 */
public final class Tensors 
{

	private Tensors() 
	{
		// NB: Prevent instantiation of utility class.
	}

	// --------- TENSOR to RAI ---------

	// NB: The following "agnostic" API is somehow bad due to recursive generics.
	// The Img<?> sucks because we don't know it's RealType.
	// But "? extends RealType<?>" is also a problem because ?s don't match.
	// And putting a T param here breaks things downstream: calling code
	// has not T context and so you can assign things to improper types.
//	public static Img<?> img(final Tensor image) {
//		switch (image.dataType()) {
//			case BOOL:
//				return imgBool(image);
//			case DOUBLE:
//				return imgDouble(image);
//			case FLOAT:
//				return imgFloat(image);
//			case INT32:
//				return imgInt(image);
//			case INT64:
//				return imgLong(image);
//			case STRING:
//				return imgString(image);
//			case UINT8:
//				return imgByte(image);
//			default:
//				throw new UnsupportedOperationException();
//		}
//	}

	/**
	 * Creates an image of type {@link ByteType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#UINT8}.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting image will have dimensions corresponding to the reversed
	 * shape of the Tensor. See {@link #imgByteDirect(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p><p>
	 * Note also that no exception is thrown if the data type is not
	 * {@link DataType#UINT8} but it will give unexpected results.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 */
	public static Img<ByteType> imgByte(final Tensor<UInt8> image) 
	{
		final byte[] out = new byte[image.numElements()];
		image.writeTo(ByteBuffer.wrap(out));
		return ArrayImgs.bytes(out, shape(image));
	}

	/**
	 * Creates an image of type {@link DoubleType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#DOUBLE}.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting image will have dimensions corresponding to the reversed
	 * shape of the Tensor. See {@link #imgByteDirect(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not double.
	 */
	public static Img<DoubleType> imgDouble(final Tensor<Double> image) 
	{
		final double[] out = new double[image.numElements()];
		image.writeTo(DoubleBuffer.wrap(out));
		return ArrayImgs.doubles(out, shape(image));
	}

	/**
	 * Creates an image of type {@link FloatType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#FLOAT}.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting image will have dimensions corresponding to the reversed
	 * shape of the Tensor. See {@link #imgByteDirect(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not float.
	 */
	public static Img<FloatType> imgFloat(final Tensor<Float> image) 
	{
		final float[] out = new float[image.numElements()];
		image.writeTo(FloatBuffer.wrap(out));
		return ArrayImgs.floats(out, shape(image));
	}
        
        

	/**
	 * Creates an image of type {@link IntType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT32}.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting image will have dimensions corresponding to the reversed
	 * shape of the Tensor. See {@link #imgByteDirect(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not int.
	 */
	public static Img<IntType> imgInt(final Tensor<Integer> image) 
	{
		final int[] out = new int[image.numElements()];
		image.writeTo(IntBuffer.wrap(out));
		return ArrayImgs.ints(out, shape(image));
	}

	/**
	 * Creates an image of type {@link LongType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT64}.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting image will have dimensions corresponding to the reversed
	 * shape of the Tensor. See {@link #imgByteDirect(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not long.
	 */
	public static Img<LongType> imgLong(final Tensor<Long> image) 
	{
		final long[] out = new long[image.numElements()];
		image.writeTo(LongBuffer.wrap(out));
		return ArrayImgs.longs(out, shape(image));
	}

	/**
	 * Creates an image of type {@link ByteType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#UINT8}.
	 * <p>
	 * Note that no exception is thrown if the data type is not {@link DataType#UINT8}
	 * but it will give unexpected results.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return An image containing the data of the Tensor.
	 */
	public static Img<ByteType> imgByte(final Tensor<UInt8> image, int[] dimOrder) 
	{
		return reverseReorder(reverse(imgByte(image)), dimOrder);
	}

	/**
	 * Creates an image of type {@link DoubleType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#DOUBLE}.
	 * 
	 * @param image The TensorFlow Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not double.
	 */
	public static Img<DoubleType> imgDouble(final Tensor<Double> image, int[] dimOrder) 
	{
		return reverseReorder(reverse(imgDouble(image)), dimOrder);
	}

	/**
	 * Creates an image of type {@link FloatType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#FLOAT}.
	 * 
	 * @param image The TensorFlow Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not float.
	 */
	public static Img<FloatType> imgFloat(final Tensor<Float> image, int[] dimOrder) {
		return reverseReorder(reverse(imgFloat(image)), dimOrder);
	}
        
        

	/**
	 * Creates an image of type {@link IntType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT32}.
	 * 
	 * @param image The TensorFlow Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not int.
	 */
	public static Img<IntType> imgInt(final Tensor<Integer> image, int[] dimOrder) 
	{
		return reverseReorder(reverse(imgInt(image)), dimOrder);
	}

	/**
	 * Creates an image of type {@link LongType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT64}.
	 * 
	 * @param image The TensorFlow Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not long.
	 */
	public static Img<LongType> imgLong(final Tensor<Long> image, int[] dimOrder) 
	{
		return reverseReorder(reverse(imgLong(image)), dimOrder);
	}

	/**
	 * Creates an image of type {@link ByteType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#UINT8}.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting image will have dimensions directly corresponding to the
	 * shape of the Tensor. See {@link #imgByte(Tensor)} and
	 * {@link #imgByte(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p><p>
	 * Note also that no exception is thrown if the data type is not
	 * {@link DataType#UINT8} but it will give unexpected results.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 */
	public static Img<ByteType> imgByteDirect(final Tensor<UInt8> image) 
	{
		return reverse(imgByte(image));
	}

	/**
	 * Creates an image of type {@link DoubleType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#DOUBLE}.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting image will have dimensions directly corresponding to the
	 * shape of the Tensor. See {@link #imgDouble(Tensor)} and
	 * {@link #imgDouble(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not double.
	 */
	public static Img<DoubleType> imgDoubleDirect(final Tensor<Double> image) 
	{
		return reverse(imgDouble(image));
	}

	/**
	 * Creates an image of type {@link FloatType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#FLOAT}.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting image will have dimensions directly corresponding to the
	 * shape of the Tensor.  See {@link #imgFloat(Tensor)} and
	 * {@link #imgFloat(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not float.
	 */
	public static Img<FloatType> imgFloatDirect(final Tensor<Float> image) 
	{
		return reverse(imgFloat(image));
	}

	/**
	 * Creates an image of type {@link IntType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT32}.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting image will have dimensions directly corresponding to the
	 * shape of the Tensor. See {@link #imgInt(Tensor)} and
	 * {@link #imgInt(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not int.
	 */
	public static Img<IntType> imgIntDirect(final Tensor<Integer> image) 
	{
		return reverse(imgInt(image));
	}

	/**
	 * Creates an image of type {@link LongType} containing the data of a
	 * TensorFlow Tensor with the data type {@link DataType#INT64}.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting image will have dimensions directly corresponding to the
	 * shape of the Tensor. See {@link #imgLong(Tensor)} and
	 * {@link #imgLong(Tensor, int[])} if you want to handle dimensions
	 * differently.
	 * </p>
	 * @param image The TensorFlow Tensor.
	 * @return An image containing the data of the Tensor.
	 * @throws IllegalArgumentException if Tensor data type is not long.
	 */
	public static Img<LongType> imgLongDirect(final Tensor<Long> image) 
	{
		return reverse(imgLong(image));
	}

	// --------- RAI to TENSOR ---------

	/**
	 * Creates a TensorFlow Tensor containing data from the given image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image. This is probably what you want because
	 * TensorFlow uses the dimension order CYX while ImageJ uses XYC. See
	 * {@link #tensorDirect(RandomAccessibleInterval)} and
	 * {@link #tensor(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 * @throws IllegalArgumentException if the type of the image is not supported.
	 *          Supported types are {@link ByteType}, {@link DoubleType},
	 *          {@link FloatType}, {@link IntType} and {@link LongType}.
	 */
	public static <T extends RealType<T>> Tensor<?> tensor(
		final RandomAccessibleInterval<T> image)
	{
		// NB: In the functions called we use reversed dimensions because
		// tensorflow iterates the array differently than imglib2.
		// Example: array: [ 1, 2, 3, 4, 5, 6 ], shape: [ 3, 2 ]
		// imglib2      tensorflow
		// | 1 2 3 |    | 1 3 5 |
		// | 4 5 6 |    | 2 4 6 |

		final T type = Util.getTypeFromInterval(image);
		if (type instanceof ByteType) {
			@SuppressWarnings("unchecked")
			final RandomAccessibleInterval<ByteType> typedImage =
				(RandomAccessibleInterval<ByteType>) image;
			return tensorByte(typedImage);
		}
		if (type instanceof DoubleType) {
			@SuppressWarnings("unchecked")
			final RandomAccessibleInterval<DoubleType> typedImage =
				(RandomAccessibleInterval<DoubleType>) image;
			return tensorDouble(typedImage);
		}
		if (type instanceof FloatType) {
			@SuppressWarnings("unchecked")
			final RandomAccessibleInterval<FloatType> typedImage =
				(RandomAccessibleInterval<FloatType>) image;
			return tensorFloat(typedImage);
		}
		if (type instanceof IntType) {
			@SuppressWarnings("unchecked")
			final RandomAccessibleInterval<IntType> typedImage =
				(RandomAccessibleInterval<IntType>) image;
			return tensorInt(typedImage);
		}
		if (type instanceof LongType) {
			@SuppressWarnings("unchecked")
			final RandomAccessibleInterval<LongType> typedImage =
				(RandomAccessibleInterval<LongType>) image;
			return tensorLong(typedImage);
		}
		throw new IllegalArgumentException("Unsupported image type: " + //
			type.getClass().getName());
	}

	// "higher level" routines that give ways of adjusting dimension order

	/**
	 * Creates a TensorFlow Tensor containing data from the given image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 * @throws IllegalArgumentException if the type of the image is not supported.
	 *          Supported types are {@link ByteType}, {@link DoubleType},
	 *          {@link FloatType}, {@link IntType} and {@link LongType}.
	 */
	public static <T extends RealType<T>> Tensor<?> tensor(
		final RandomAccessibleInterval<T> image, int[] dimOrder)
	{
		// TODO Are 2 calls bad? More views are created but they should be smart
		// about this
		return tensor(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensor(RandomAccessibleInterval)} and
	 * {@link #tensor(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 * @throws IllegalArgumentException if the type of the image is not supported.
	 *          Supported types are {@link ByteType}, {@link DoubleType},
	 *          {@link FloatType}, {@link IntType} and {@link LongType}.
	 */
	public static <T extends RealType<T>> Tensor<?> tensorDirect(
		final RandomAccessibleInterval<T> image)
	{
		return tensor(reverse(image));
	}

	// "low level" methods that do NOT adjust dimensions

	/**
	 * Creates a TensorFlow Tensor containing data from the given byte image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<UInt8> tensorByte(
		final RandomAccessibleInterval<ByteType> image)
	{
		final byte[] value = byteArray(image);
		ByteBuffer buffer = ByteBuffer.wrap(value);
		return Tensor.create(UInt8.class, shape(image), buffer);
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given double image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Double> tensorDouble(
		final RandomAccessibleInterval<DoubleType> image)
	{
		final double[] value = doubleArray(image);
		DoubleBuffer buffer = DoubleBuffer.wrap(value);
		return Tensor.create(shape(image), buffer);
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given float image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Float> tensorFloat(
		final RandomAccessibleInterval<FloatType> image)
	{
		final float[] value = floatArray(image);
		FloatBuffer buffer = FloatBuffer.wrap(value);
		return Tensor.create(shape(image), buffer);
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given int image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Integer> tensorInt(
		final RandomAccessibleInterval<IntType> image)
	{
		final int[] value = intArray(image);
		IntBuffer buffer = IntBuffer.wrap(value);
		return Tensor.create(shape(image), buffer);
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given long image.
	 * <p>
	 * Note that this does _not_ adjust any dimensions. This means that
	 * the resulting Tensor will have a shape corresponding to the reversed
	 * dimensions of the image.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Long> tensorLong(
		final RandomAccessibleInterval<LongType> image)
	{
		final long[] value = longArray(image);
		LongBuffer buffer = LongBuffer.wrap(value);
		return Tensor.create(shape(image), buffer);
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given byte image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<UInt8> tensorByte(
		final RandomAccessibleInterval<ByteType> image, final int[] dimOrder)
	{
		return tensorByte(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given double image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Double> tensorDouble(
		final RandomAccessibleInterval<DoubleType> image, final int[] dimOrder)
	{
		return tensorDouble(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given float image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Float> tensorFloat(
		final RandomAccessibleInterval<FloatType> image, final int[] dimOrder)
	{
		return tensorFloat(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given int image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Integer> tensorInt(
		final RandomAccessibleInterval<IntType> image, final int[] dimOrder)
	{
		return tensorInt(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given long image.
	 * <p>
	 * Note that this will use the backing RAI's primitive array when one is
	 * available and no dimensions where swapped. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @param dimOrder Defines the mapping of the dimensions between the image
	 *          and the Tensor where the index corresponds to the dimension
	 *          in the image and the value corresponds to the dimension in the
	 *          Tensor. TODO Example?
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Long> tensorLong(
		final RandomAccessibleInterval<LongType> image, final int[] dimOrder)
	{
		return tensorLong(reverse(reorder(image, dimOrder)));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given byte image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensorByte(RandomAccessibleInterval)} and
	 * {@link #tensorByte(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<UInt8> tensorByteDirect(
		final RandomAccessibleInterval<ByteType> image)
	{
		return tensorByte(reverse(image));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given double image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensorDouble(RandomAccessibleInterval)} and
	 * {@link #tensorDouble(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Double> tensorDoubleDirect(
		final RandomAccessibleInterval<DoubleType> image)
	{
		return tensorDouble(reverse(image));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given float image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensorFloat(RandomAccessibleInterval)} and
	 * {@link #tensorFloat(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Float> tensorFloatDirect(
		final RandomAccessibleInterval<FloatType> image)
	{
		return tensorFloat(reverse(image));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given int image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensorInt(RandomAccessibleInterval)} and
	 * {@link #tensorInt(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Integer> tensorIntDirect(
		final RandomAccessibleInterval<IntType> image)
	{
		return tensorInt(reverse(image));
	}

	/**
	 * Creates a TensorFlow Tensor containing data from the given long image.
	 * <p>
	 * Note that this _does_ adjust the dimensions. This means that
	 * the resulting Tensor will have a shape directly corresponding to the
	 * dimensions of the image. Make sure the dimensions are as you want
	 * them in TensorFlow. See {@link #tensorLong(RandomAccessibleInterval)} and
	 * {@link #tensorLong(RandomAccessibleInterval, int[])} if you want to handle
	 * dimensions differently.
	 * </p><p>
	 * Also note that this will use the backing RAI's primitive array when one is
	 * available. Otherwise a copy will be made.
	 * </p>
	 * @param image The image which should be put into the Tensor.
	 * @return A Tensor containing the data of the image.
	 */
	public static Tensor<Long> tensorLongDirect(
		final RandomAccessibleInterval<LongType> image)
	{
		return tensorLong(reverse(image));
	}

	// --------- DIMENSIONAL HELPER METHODS ---------

	/**
	 * Gets the TensorFlow shape of an image. This is the same as the image's
	 * dimension lengths, but in reversed order.
	 * 
	 * @param image The image whose shape is desired.
	 * @return The TensorFlow shape.
	 */
	private static long[] shape(final Dimensions image) {
		long[] shape = new long[image.numDimensions()];
		for (int d = 0; d < shape.length; d++) {
			shape[d] = image.dimension(shape.length - d - 1);
		}
		return shape;
	}

	/**
	 * Gets the imglib dimension length of a tensor. This is the same as the
	 * tensor's shape but in reversed order.
	 * 
	 * @param tensor The tensor whose dimension length are desired.
	 * @return The imglib dimension length.
	 */
	private static long[] shape(final Tensor<?> tensor) {
		return shape(new FinalDimensions(tensor.shape()));
	}

	/** Flips all dimensions {@code d0,d1,...,dn -> dn,...,d1,d0}. */
	public static <T extends RealType<T>> Img<T> reverse(Img<T> image)
	{
		RandomAccessibleInterval<T> reversed = reverse((RandomAccessibleInterval<T>) image);
		return ImgView.wrap(reversed, image.factory());
	}

	/** Flips all dimensions {@code d0,d1,...,dn -> dn,...,d1,d0}. */
	public static <T extends RealType<T>> RandomAccessibleInterval<T> reverse(
		RandomAccessibleInterval<T> image)
	{
		RandomAccessibleInterval<T> reversed = image;
		for (int d = 0; d < image.numDimensions() / 2; d++) {
			reversed = Views.permute(reversed, d, image.numDimensions() - d - 1);
		}
		return reversed;
	}

	private static <T extends RealType<T>> Img<T> reorder(
		Img<T> image, int[] dimOrder)
	{
		RandomAccessibleInterval<T> result = reorder((RandomAccessibleInterval<T>) image, dimOrder);
		return ImgView.wrap(result, image.factory());
	}

	private static <T extends RealType<T>> RandomAccessibleInterval<T> reorder(
		RandomAccessibleInterval<T> image, int[] dimOrder)
	{
		RandomAccessibleInterval<T> output = image;

		// Array which contains for each dimension information on which dimension it is right now
		int[] moved = IntStream.range(0, image.numDimensions()).toArray();

		// Loop over all dimensions and move it to the right spot
		for (int i = 0; i < image.numDimensions(); i++) {
			int from = moved[i];
			int to = dimOrder[i];

			// Move the dimension to the right dimension
			output = Views.permute(output, from, to);

			// Now we have to update which dimension was moved where
			moved[i] = to;
			moved = Arrays.stream(moved).map(v -> v == to ? from : v).toArray();
		}
		return output;
	}

	private static <T extends RealType<T>> Img<T> reverseReorder(Img<T> image, int[] dimOrder) {
		int[] reverseDimOrder = new int[dimOrder.length];
		for (int i = 0; i < dimOrder.length; i++) {
			reverseDimOrder[dimOrder[i]] = i;
		}
		return reorder(image, reverseDimOrder);
	}

	// --------- HELPER STUFF ---------

	// TODO: consider also putting this outside Tensors, and instead in ImgLib2 core.
	// _Maybe_ mix with extractFloatArray? Probably better to keep separate layer though.
	// This is really "extract or create float array"
	// So maybe we want a "createFloatArray" which _always_ does the copy.
	// And the a nice "extractOrCreate" guy that calls extract followed by create
	// as is done here.

	private static byte[] byteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		final byte[] array = extractByteArray(image);
		return array == null ? createByteArray(image) : array;
	}

	private static double[] doubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		final double[] array = extractDoubleArray(image);
		return array == null ? createDoubleArray(image) : array;
	}

	private static float[] floatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		final float[] array = extractFloatArray(image);
		return array == null ? createFloatArray(image) : array;
	}

	private static int[] intArray(
		final RandomAccessibleInterval<IntType> image)
	{
		final int[] array = extractIntArray(image);
		return array == null ? createIntArray(image) : array;
	}

	private static long[] longArray(
		final RandomAccessibleInterval<LongType> image)
	{
		final long[] array = extractLongArray(image);
		return array == null ? createLongArray(image) : array;
	}

	private static byte[] createByteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<ByteType, ByteArray> dest = ArrayImgs.bytes(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static double[] createDoubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<DoubleType, DoubleArray> dest = ArrayImgs.doubles(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static float[] createFloatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<FloatType, FloatArray> dest = ArrayImgs.floats(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static int[] createIntArray(
		final RandomAccessibleInterval<IntType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<IntType, IntArray> dest = ArrayImgs.ints(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static long[] createLongArray(
		final RandomAccessibleInterval<LongType> image)
	{
		final long[] dims = Intervals.dimensionsAsLongArray(image);
		final ArrayImg<LongType, LongArray> dest = ArrayImgs.longs(dims);
		copy(image, dest);
		return dest.update(null).getCurrentStorageArray();
	}

	private static byte[] extractByteArray(
		final RandomAccessibleInterval<ByteType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<ByteType, ?> arrayImg = (ArrayImg<ByteType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof ByteArray ? //
			((ByteArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static double[] extractDoubleArray(
		final RandomAccessibleInterval<DoubleType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<DoubleType, ?> arrayImg = (ArrayImg<DoubleType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof DoubleArray ? //
			((DoubleArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static float[] extractFloatArray(
		final RandomAccessibleInterval<FloatType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<FloatType, ?> arrayImg = (ArrayImg<FloatType, ?>) image;
		// GOOD NEWS: float[] rasterization order is dimension-wise!
		// BAD NEWS: it always goes d0,d1,d2,.... is that the order we need?
		// MORE BAD NEWS: As soon as you use Views.permute, image is not ArrayImg anymore.
		// SO: This only works if you give a RAI that happens to be laid out directly as TensorFlow desires.
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof FloatArray ? //
			((FloatArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static int[] extractIntArray(
		final RandomAccessibleInterval<IntType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<IntType, ?> arrayImg = (ArrayImg<IntType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof IntArray ? //
			((IntArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static long[] extractLongArray(
		final RandomAccessibleInterval<LongType> image)
	{
		if (!(image instanceof ArrayImg)) return null;
		@SuppressWarnings("unchecked")
		final ArrayImg<LongType, ?> arrayImg = (ArrayImg<LongType, ?>) image;
		final Object dataAccess = arrayImg.update(null);
		return dataAccess instanceof LongArray ? //
			((LongArray) dataAccess).getCurrentStorageArray() : null;
	}

	private static <T extends RealType<T>> void copy(
		final RandomAccessibleInterval<T> source,
		final IterableInterval<T> dest)
	{
		final RandomAccess<T> sourceAccess = source.randomAccess();
		final Cursor<T> destCursor = dest.localizingCursor();
		while (destCursor.hasNext()) {
			destCursor.fwd();
			sourceAccess.setPosition(destCursor);
			destCursor.get().set(sourceAccess.get());
		}
	}
        
    public static float max_image(final RandomAccessibleInterval<FloatType> ix1_m1)
    {
        final float[] value_m1 = floatArray(ix1_m1);
        final int ndims = ix1_m1.numDimensions();
        final long[] dims = Intervals.dimensionsAsLongArray(ix1_m1);
        //System.out.println("dims are here" + dims);
        long all_dims=1;
        for (int j=0;j<ndims;j++)
        {
            all_dims = all_dims*dims[j];
        }
        float max_im = 0;
        int i;
        for (i = 0; i < all_dims; i++) 
        { 
            
                if (value_m1[i]> max_im)
                {
                        max_im = value_m1[i];
                }
        }
        
        //if (max_im >255) //16-bit data
        //{
        //    max_im = 65535;
        //}
        //else
        //{
        //    max_im = 255; //8-bit data
        //}
        
        return max_im;
    }
}
