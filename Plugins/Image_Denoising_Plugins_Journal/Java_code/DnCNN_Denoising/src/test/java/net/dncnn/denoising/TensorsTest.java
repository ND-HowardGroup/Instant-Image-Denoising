/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2017 - 2020 Board of Regents of the University of
 * Wisconsin-Madison and Google, Inc.
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

package net.dncnn.denoising;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

import org.junit.Test;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import com.google.common.base.Function;

import net.dncnn.denoising.Tensors;
import net.imglib2.Point;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;

public class TensorsTest 
{

	// --------- TENSOR to RAI ---------

	/** Tests the img<Type>(Tensor) functions */
	@Test
	public void testTensorToImgReverse() 
	{
		final long[] shape = new long[] { 20, 10, 3 };
		final int[] mapping = new int[] { 2, 1, 0 };
		final long[] dims = new long[] { 3, 10, 20 };
		final int n = shape.length;
		final int size = 600;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor<UInt8> tensorByte = Tensor.create(UInt8.class, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByte(tensorByte);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor<Double> tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDouble(tensorDouble);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor<Float> tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloat(tensorFloat);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor<Integer> tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgInt(tensorInt);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor<Long> tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLong(tensorLong);

		// Check all created images
		checkImage(imgByte, n, dims, points);
		checkImage(imgDouble, n, dims, points);
		checkImage(imgFloat, n, dims, points);
		checkImage(imgInt, n, dims, points);
		checkImage(imgLong, n, dims, points);
	}

	/** Tests the img<Type>Direct(Tensor) functions */
	@Test
	public void testTensorToImgDirect() 
	{
		final long[] shape = new long[] { 20, 10, 3 };
		final int[] mapping = new int[] { 0, 1, 2 };
		final int n = shape.length;
		final int size = 600;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor<UInt8> tensorByte = Tensor.create(UInt8.class, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByteDirect(tensorByte);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor<Double> tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDoubleDirect(tensorDouble);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor<Float> tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloatDirect(tensorFloat);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor<Integer> tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgIntDirect(tensorInt);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor<Long> tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLongDirect(tensorLong);

		// Check all created images
		checkImage(imgByte, n, shape, points);
		checkImage(imgDouble, n, shape, points);
		checkImage(imgFloat, n, shape, points);
		checkImage(imgInt, n, shape, points);
		checkImage(imgLong, n, shape, points);
	}

	/** Tests the img<Type>(Tensor, int[]) functions */
	@Test
	public void testTensorToImgMapping() 
	{
		final long[] shape = new long[] { 3, 5, 2, 4 };
		final int[] mapping = new int[] { 1, 3, 0, 2 }; // A strange mapping
		final long[] dims = new long[] { 5, 4, 3, 2 };
		final int n = shape.length;
		final int size = 120;

		// Get some points to mark
		List<Point> points = createTestPoints(n);

		// Create Tensors of different type and convert them to images
		ByteBuffer dataByte = ByteBuffer.allocateDirect(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataByte.put(i, (byte) (int) v));
		Tensor<UInt8> tensorByte = Tensor.create(UInt8.class, shape, dataByte);
		Img<ByteType> imgByte = Tensors.imgByte(tensorByte, mapping);

		DoubleBuffer dataDouble = DoubleBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataDouble.put(i, v));
		Tensor<Double> tensorDouble = Tensor.create(shape, dataDouble);
		Img<DoubleType> imgDouble = Tensors.imgDouble(tensorDouble, mapping);

		FloatBuffer dataFloat = FloatBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataFloat.put(i, v));
		Tensor<Float> tensorFloat = Tensor.create(shape, dataFloat);
		Img<FloatType> imgFloat = Tensors.imgFloat(tensorFloat, mapping);

		IntBuffer dataInt = IntBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataInt.put(i, v));
		Tensor<Integer> tensorInt = Tensor.create(shape, dataInt);
		Img<IntType> imgInt = Tensors.imgInt(tensorInt, mapping);

		LongBuffer dataLong = LongBuffer.allocate(size);
		execForPointsWithBufferIndex(shape, mapping, points, (i, v) -> dataLong.put(i, v));
		Tensor<Long> tensorLong = Tensor.create(shape, dataLong);
		Img<LongType> imgLong = Tensors.imgLong(tensorLong, mapping);

		// Check all created images
		checkImage(imgByte, n, dims, points);
		checkImage(imgDouble, n, dims, points);
		checkImage(imgFloat, n, dims, points);
		checkImage(imgInt, n, dims, points);
		checkImage(imgLong, n, dims, points);
	}

	/** Checks one image for the dimensions and marked points */
	private <T extends RealType<T>> void checkImage(final Img<T> img, final int n, final long[] dims,
			final List<Point> points) 
	{
		assertArrayEquals(dims, Intervals.dimensionsAsLongArray(img));
		assertEquals(n, img.numDimensions());
		checkPoints(img, points);
	}

	// --------- RAI to TENSOR ---------

	/** Tests the tensor(RAI) function */
	@Test
	public void testImgToTensorReverse() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final long[] shape = new long[] { 3, 10, 20 };
		final int n = dims.length;

		// ByteType
		testImg2TensorReverse(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, shape,
				DataType.UINT8);

		// DoubleType
		testImg2TensorReverse(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, shape,
				DataType.DOUBLE);

		// FloatType
		testImg2TensorReverse(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, shape,
				DataType.FLOAT);

		// IntType
		testImg2TensorReverse(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, shape,
				DataType.INT32);

		// LongType
		testImg2TensorReverse(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, shape,
				DataType.INT64);
	}

	/** Tests the tensorDirect(RAI) function */
	@Test
	public void testImgToTensorDirect() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final int n = dims.length;

		// ByteType
		testImg2TensorDirect(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, dims,
				DataType.UINT8);

		// DoubleType
		testImg2TensorDirect(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, dims,
				DataType.DOUBLE);

		// FloatType
		testImg2TensorDirect(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, dims,
				DataType.FLOAT);

		// IntType
		testImg2TensorDirect(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, dims,
				DataType.INT32);

		// LongType
		testImg2TensorDirect(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, dims,
				DataType.INT64);
	}

	/** Tests the tensor(RAI, int[]) function */
	@Test
	public void testImgToTensorMapping() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 5, 4, 3, 2 };
		final int[] mapping = new int[] { 1, 3, 0, 2 }; // A strange mapping
		final long[] shape = new long[] { 3, 5, 2, 4 };
		final int n = dims.length;

		// ByteType
		testImg2TensorMapping(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), mapping, n, shape,
				DataType.UINT8);

		// DoubleType
		testImg2TensorMapping(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), mapping, n,
				shape, DataType.DOUBLE);

		// FloatType
		testImg2TensorMapping(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), mapping, n, shape,
				DataType.FLOAT);

		// IntType
		testImg2TensorMapping(new ArrayImgFactory<IntType>().create(dims, new IntType()), mapping, n, shape,
				DataType.INT32);

		// LongType
		testImg2TensorMapping(new ArrayImgFactory<LongType>().create(dims, new LongType()), mapping, n, shape,
				DataType.INT64);
	}

	/** Tests the tensor<Type>(RAI) function */
	@Test
	public void testImgToTensorReverseTyped() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final long[] shape = new long[] { 3, 10, 20 };
		final int n = dims.length;

		// ByteType
		testImg2TensorReverseTyped(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, shape,
				UInt8.class, (i) -> Tensors.tensorByte(i));

		// DoubleType
		testImg2TensorReverseTyped(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, shape,
				Double.class, (i) -> Tensors.tensorDouble(i));

		// FloatType
		testImg2TensorReverseTyped(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, shape,
				Float.class, (i) -> Tensors.tensorFloat(i));

		// IntType
		testImg2TensorReverseTyped(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, shape,
				Integer.class, (i) -> Tensors.tensorInt(i));

		// LongType
		testImg2TensorReverseTyped(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, shape,
				Long.class, (i) -> Tensors.tensorLong(i));
	}

	/** Tests the tensor<Type>Direct(RAI) function */
	@Test
	public void testImgToTensorDirectTyped() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 20, 10, 3 };
		final int n = dims.length;

		// ByteType
		testImg2TensorDirectTyped(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), n, dims,
				UInt8.class, (i) -> Tensors.tensorByteDirect(i));

		// DoubleType
		testImg2TensorDirectTyped(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), n, dims,
				Double.class, (i) -> Tensors.tensorDoubleDirect(i));

		// FloatType
		testImg2TensorDirectTyped(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), n, dims,
				Float.class, (i) -> Tensors.tensorFloatDirect(i));

		// IntType
		testImg2TensorDirectTyped(new ArrayImgFactory<IntType>().create(dims, new IntType()), n, dims,
				Integer.class, (i) -> Tensors.tensorIntDirect(i));

		// LongType
		testImg2TensorDirectTyped(new ArrayImgFactory<LongType>().create(dims, new LongType()), n, dims,
				Long.class, (i) -> Tensors.tensorLongDirect(i));
	}

	/** Tests the tensor<Type>(RAI, int[]) function */
	@Test
	public void testImgToTensorMappingTyped() 
	{
		assertEquals(1, 1);

		final long[] dims = new long[] { 5, 4, 3, 2 };
		final int[] mapping = new int[] { 1, 3, 0, 2 }; // A strange mapping
		final long[] shape = new long[] { 3, 5, 2, 4 };
		final int n = dims.length;

		// ByteType
		testImg2TensorMappingTyped(new ArrayImgFactory<ByteType>().create(dims, new ByteType()), mapping, n, shape,
				UInt8.class, (i) -> Tensors.tensorByte(i, mapping));

		// DoubleType
		testImg2TensorMappingTyped(new ArrayImgFactory<DoubleType>().create(dims, new DoubleType()), mapping, n, shape,
				Double.class, (i) -> Tensors.tensorDouble(i, mapping));

		// FloatType
		testImg2TensorMappingTyped(new ArrayImgFactory<FloatType>().create(dims, new FloatType()), mapping, n, shape,
				Float.class, (i) -> Tensors.tensorFloat(i, mapping));

		// IntType
		testImg2TensorMappingTyped(new ArrayImgFactory<IntType>().create(dims, new IntType()), mapping, n, shape,
				Integer.class, (i) -> Tensors.tensorInt(i, mapping));

		// LongType
		testImg2TensorMappingTyped(new ArrayImgFactory<LongType>().create(dims, new LongType()), mapping, n, shape,
				Long.class, (i) -> Tensors.tensorLong(i, mapping));
	}

	/** Tests the tensor(RAI) function for one image */
	private <T extends RealType<T>> void testImg2TensorReverse(final Img<T> img, final int n, final long[] shape,
			final DataType t) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor<?> tensor = Tensors.tensor(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).map(i -> n - 1 - i).toArray(), points);
	}

	/** Tests the tensorDirect(RAI) function for one image */
	private <T extends RealType<T>> void testImg2TensorDirect(final Img<T> img, final int n, final long[] shape,
			final DataType t) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor<?> tensor = Tensors.tensorDirect(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).toArray(), points);
	}

	/** Tests the tensor(RAI, int[]) function for one image */
	private <T extends RealType<T>> void testImg2TensorMapping(final Img<T> img, final int[] mapping,
			final int n, final long[] shape, final DataType t) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);
		Tensor<?> tensor = Tensors.tensor(img, mapping);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(t, tensor.dataType());
		checkPointsTensor(tensor, mapping, points);
	}

	/** Tests the tensor<Type>(RAI) typed function for one image */
	private <T extends RealType<T>, U> void testImg2TensorReverseTyped(final Img<T> img, final int n, final long[] shape,
			Class<U> t, final Function<Img<T>, Tensor<U>> converter) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor<U> tensor = converter.apply(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(DataType.fromClass(t), tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).map(i -> n - 1 - i).toArray(), points);
	}

	/** Tests the tensor<Type>Direct(RAI) typed function for one image */
	private <T extends RealType<T>, U> void testImg2TensorDirectTyped(final Img<T> img, final int n, final long[] shape,
			Class<U> t, final Function<Img<T>, Tensor<U>> converter) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor<U> tensor = converter.apply(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(DataType.fromClass(t), tensor.dataType());
		checkPointsTensor(tensor, IntStream.range(0, n).toArray(), points);
	}

	/** Tests the tensor<Type>(RAI, int[]) typed function for one image */
	private <T extends RealType<T>, U> void testImg2TensorMappingTyped(final Img<T> img, final int[] mapping, final int n,
			final long[] shape, Class<U> t, final Function<Img<T>, Tensor<U>> converter) 
	{
		// Put some values to check into the image
		List<Point> points = createTestPoints(n);
		markPoints(img, points);

		Tensor<U> tensor = converter.apply(img);

		assertArrayEquals(shape, tensor.shape());
		assertEquals(n, tensor.numDimensions());
		assertEquals(DataType.fromClass(t), tensor.dataType());
		checkPointsTensor(tensor, mapping, points);
	}

	/** Creates some interesting points to mark */
	private List<Point> createTestPoints(int n) 
	{
		List<Point> points = new ArrayList<>();
		points.add(new Point(n));
		for (int d = 0; d < n; d++) 
		{
			Point p = new Point(n);
			p.fwd(d);
			points.add(p);
		}
		return points;
	}

	/** Marks a list of points in an image */
	private <T extends RealType<T>> void markPoints(final Img<T> img, final List<Point> points) 
	{
		for (int i = 0; i < points.size(); i++) 
		{
			RandomAccess<T> randomAccess = img.randomAccess();
			randomAccess.setPosition(points.get(i));
			randomAccess.get().setReal(i + 1);
		}
	}

	/** Checks if points in an image are set to the right value */
	private <T extends RealType<T>> void checkPoints(final Img<T> img, List<Point> points) 
	{
		for (int i = 0; i < points.size(); i++) 
		{
			RandomAccess<T> randomAccess = img.randomAccess();
			randomAccess.setPosition(points.get(i));
			assertEquals(i + 1, randomAccess.get().getRealFloat(), 0.001);
		}
	}

	/**
	 * Calculates the index of a point in a buffer of a Tensor of the given
	 * shape
	 */
	private int calcIndex(final long[] shape, final int[] mapping, final Point p) 
	{
		assert p.numDimensions() == shape.length;
		int n = shape.length;

		// switch index and value of the mapping
		int[] switchedMapping = new int[mapping.length];
		IntStream.range(0, mapping.length).forEach(i -> switchedMapping[mapping[i]] = i);

		int index = 0;
		for (int dimPreSize = 1, d = n - 1; d >= 0; d--) {
			index += dimPreSize * p.getIntPosition(switchedMapping[d]);
			dimPreSize *= shape[d];
		}
		return index;
	}

	/** Checks the given points for the given tensor */
	private void checkPointsTensor(final Tensor<?> tensor, final int[] mapping, final List<Point> points) 
	{
		switch (tensor.dataType()) 
		{
		case UINT8: 
		{
			ByteBuffer buffer = ByteBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case DOUBLE: 
		{
			DoubleBuffer buffer = DoubleBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case FLOAT: 
		{
			FloatBuffer buffer = FloatBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case INT32: 
		{
			IntBuffer buffer = IntBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		case INT64: 
		{
			LongBuffer buffer = LongBuffer.allocate(tensor.numElements());
			tensor.writeTo(buffer);
			execForPointsWithBufferIndex(tensor.shape(), mapping, points,
					(i, v) -> assertEquals(v, buffer.get(i), 0.001));
		}
			break;
		default:
			// This should not happen because the type is checked before
			fail("Tensor has unsupported type.");
		}
	}

	/**
	 * Calculates the index of a point in a Tensor buffer and executes the
	 * BiConsumer with the index and the value of the point (which is the index
	 * in the list + 1).
	 */
	private void execForPointsWithBufferIndex(final long[] shape, final int[] mapping, final List<Point> points,
			final BiConsumer<Integer, Integer> exec) 
	{
		for (int i = 0; i < points.size(); i++) 
		{
			exec.accept(calcIndex(shape, mapping, points.get(i)), i + 1);
		}
	}
}
