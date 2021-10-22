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
/**
 * Default implementation of {@link GraphBuilder}.
 *
 * @author Varun Mannam
 */

package net.image.denoising;

import java.nio.FloatBuffer;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

// In the fullness of time, equivalents of the methods of this class should be
// auto-generated from the OpDefs linked into libtensorflow_jni.so. That would
// match what is done in other languages like Python, C++ and Go.

public class GraphBuilder 
{

	private final Graph g;

	public GraphBuilder(final Graph g) 
        {
		this.g = g;
	}

	public <T, U, V> Output<V> div(final Output<T> x, final Output<U> y) 
        {
		return binaryOp3("Div", x, y, "Div");
	}

	public <T, U, V> Output<V> div(final Output<T> x, final Output<U> y, final String name)
	{
		return binaryOp3("Div", x, y, name);
	}

	public <T> Output<T> sub(final Output<T> x, final Output<T> y) 
        {
		return binaryOp("Sub", x, y, "Sub");
	}

	public <T> Output<T> sub(final Output<T> x, final Output<T> y, final String name)
	{
		return binaryOp("Sub", x, y, name);
	}

	public <T, U, V> Output<V> resizeBilinear(final Output<T> images, final Output<U> size)
        {
		return binaryOp3("ResizeBilinear", images, size, "ResizeBilinear");
	}

	public <T, U, V> Output<V> resizeBilinear(final Output<T> images, final Output<U> size, final String name)
	{
		return binaryOp3("ResizeBilinear", images, size, name);
	}

	public <T, U, V> Output<V> expandDims(final Output<T> input, final Output<U> dim) 
        {
		return binaryOp3("ExpandDims", input, dim, "ExpandDims");
	}

	public <T, U, V> Output<V> expandDims(final Output<T> input, final Output<U> dim, final String name)
	{
		return binaryOp3("ExpandDims", input, dim, name);
	}

	public Output<?> constant(final String name, final Object value) 
        {
		try (Tensor<?> t = Tensor.create(value)) 
                {
			return constant(name, t);
		}
	}

	public Output<Float> constant(final String name, final float[] value, final long... shape)
	{
		try (Tensor<Float> t = Tensor.create(shape, FloatBuffer.wrap(value))) 
                {
			return constant(name, t);
		}
	}

	public <T> Output<T> constant(final String name, final Tensor<T> value) 
        {
		return g.opBuilder("Const", name).setAttr("dtype", value.dataType()).setAttr("value", value).build().output(0);
	}

	private <T> Output<T> binaryOp(final String type, final Output<T> in1,final Output<T> in2, final String name)
	{
		return g.opBuilder(type, name).addInput(in1).addInput(in2).build().output(0);
	}

	private <T, U, V> Output<V> binaryOp3(final String type, final Output<T> in1,final Output<U> in2, final String name)
	{
		return g.opBuilder(type, name).addInput(in1).addInput(in2).build().output(0);
	}
}
