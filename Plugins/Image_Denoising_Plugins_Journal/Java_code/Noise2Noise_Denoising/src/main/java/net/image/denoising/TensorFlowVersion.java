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

package net.image.denoising;

import java.util.Objects;
import java.util.Optional;

/**
 * A class describing the details of a specific TensorFlow version.
 *
 * @author Varun Mannam
 * 
 */
public class TensorFlowVersion 
{

	protected final String tfVersion;
	protected final Optional<Boolean> supportsGPU;
	protected Optional<String> cuda;
	protected Optional<String> cudnn;

	/**
	 * @param version         the TensorFlow version number
	 * @param supportsGPU     whether this version runs on the GPU
	 * @param compatibleCUDA  the CUDA version compatible with this TensorFlow
	 *                        version
	 * @param compatibleCuDNN the CuDNN version compatible with this TensorFlow
	 *                        version
	 */
	public TensorFlowVersion(final String version, final Boolean supportsGPU, final String compatibleCUDA,
			final String compatibleCuDNN) 
	{
		this.tfVersion = version;
		this.supportsGPU = Optional.ofNullable(supportsGPU);
		this.cuda = Optional.ofNullable(compatibleCUDA);
		this.cudnn = Optional.ofNullable(compatibleCuDNN);
	}

	public TensorFlowVersion(TensorFlowVersion other) 
	{
		this.tfVersion  = other.tfVersion;
		this.supportsGPU = other.supportsGPU;
		this.cuda = other.cuda;
		this.cudnn = other.cudnn;
	}

	/**
	 * @return the version number of TensorFlow, e.g. 0.13.1
	 */
	public String getVersionNumber() 
	{
		return tfVersion;
	}

	/**
	 * @return if the version supports the usage of the GPU
	 */
	public Optional<Boolean> usesGPU() 
	{
		return supportsGPU;
	}

	/**
	 * @return the CUDA version this TensorFlow version is compatible with
	 */
	public Optional<String> getCompatibleCUDA() 
	{
		return cuda;
	}

	/**
	 * @return the CuDNN version this TensorFlow version is compatible with
	 */
	public Optional<String> getCompatibleCuDNN() 
	{
		return cudnn;
	}

	@Override
	public boolean equals(final Object obj) 
	{
		if (!(obj.getClass().equals(this.getClass()))) return false;
		final TensorFlowVersion o = (TensorFlowVersion) obj;
		return tfVersion.equals(o.tfVersion) //
				&& supportsGPU.equals(o.supportsGPU);
	}

	@Override
	public int hashCode() 
	{
		return Objects.hash(tfVersion, supportsGPU, cuda, cudnn);
	}

	@Override
	public String toString() 
	{
		String text = "TF " + tfVersion;
		if(supportsGPU.isPresent()) text += " " + (supportsGPU.get() ? "GPU" : "CPU");
		if (cuda.isPresent() || cudnn.isPresent()) 
		{
			text += " (";
			if (cuda.isPresent())
				text += "CUDA " + cuda.get();
			if (cuda.isPresent() && cudnn.isPresent())
				text += ", ";
			if (cudnn.isPresent())
				text += "CuDNN " + cudnn.get();
			text += ")";
		}
		return text;
	}
}
