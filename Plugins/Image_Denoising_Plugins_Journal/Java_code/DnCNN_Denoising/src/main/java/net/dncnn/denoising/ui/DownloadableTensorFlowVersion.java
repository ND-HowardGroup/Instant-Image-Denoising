/*-
 * #%L
 * ImageJ/TensorFlow integration.
 * %%
 * Copyright (C) 2019 Board of Regents of the University of Notre Dame Electrical Engineering
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

package net.dncnn.denoising.ui;

import java.net.URL;
import java.util.Objects;

import net.dncnn.denoising.TensorFlowVersion;

/**
 * This class extends {@link TensorFlowVersion} with knowledge about the web source of this version, the matching platform and whether it has already been downloaded.
 *
 * @author Varun Mannam
 * 
 */
class DownloadableTensorFlowVersion extends TensorFlowVersion 
{
	private URL url;
	private String platform;
	private String localPath;

	private boolean active = false;
	private boolean downloaded = false;

	/**
	 * @param url the URL of the packed JNI file
	 * @param version the TensorFlow version number
	 * @param os the platform this version is associated with
	 * @param supportsGPU whether this version runs on the GPU
	 * @param compatibleCUDA the CUDA version compatible with this TensorFlow version
	 * @param compatibleCuDNN the CuDNN version compatible with this TensorFlow version
	 */
	DownloadableTensorFlowVersion(URL url, String version, String os, boolean supportsGPU, String compatibleCUDA, String compatibleCuDNN) {
		super(version, supportsGPU, compatibleCUDA, compatibleCuDNN);
		this.platform = os;
		this.url = url;
	}

	/**
	 * @param version the TensorFlow version number
	 * @param supportsGPU whether this version runs on the GPU
	 */
	DownloadableTensorFlowVersion(String version, boolean supportsGPU) 
	{
		super(version, supportsGPU, null, null);
	}

	DownloadableTensorFlowVersion(TensorFlowVersion other) 
	{
		super(other);
	}

	/**
	 * @return the URL where an archive of this version can be downloaded from
	 */
	public URL getURL() 
	{
		return url;
	}

	void setURL(URL url) 
	{
		this.url = url;
	}

	/**
	 * @return the platform this version is associated with (linux64, linux32, win64, win32, macosx)
	 */
	public String getPlatform() 
	{
		return platform;
	}

	void setPlatform(String platform) 
	{
		this.platform = platform;
	}

	/**
	 * @return the path this version is cached to
	 */
	public String getLocalPath() 
	{
		return localPath;
	}

	/**
	 * @return whether this version is currently being used
	 */
	public boolean isActive() 
	{
		return active;
	}

	void setActive(boolean active) 
	{
		this.active = active;
	}

	/**
	 * @return whether this version is cashed to a local path
	 */
	public boolean isCached() 
	{
		return downloaded;
	}

	/**
	 * @return String describing the origin of this version
	 */
	public String getOriginDescription() 
	{
		if(downloaded) return localPath;
		return url.toString();
	}

	/**
	 * @return the TensorFlow version number in an easily comparable format (e.g. 0.13.1 to 000.013.001)
	 */
	String getComparableTFVersion() 
	{
		String orderableVersion = "";
		String[] split = tfVersion.split("\\.");
		for(String part : split) 
		{
			orderableVersion += String.format("%03d%n", Integer.valueOf(part));
		}
		return orderableVersion;
	}

	void setCached(String localPath) 
	{
		downloaded = true;
		this.localPath = localPath;
	}

	void discardCache() 
	{
		downloaded = false;
	}

	void harvest(DownloadableTensorFlowVersion other) 
	{
		if(!cuda.isPresent() && other.cuda.isPresent()) 
		{
			cuda = other.cuda;
		}
		if(!cudnn.isPresent() && other.cudnn.isPresent()) 
		{
			cudnn = other.cudnn;
		}
		if(url == null) url = other.url;
		if(localPath == null) localPath = other.localPath;
		downloaded = other.downloaded;
	}

	@Override
	public boolean equals(final Object obj) 
	{
		if (!(obj.getClass().equals(this.getClass()))) return false;
		final DownloadableTensorFlowVersion o = (DownloadableTensorFlowVersion) obj;
		return super.equals(o) && Objects.equals(platform, o.platform);
	}
}
