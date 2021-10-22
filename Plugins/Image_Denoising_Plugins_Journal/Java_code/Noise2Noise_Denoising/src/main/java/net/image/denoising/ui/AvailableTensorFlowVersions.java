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

package net.image.denoising.ui;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class providing a hardcoded list of available native TensorFlow library versions which can be downloaded from the google servers.
 *
 * @author Varun Mannam
 */
public class AvailableTensorFlowVersions 
{

	/**
	 * @return a list of downloadable native TensorFlow versions for all operating systems, including their URL and optionally their CUDA / CuDNN compatibility
	 */
	public static List<DownloadableTensorFlowVersion> get() 
        {
		List<DownloadableTensorFlowVersion> versions = new ArrayList<>();
		versions.clear();
		//linux64
		versions.add(version("linux64", "1.2.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.2.0.tar.gz"));
		versions.add(version("linux64", "1.2.0", "GPU", "8.0", "5.1", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.2.0.tar.gz"));
		versions.add(version("linux64", "1.3.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.3.0.tar.gz"));
		versions.add(version("linux64", "1.3.0", "GPU", "8.0", "6", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.3.0.tar.gz"));
		versions.add(version("linux64", "1.4.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.4.1.tar.gz"));
		versions.add(version("linux64", "1.4.1", "GPU", "8.0", "6", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.4.1.tar.gz"));
		versions.add(version("linux64", "1.5.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.5.0.tar.gz"));
		versions.add(version("linux64", "1.5.0", "GPU", "9.0", "7", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.5.0.tar.gz"));
		versions.add(version("linux64", "1.6.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.6.0.tar.gz"));
		versions.add(version("linux64", "1.6.0", "GPU", "9.0", "7", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.6.0.tar.gz"));
		versions.add(version("linux64", "1.7.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.7.0.tar.gz"));
		versions.add(version("linux64", "1.7.0", "GPU", "9.0", "7", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.7.0.tar.gz"));
		versions.add(version("linux64", "1.8.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.8.0.tar.gz"));
		versions.add(version("linux64", "1.8.0", "GPU", "9.0", "7", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.8.0.tar.gz"));
		versions.add(version("linux64", "1.9.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.9.0.tar.gz"));
		versions.add(version("linux64", "1.9.0", "GPU", "9.0", "7", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.9.0.tar.gz"));
		versions.add(version("linux64", "1.10.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.10.1.tar.gz"));
		versions.add(version("linux64", "1.10.1", "GPU", "9.0", "7.?", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.10.1.tar.gz"));
		versions.add(version("linux64", "1.11.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.11.0.tar.gz"));
		versions.add(version("linux64", "1.11.0", "GPU", "9.0", ">= 7.2", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.11.0.tar.gz"));
		versions.add(version("linux64", "1.12.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.12.0.tar.gz"));
		versions.add(version("linux64", "1.12.0", "GPU", "9.0", ">= 7.2", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.12.0.tar.gz"));
		versions.add(version("linux64", "1.13.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.13.1.tar.gz"));
		versions.add(version("linux64", "1.13.1", "GPU", "10.0", "7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.13.1.tar.gz"));
		versions.add(version("linux64", "1.14.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.14.0.tar.gz"));
		versions.add(version("linux64", "1.14.0", "GPU", "10.0", ">= 7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.14.0.tar.gz"));
		versions.add(version("linux64", "1.15.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.15.0.tar.gz"));
		versions.add(version("linux64", "1.15.0", "GPU", "10.0", ">= 7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.15.0.tar.gz"));
		//win64
		versions.add(version("win64", "1.2.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.2.0.zip"));
		versions.add(version("win64", "1.3.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.3.0.zip"));
		versions.add(version("win64", "1.4.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.4.1.zip"));
		versions.add(version("win64", "1.5.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.5.0.zip"));
		versions.add(version("win64", "1.6.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.6.0.zip"));
		versions.add(version("win64", "1.7.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.7.0.zip"));
		versions.add(version("win64", "1.8.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.8.0.zip"));
		versions.add(version("win64", "1.9.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.9.0.zip"));
		versions.add(version("win64", "1.10.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.10.0.zip"));
		versions.add(version("win64", "1.11.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.11.0.zip"));
		versions.add(version("win64", "1.12.0", "GPU", "9.0", ">= 7.2", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.12.0.zip"));
		versions.add(version("win64", "1.12.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.12.0.zip"));
		versions.add(version("win64", "1.13.1", "GPU", "10.0", "7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.13.1.zip"));
		versions.add(version("win64", "1.13.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.13.1.zip"));
		versions.add(version("win64", "1.14.0", "GPU", "10.0", ">= 7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.14.0.zip"));
		versions.add(version("win64", "1.14.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.14.0.zip"));
		versions.add(version("win64", "1.15.0", "GPU", "10.0", ">= 7.4", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.15.0.zip"));
		versions.add(version("win64", "1.15.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.15.0.zip"));
		//macosx
		versions.add(version("macosx", "1.2.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.2.0.tar.gz"));
		versions.add(version("macosx", "1.3.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.3.0.tar.gz"));
		versions.add(version("macosx", "1.4.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.4.1.tar.gz"));
		versions.add(version("macosx", "1.5.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.5.0.tar.gz"));
		versions.add(version("macosx", "1.6.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.6.0.tar.gz"));
		versions.add(version("macosx", "1.7.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.7.0.tar.gz"));
		versions.add(version("macosx", "1.8.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.8.0.tar.gz"));
		versions.add(version("macosx", "1.9.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.9.0.tar.gz"));
		versions.add(version("macosx", "1.10.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.10.1.tar.gz"));
		versions.add(version("macosx", "1.11.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.11.0.tar.gz"));
		versions.add(version("macosx", "1.12.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.12.0.tar.gz"));
		versions.add(version("macosx", "1.13.1", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.13.1.tar.gz"));
		versions.add(version("macosx", "1.14.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.14.0.tar.gz"));
		versions.add(version("macosx", "1.14.0", "CPU", "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.15.0.tar.gz"));
		return versions;
	}

	private static DownloadableTensorFlowVersion version(String platform, String tensorFlowVersion, String mode, String url) 
        {
			DownloadableTensorFlowVersion version = new DownloadableTensorFlowVersion(tensorFlowVersion, mode.equals("GPU"));
			try 
			{
				version.setURL(new URL(url));
			} 
            catch (MalformedURLException e) 
            {
            	e.printStackTrace();
            }
			version.setPlatform(platform);
			return version;
        }

	private static DownloadableTensorFlowVersion version(String platform, String tensorFlowVersion, String mode, String cuda, String cudnn, String url)
        {
			try 
            {
				return new DownloadableTensorFlowVersion(new URL(url), tensorFlowVersion, platform, mode.equals("GPU"), cuda, cudnn);
            } 
            catch (MalformedURLException e)
            {
            	e.printStackTrace();
            	return null;
            }
	}
}
