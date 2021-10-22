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

package net.dncnn.denoising.util;

import net.dncnn.denoising.TensorFlowVersion;
import net.imagej.updater.util.UpdaterUtil;
import org.scijava.log.Logger;
import org.tensorflow.TensorFlow;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Utility methods for dealing with the Java TensorFlow library
 *
 * @author Varun Mannam
 * 
 */
public final class TensorFlowUtil 
{

	private static final String TFVERSIONFILE = ".tensorflowversion";
	private static final String CRASHFILE = ".crashed";
	private static final String LIBDIR = "lib";
	private static final String UPDATEDIR = "update";

	private static final String PLATFORM = UpdaterUtil.getPlatform();

	private static Pattern jarVersionPattern = Pattern.compile("(?<=(libtensorflow-)).*(?=(.jar))");

	private TensorFlowUtil(){}

	/**
	 * @param jar the JAR file of the TensorFlow library
	 * @return the version which will be loaded by the JAR file
	 */
	public static TensorFlowVersion getTensorFlowJARVersion(URL jar) 
	{
		Matcher matcher = jarVersionPattern.matcher(jar.getPath());
		if(matcher.find()) {
			// guess GPU support by looking for tensorflow_jni_gpu in the class path
			boolean supportsGPU = false;
			ClassLoader cl = ClassLoader.getSystemClassLoader();
			for(URL url: ((URLClassLoader)cl).getURLs())
			{
				if(url.getFile().contains("libtensorflow_jni_gpu")) 
				{
					supportsGPU = true;
					break;
				}
			}
			return new TensorFlowVersion(matcher.group(), supportsGPU, null, null);
		}
		return null;
	}

	/**
	 * @return The TensorFlow version included in the class path which will be loaded by default if no other native library is loaded beforehand.
	 */
	public static TensorFlowVersion versionFromClassPathJAR() 
	{
		return getTensorFlowJARVersion(getTensorFlowJAR());
	}

	/**
	 * @return The JAR file URL shipping TensorFlow in Java
	 */
	public static URL getTensorFlowJAR() 
	{
		URL resource = TensorFlow.class.getResource("TensorFlow.class");
		JarURLConnection connection = null;
		try 
		{
			connection = (JarURLConnection) resource.openConnection();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		return connection.getJarFileURL();
	}

	/**
	 * Deletes all TensorFlow native library files in {@link #getLibDir(String)}.
	 * @param root the root path of ImageJ
	 * @param logger
	 */
	public static void removeNativeLibraries(String root, Logger logger)  
	{
		final File folder = new File(TensorFlowUtil.getLibDir(root), PLATFORM);
		if(!folder.exists()) 
		{
			return;
		}
		final File[] listOfFiles = folder.listFiles();
		for (File file : listOfFiles) 
		{
			if (file.getName().toLowerCase().contains("tensorflow")) 
			{
				logger.info("Deleting " + file);
				file.delete();
			}
		}
	}

	/**
	 * Reading the content of the {@link #TFVERSIONFILE} indicating which native version of TensorFlow is installed
	 * @param root the root path of ImageJ
	 * @return the current TensorFlow version installed in ImageJ/lib
	 * @throws IOException in case the util file containing the native version number cannot be read (must not mean there is no usable native version)
	 */
	public static TensorFlowVersion readNativeVersionFile(String root) throws IOException 
	{
		if(getNativeVersionFile(root).exists()) 
		{
			Path path = getNativeVersionFile(root).toPath();
			final String versionstr = new String(Files.readAllBytes(path));
			final String[] parts = versionstr.split(",");
			if(parts.length >= 3) {
				String version = parts[1];
				boolean gpuSupport = parts[2].toLowerCase().equals("gpu");
				if(parts.length == 3) 
				{
					return new TensorFlowVersion(version, gpuSupport, null, null);
				}
				if(parts.length == 5) 
				{
					String cuda = parts[3];
					String cudnn = parts[4];
					return new TensorFlowVersion(version, gpuSupport, cuda, cudnn);
				}
			} 
			else 
			{
				throw new IOException("Content of " + path + " does not match expected format");
			}
		}
		// unknown version origin
		return new TensorFlowVersion(TensorFlow.version(), null, null, null);
	}

	/**
	 * Writing the content of the {@link #TFVERSIONFILE} indicating which native version of TensorFlow is installed
	 * @param root the root path of ImageJ
	 * @param platform the platform of the user (e.g. linux64, win64, macosx)
	 * @param version the installed TensorFlow version
	 */
	public static void writeNativeVersionFile(String root, String platform, TensorFlowVersion version) 
	{
		// create content
		StringBuilder content = new StringBuilder();
		content.append(platform);
		content.append(",");
		content.append(version.getVersionNumber());
		content.append(",");
		if(version.usesGPU().isPresent()) 
		{
			content.append(version.usesGPU().get() ? "GPU" : "CPU");
		} 
		else 
		{
			content.append("?");
		}
		if(version.getCompatibleCuDNN().isPresent() || version.getCompatibleCUDA().isPresent()) 
		{
			if(version.getCompatibleCUDA().isPresent()) 
			{
				content.append(",");
				content.append(version.getCompatibleCUDA().get());
			} 
			else 
			{
				content.append("?");
			}
			if(version.getCompatibleCuDNN().isPresent()) 
			{
				content.append(",");
				content.append(version.getCompatibleCuDNN().get());
			} 
			else 
			{
				content.append("?");
			}
		}
		//write content to file
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(getNativeVersionFile(root)))) 
		{
			writer.write(content.toString());
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}

	/**
	 * @param root the root path of ImageJ
	 * @return the crash File location indication if loading TensorFlow resulted in a JVM crash
	 */
	public static File getCrashFile(String root) 
	{
		return new File(getLibDir(root) + PLATFORM + File.separator + CRASHFILE);
	}

	/**
	 * @param root the root path of ImageJ
	 * @return the file indication which native TensorFlow version is currently installed
	 */
	public static File getNativeVersionFile(String root) 
	{
		return new File(getLibDir(root) + PLATFORM + File.separator + TFVERSIONFILE);
	}

	/**
	 * @param root the root path of ImageJ
	 * @return the directory where native libraries can be placed so that they are loadable from Java
	 */
	public static String getLibDir(String root) 
	{
		return root + File.separator + LIBDIR + File.separator;
	}

	/**
	 * @param root the root path of ImageJ
	 * @return the directory where native libraries can be placed so that the updater will move them into {@link #getLibDir(String)}
	 */
	public static String getUpdateLibDir(String root) 
	{
		return root + File.separator + UPDATEDIR + File.separator + LIBDIR + File.separator;
	}
}
