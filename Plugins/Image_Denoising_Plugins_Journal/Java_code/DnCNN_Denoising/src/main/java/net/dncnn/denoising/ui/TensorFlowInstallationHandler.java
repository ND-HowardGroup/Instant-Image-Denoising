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

import org.scijava.app.AppService;
import org.scijava.app.StatusService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.plugin.SciJavaPlugin;

import net.dncnn.denoising.util.TensorFlowUtil;
import net.dncnn.denoising.util.UnpackUtil;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * This class handles instances of {@link DownloadableTensorFlowVersion}.
 * It is responsible for downloading, caching, unpacking, and installing them.
 *
 * @author Varun Mannam
 * 
 */
class TensorFlowInstallationHandler 
{

	@Parameter
	private AppService appService;

	@Parameter
	private LogService logService;
	
	@Parameter
	private StatusService statusService;

	private static final String DOWNLOADDIR = "downloads/";

	/**
	 * Checks for a specific version whether it is downloaded and installed.
	 * @param version the version which will be checked
	 */
	void updateCacheStatus(final DownloadableTensorFlowVersion version) 
	{
		if(version.getURL() == null) return;
		if (version.getURL().getProtocol().equalsIgnoreCase("file")) 
		{
			version.setCached(version.getURL().getFile());
		} 
		else 
		{
			String path = getDownloadDir() + getNameFromURL(version.getURL());
			if (new File(path).exists()) 
			{
				version.setCached(path);
			}
		}
	}

	private String getNameFromURL(URL url) 
	{
		String[] parts = url.getPath().split("/");
		if(parts.length > 0) return parts[parts.length - 1];
		else return null;
	}

	/**
	 * Activates a specific TensorFlow library which will then be used after restarting the application.
	 * @param version the version being activated
	 * @throws IOException thrown in case downloading the version failed or the version could not be unpacked
	 */
	void activateVersion(DownloadableTensorFlowVersion version) throws IOException 
	{
		if (!version.isCached()) 
		{
			downloadVersion(version.getURL());
		}
		updateCacheStatus(version);
		if (!version.isActive()) 
		{
			installVersion(version);
		}
	}

	private void downloadVersion(URL url) throws IOException
	{
		createDownloadDir();
		String filename = url.getFile().substring(url.getFile().lastIndexOf("/") + 1);
		String localFile = getDownloadDir() + filename;
		logService.info("Downloading " + url + " to " + localFile);
		InputStream in = url.openStream();
		Files.copy(in, Paths.get(localFile), StandardCopyOption.REPLACE_EXISTING);
	}

	private void createDownloadDir() 
	{
		File downloadDir = new File(getDownloadDir());
		if (!downloadDir.exists()) 
		{
			downloadDir.mkdirs();
		}
	}

	private void installVersion(DownloadableTensorFlowVersion version) throws IOException 
	{

		logService.info("Installing " + version);

		File outputDir = new File(TensorFlowUtil.getUpdateLibDir(getRoot()) + version.getPlatform() + File.separator);

		if (version.getLocalPath().contains(".zip")) 
		{
			UnpackUtil.unZip(version.getLocalPath(), outputDir, logService, statusService);
			TensorFlowUtil.writeNativeVersionFile(getRoot(), version.getPlatform(), version);
		}
		else if (version.getLocalPath().endsWith(".tar.gz")) 
		{
			String symLinkOutputDir = TensorFlowUtil.getLibDir(getRoot()) + version.getPlatform() + File.separator;;
			UnpackUtil.unGZip(version.getLocalPath(), outputDir, symLinkOutputDir, logService, statusService);
			TensorFlowUtil.writeNativeVersionFile(getRoot(), version.getPlatform(), version);
		}

		if (version.getLocalPath().endsWith(".jar")) 
		{
			// using default JAR version. nothing to do.
			logService.info("Using default JAR TensorFlow version.");
		}

		statusService.clearStatus();

		TensorFlowUtil.getCrashFile(getRoot()).delete();
	}

	private String getRoot() {
		return appService.getApp().getBaseDirectory().getAbsolutePath();
	}

	private String getDownloadDir() {
		return appService.getApp().getBaseDirectory().getAbsolutePath() + File.separator + DOWNLOADDIR;
	}

}
