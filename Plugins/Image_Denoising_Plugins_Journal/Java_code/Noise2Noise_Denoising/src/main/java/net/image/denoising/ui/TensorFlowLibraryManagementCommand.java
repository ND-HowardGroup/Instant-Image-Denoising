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

package net.image.denoising.ui;

import net.imagej.ImageJ;
import net.image.denoising.TensorFlowService;
import net.image.denoising.TensorFlowVersion;
import net.image.denoising.util.TensorFlowUtil;
import net.imagej.updater.util.UpdaterUtil;
import org.scijava.Context;
import org.scijava.app.AppService;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * This command provides an interface for activating a specific Java TensorFlow version.
 * The user can choose between the version included via maven and archived native library versions
 * which can be downloaded from the Google servers.
 *
 * @author Varun Mannam
 * 
 */
@Plugin(type = Command.class, menuPath = "Edit>Options>TensorFlow...")
public class TensorFlowLibraryManagementCommand implements Command 
{

	@Parameter
	private TensorFlowService tensorFlowService;

	@Parameter
	private AppService appService;

	@Parameter
	private LogService logService;

	@Parameter
	private Context context;

	DownloadableTensorFlowVersion currentVersion;
	List<DownloadableTensorFlowVersion> availableVersions = new ArrayList<>();

	String platform = UpdaterUtil.getPlatform();

	private TensorFlowInstallationHandler installationHandler;

	@Override
	public void run() 
	{
		tensorFlowService.loadLibrary();
		installationHandler = new TensorFlowInstallationHandler();
		context.inject(installationHandler);
		final TensorFlowLibraryManagementFrame frame = new TensorFlowLibraryManagementFrame(tensorFlowService, installationHandler);
		frame.init();
		initAvailableVersions();
		frame.updateChoices(availableVersions);
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setMinimumSize(new Dimension(0, 200));
		frame.setVisible(true);

	}

	private void initAvailableVersions() 
	{
		initCurrentVersion();
		if(currentVersion != null) addAvailableVersion(currentVersion);
		addAvailableVersion(getTensorFlowJARVersion());
		AvailableTensorFlowVersions.get().forEach(version -> addAvailableVersion(version));
	}

	private void initCurrentVersion() 
	{
		TensorFlowVersion current = tensorFlowService.getTensorFlowVersion();
		if(current != null) {
			currentVersion = new DownloadableTensorFlowVersion(current);
			currentVersion.setPlatform(platform);
			currentVersion.setActive(true);
		}
	}

	private DownloadableTensorFlowVersion getTensorFlowJARVersion() 
	{
		DownloadableTensorFlowVersion version = new DownloadableTensorFlowVersion(TensorFlowUtil.versionFromClassPathJAR());
		version.setPlatform(platform);
		version.setURL(TensorFlowUtil.getTensorFlowJAR());
		return version;
	}

	private void addAvailableVersion(DownloadableTensorFlowVersion version) 
	{
		if(!version.getPlatform().equals(platform)) return;
		installationHandler.updateCacheStatus(version);
		for (DownloadableTensorFlowVersion other : availableVersions) 
		{
			if (other.equals(version)) 
			{
				other.harvest(version);
				return;
			}
		}
		availableVersions.add(version);
	}

	public static void main(String... args) 
	{
		ImageJ ij = new ImageJ();
		ij.ui().showUI();
		ij.command().run(TensorFlowLibraryManagementCommand.class, true);
	}
}
