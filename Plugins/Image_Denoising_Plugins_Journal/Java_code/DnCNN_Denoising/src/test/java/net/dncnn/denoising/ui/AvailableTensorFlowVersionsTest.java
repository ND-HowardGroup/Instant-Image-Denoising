package net.dncnn.denoising.ui;

import org.junit.Test;

import net.dncnn.denoising.ui.AvailableTensorFlowVersions;
import net.dncnn.denoising.ui.DownloadableTensorFlowVersion;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class AvailableTensorFlowVersionsTest 
{

	@Test
	public void testURLs() throws IOException 
	{
		List<DownloadableTensorFlowVersion> versions = AvailableTensorFlowVersions.get();
		for (DownloadableTensorFlowVersion version : versions) 
		{
			System.out.println("Testing " + version.getURL());
			HttpURLConnection huc =  (HttpURLConnection)  version.getURL().openConnection();
			huc.setRequestMethod("HEAD");
			huc.connect();
			assertEquals(HttpURLConnection.HTTP_OK, huc.getResponseCode());
		}
	}

}
