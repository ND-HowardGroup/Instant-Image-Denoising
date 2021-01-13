package net.image.denoising.ui;

import org.junit.Test;

import net.image.denoising.TensorFlowVersion;
import net.image.denoising.ui.DownloadableTensorFlowVersion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class TensorFlowVersionTest 
{

	@Test
	public void compareVersions() 
	{
		TensorFlowVersion v1 = new TensorFlowVersion("1.4.0", true, null, null);
		TensorFlowVersion v2 = new TensorFlowVersion("1.4.0", true, null, null);
		assertEquals(v1, v2);

		v1 = new TensorFlowVersion("1.4.0", true, null, null);
		v2 = new TensorFlowVersion("1.4.0", false, null, null);
		assertNotEquals(v1, v2);

		v1 = new TensorFlowVersion("1.4.0", false, null, null);
		v2 = new TensorFlowVersion("1.4.0", false, "x.x", "y.y");
		assertEquals(v1, v2);

		DownloadableTensorFlowVersion v3 = new DownloadableTensorFlowVersion("1.4.0", false);
		assertNotEquals(v3, v1);
		assertNotEquals(v1, v3);

	}

}
