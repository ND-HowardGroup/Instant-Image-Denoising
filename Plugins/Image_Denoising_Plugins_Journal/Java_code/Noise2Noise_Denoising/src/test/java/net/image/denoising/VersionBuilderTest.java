package net.image.denoising;

import org.junit.Test;
import org.tensorflow.TensorFlow;

import net.image.denoising.TensorFlowVersion;
import net.image.denoising.util.TensorFlowUtil;

import java.net.URL;

import static org.junit.Assert.assertEquals;

public class VersionBuilderTest 
{
	@Test
	public void testJARVersion() 
	{
		URL source = TensorFlow.class.getResource("TensorFlow.class");
		TensorFlowVersion version = TensorFlowUtil.getTensorFlowJARVersion(source);
		assertEquals(TensorFlow.version(), version.getVersionNumber());
	}
}
