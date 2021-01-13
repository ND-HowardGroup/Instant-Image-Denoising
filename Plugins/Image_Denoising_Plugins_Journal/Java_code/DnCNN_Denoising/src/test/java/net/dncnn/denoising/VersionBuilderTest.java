package net.dncnn.denoising;

import org.junit.Test;
import org.tensorflow.TensorFlow;

import net.dncnn.denoising.TensorFlowVersion;
import net.dncnn.denoising.util.TensorFlowUtil;

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
