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

package net.dncnn.denoising;

/**
 * The loading status of the TensorFlow native library.
 * @author Varun Mannam
 * 
 */
public final class TensorFlowLibraryStatus 
{

	private final boolean loaded;
	private final boolean crashed;
	private final boolean failed;
	private final String info;

	static TensorFlowLibraryStatus notLoaded() {
		return new TensorFlowLibraryStatus(false, false, false, "");
	}

	static TensorFlowLibraryStatus loaded(final String info) {
		return new TensorFlowLibraryStatus(true, false, false, info);
	}

	static TensorFlowLibraryStatus crashed(final String info) {
		return new TensorFlowLibraryStatus(false, true, false, info);
	}

	static TensorFlowLibraryStatus failed(final String info) {
		return new TensorFlowLibraryStatus(false, false, true, info);
	}

	private TensorFlowLibraryStatus(final boolean loaded, final boolean crashed, final boolean failed,
			final String info) {
		this.loaded = loaded;
		this.crashed = crashed;
		this.failed = failed;
		this.info = info;
	}

	/**
	 * @return whether there was an attempt to load the TensorFlow library
	 */
	public boolean triedLoading() {
		return loaded || crashed || failed;
	}

	/**
	 * @return whether the TensorFlow library was successfully loaded
	 */
	public boolean isLoaded() {
		return loaded;
	}

	/**
	 * @return whether the TensorFlow library failed to load
	 */
	public boolean isFailed() {
		return failed;
	}

	/**
	 * @return whether a crash log file was found indicating that the TensorFlow library crashed the JVM during {@link TensorFlowService#loadLibrary()}
	 */
	public boolean isCrashed() {
		return crashed;
	}

	/**
	 * @return a message describing the failed or successful loading attempt of the TensorFlow Library
	 */
	public String getInfo() {
		return info;
	}
}
