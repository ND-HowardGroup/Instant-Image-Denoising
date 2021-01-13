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

package net.image.denoising.util;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.scijava.app.StatusService;
import org.scijava.log.LogService;
import org.scijava.util.ByteArray;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Utility class for unpacking archives
 * TODO: generalize and move it to a place where others can use it as well
 *
 * @author Varun Mannam
 */
public final class UnpackUtil {

	private UnpackUtil(){}

	public static void unGZip(String tarGzFile, File output, String symLinkOutputDir, LogService log, StatusService status) throws IOException 
	{
		log("Unpacking " + tarGzFile + " to " + output, log , status);
		if (!output.exists()) 
		{
			output.mkdirs();
		}

		String tarFileName = tarGzFile.replace(".gz", "");
		FileInputStream instream = new FileInputStream(tarGzFile);
		GZIPInputStream ginstream = new GZIPInputStream(instream);
		FileOutputStream outstream = new FileOutputStream(tarFileName);
		byte[] buf = new byte[1024];
		int len;
		while ((len = ginstream.read(buf)) > 0) 
		{
			outstream.write(buf, 0, len);

		}
		ginstream.close();
		outstream.close();
		TarArchiveInputStream myTarFile = new TarArchiveInputStream(new FileInputStream(tarFileName));
		TarArchiveEntry entry;
		while ((entry = myTarFile.getNextTarEntry()) != null) 
		{
			if (entry.isSymbolicLink() || entry.isLink()) 
			{
				Path source = new File(output, entry.getName()).toPath();
				Path target = new File(symLinkOutputDir, entry.getLinkName()).toPath();
				deleteIfExists(source.toAbsolutePath().toString());
				if (entry.isSymbolicLink()) 
				{
					log("Creating symbolic link: " + source + " -> " + target, log, status);
					Files.createSymbolicLink(source, target);
				} 
				else 
				{
					log("Creating link: " + source + " -> " + target, log, status);
					Files.createLink(source, target);
				}
			} 
			else 
			{
				File outEntry = new File(output + "/" + entry.getName());
				if (!outEntry.getParentFile().exists()) 
				{
					outEntry.getParentFile().mkdirs();
				}
				if (outEntry.isDirectory())
				{
					continue;
				}
				log("Writing " + outEntry, log, status);
				final byte[] buf1 = new byte[64 * 1024];
				final int size1 = (int) entry.getSize();
				int len1 = 0;
				try (final FileOutputStream outEntryStream = new FileOutputStream(outEntry)) 
				{
					while (true) 
					{
						status.showStatus(len1, size1, "Unpacking " + entry.getName());
						final int r = myTarFile.read(buf1, 0, buf1.length);
						if (r < 0) break; // end of entry
						len1 += r;
						outEntryStream.write(buf1, 0, r);
					}
				}
			}
		}
		status.clearStatus();
		myTarFile.close();
		File tarFile = new File(tarFileName);
		tarFile.delete();
	}

	public static void unZip(String zipFile, File output, LogService log, StatusService status) throws IOException 
	{
		log("Unpacking " + zipFile + " to " + output, log, status);
		output.mkdirs();
		try (final ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile))) 
		{
			unZip(zis, output, log, status);
		}
	}

	public static void unZip(File output, ByteArray byteArray, LogService log, StatusService status) throws IOException 
	{
		// Extract the contents of the compressed data to the model cache.
		final ByteArrayInputStream bais = new ByteArrayInputStream(//
				byteArray.getArray(), 0, byteArray.size());
		output.mkdirs();
		try (final ZipInputStream zis = new ZipInputStream(bais)) 
		{
			unZip(zis, output, log, status);
		}
	}

	private static void unZip(ZipInputStream zis, File output, LogService log, StatusService status) throws IOException 
	{
		final byte[] buf = new byte[64 * 1024];
		while (true) {
			final ZipEntry entry = zis.getNextEntry();
			if (entry == null) break; // All done!
			final String name = entry.getName();
			log("Unpacking " + name, log, status);
			final File outFile = new File(output, name);
			if (entry.isDirectory()) {
				outFile.mkdirs();
			}
			else 
			{
				final int size = (int) entry.getSize();
				int len = 0;
				try (final FileOutputStream out = new FileOutputStream(outFile)) 
				{
					while (true) 
					{
						status.showStatus(len, size, "Unpacking " + name);
						final int r = zis.read(buf);
						if (r < 0) break; // end of entry
						len += r;
						out.write(buf, 0, r);
					}
				}
			}
		}
		status.clearStatus();
	}

	private static void deleteIfExists(String filePath) 
	{
		File file = new File(filePath);
		if(file.exists()) 
		{
			file.delete();
		}
	}

	private static void log(String msg, LogService log, StatusService status) 
	{
		log.info(msg);
		status.showStatus(msg);
	}

}
