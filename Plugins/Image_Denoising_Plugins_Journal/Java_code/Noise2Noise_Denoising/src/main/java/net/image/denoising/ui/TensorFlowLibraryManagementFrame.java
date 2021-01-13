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

import net.image.denoising.TensorFlowService;
import net.miginfocom.swing.MigLayout;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.*;
import java.util.List;

/**
 * The UI part of the {@link TensorFlowLibraryManagementCommand}
 *
 * @author Varun Mannam
 * 
 */
class TensorFlowLibraryManagementFrame extends JFrame 
{

	private static final long serialVersionUID = 1L;

	private static final Color LIST_BACKGROUND_COLOR = new Color(250,250,250);
	private static final String NOFILTER = "-";

	private TensorFlowService tensorFlowService;
	private TensorFlowInstallationHandler installationHandler;

	private JComboBox<String> gpuChoiceBox;
	private JComboBox<String> cudaChoiceBox;
	private JComboBox<String> tfChoiceBox;
	private JPanel installPanel;
	private JTextArea status;

	private List<DownloadableTensorFlowVersion> availableVersions = new ArrayList<>();
	private List<JRadioButton> buttons = new ArrayList<>();

	TensorFlowLibraryManagementFrame(final TensorFlowService tensorFlowInstallationService, final TensorFlowInstallationHandler installationHandler) {
		super("TensorFlow library version management");
		this.tensorFlowService = tensorFlowInstallationService;
		this.installationHandler = installationHandler;
	}

	public void init() 
	{
		JPanel panel = new JPanel();
		panel.setLayout(new MigLayout("height 400, wmax 600"));
		panel.add(new JLabel("Please select the TensorFlow version you would like to install."), "wrap");
		panel.add(createFilterPanel(), "wrap, span, align right");
		panel.add(createInstallPanel(), "wrap, span, grow");
		panel.add(createStatus(), "span, grow");
		setContentPane(panel);
	}

	private Component createFilterPanel() 
	{
		gpuChoiceBox = new JComboBox<>();
		gpuChoiceBox.addItem(NOFILTER);
		gpuChoiceBox.addItem("GPU");
		gpuChoiceBox.addItem("CPU");
		gpuChoiceBox.addActionListener((e) -> updateFilter());
		cudaChoiceBox = new JComboBox<>();
		cudaChoiceBox.addActionListener((e) -> updateFilter());
		tfChoiceBox = new JComboBox<>();
		tfChoiceBox.addActionListener((e) -> updateFilter());
		JPanel panel = new JPanel();
		panel.setLayout(new MigLayout());
		panel.add(makeLabel("Filter by.."));
		panel.add(makeLabel("Mode: "));
		panel.add(gpuChoiceBox);
		panel.add(makeLabel("CUDA: "));
		panel.add(cudaChoiceBox);
		panel.add(makeLabel("TensorFlow: "));
		panel.add(tfChoiceBox);
		return panel;
	}

	private Component createStatus() 
	{
		status = new JTextArea();
		status.setBorder(BorderFactory.createEmptyBorder());
		status.setEditable(false);
		status.setWrapStyleWord(true);
		status.setLineWrap(true);
		status.setOpaque(false);
		return status;
	}

	private void updateFilter() 
	{
		installPanel.removeAll();
		buttons.forEach(btn -> {
			if(filter("", gpuChoiceBox, btn)) return;
			if(filter("CUDA ", cudaChoiceBox, btn)) return;
			if(filter("TF ", tfChoiceBox, btn)) return;
			installPanel.add(btn);
		});
		installPanel.revalidate();
		installPanel.repaint();
	}

	private boolean filter(String title, JComboBox<String> choiceBox, JRadioButton btn) 
	{
		if(choiceBox.getSelectedItem().toString().equals(NOFILTER)) return false;
		return !btn.getText().contains(title + choiceBox.getSelectedItem().toString());
	}

	private JLabel makeLabel(String s) 
	{
		JLabel label = new JLabel(s);
		label.setHorizontalAlignment(SwingConstants.RIGHT);
		label.setHorizontalTextPosition(SwingConstants.RIGHT);
		return label;
	}

	private Component createInstallPanel() 
	{
		installPanel = new JPanel(new MigLayout("flowy"));
		JScrollPane scroll = new JScrollPane(installPanel);
		scroll.setBorder(BorderFactory.createEmptyBorder());
		installPanel.setBackground(LIST_BACKGROUND_COLOR);
		return scroll;
	}

	void updateChoices(List<DownloadableTensorFlowVersion> availableVersions) 
	{
		availableVersions.sort(Comparator.comparing(DownloadableTensorFlowVersion::getComparableTFVersion).reversed());
		this.availableVersions = availableVersions;
		updateCUDAChoices();
		updateTFChoices();
		ButtonGroup versionGroup = new ButtonGroup();
		installPanel.removeAll();
		for( DownloadableTensorFlowVersion version : availableVersions) 
		{
			JRadioButton btn = new JRadioButton(version.toString());
			btn.setToolTipText(version.getOriginDescription());
			btn.setSelected(version.isActive());
			btn.setOpaque(false);
			versionGroup.add(btn);
			buttons.add(btn);
			btn.addActionListener(e -> {
				if(btn.isSelected()) {
					new Thread(() -> activateVersion(version)).start();
				}
			});
		}
		updateFilter();
		updateStatus();
	}

	private void updateStatus() {
		status.setText(tensorFlowService.getStatus().getInfo());
		if(!tensorFlowService.getStatus().isLoaded()) 
		{
			status.setForeground(Color.red);
		} 
		else 
		{
			status.setForeground(Color.black);
		}
	}

	private void activateVersion(DownloadableTensorFlowVersion version) 
	{
		if(version.isActive()) 
		{
			System.out.println("[WARNING] Cannot activate version, already active: " + version);
			return;
		}
		showWaitMessage();
		try 
		{
			installationHandler.activateVersion(version);
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
			Object[] options = {"Yes",
					"No",
					"Cancel"};
			int choice = JOptionPane.showOptionDialog(this,
					"Error while unpacking library file " + version.getLocalPath() + ".\nShould it be downloaded again?",
					"Unpacking library error",
					JOptionPane.YES_NO_CANCEL_OPTION,
					JOptionPane.QUESTION_MESSAGE,
					null,
					options,
					options[0]);
			if(choice == 0) 
			{
				version.discardCache();
				try 
				{
					installationHandler.activateVersion(version);
				} 
				catch (IOException e1) 
				{
					e1.printStackTrace();
				}
			} 
			else 
			{
				status.setText("Installation failed: " + e.getClass().getName() + ": " + e.getMessage());
				return;
			}
		}
		status.setText("");
		JOptionPane.showMessageDialog(null,
				"Installed selected TensorFlow version. Please restart Fiji to load it.",
				"Please restart",
				JOptionPane.PLAIN_MESSAGE);
		dispose();
	}

	private void showWaitMessage() 
	{
		status.setForeground(Color.black);
		status.setText("Please wait..");
	}

	private void updateCUDAChoices() 
	{
		Set<String> choices = new LinkedHashSet<>();
		for(DownloadableTensorFlowVersion version :availableVersions) 
		{
			if(version.getCompatibleCUDA().isPresent())
				choices.add(version.getCompatibleCUDA().get());
		}
		cudaChoiceBox.removeAllItems();
		cudaChoiceBox.addItem(NOFILTER);
		for(String choice : choices) {
			cudaChoiceBox.addItem(choice);
		}
	}

	private void updateTFChoices() 
	{
		Set<String> choices = new LinkedHashSet<>();
		for(DownloadableTensorFlowVersion version :availableVersions) 
		{
			if(version.getVersionNumber() != null) choices.add(version.getVersionNumber());
		}
		tfChoiceBox.removeAllItems();
		tfChoiceBox.addItem(NOFILTER);
		for(String choice : choices) {
			tfChoiceBox.addItem(choice);
		}
	}

}
