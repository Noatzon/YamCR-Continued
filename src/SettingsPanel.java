import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class SettingsPanel extends JPanel{
	private static final long serialVersionUID = 1L;

	public static boolean RECOG_EVERY_FRAME = true;

	public static int RECOG_THRESH = 27;

	public SettingsPanel()
	{
		super();
		setLayout(new FlowLayout());

		JCheckBox recog = new JCheckBox("Only trigger recognition manually",!RECOG_EVERY_FRAME);
		recog.setToolTipText("Only trigger recognition manually");
		recog.addActionListener(new ActionListener(){
			public void actionPerformed(ActionEvent e) {
				RECOG_EVERY_FRAME = !recog.isSelected();
			}
		});
		add(recog);

		JSlider thresh = new JSlider(JSlider.HORIZONTAL, 0, 100, RECOG_THRESH);
		JLabel score = new JLabel("Score Threshold: "+RECOG_THRESH);
		thresh.addChangeListener(new ChangeListener(){
			@Override
			public void stateChanged(ChangeEvent arg0) {
				RECOG_THRESH = thresh.getValue();
				score.setText("Score Threshold: "+RECOG_THRESH);
			}
		});
		add(score);
		add(thresh);

		JComboBox<RecognitionStrategy> stratSelect = 
			new JComboBox<RecognitionStrategy>(StrategySelect.getStrats());
		stratSelect.setSelectedIndex(0);
		stratSelect.addActionListener(new ActionListener()
		{
			@SuppressWarnings("unchecked")
			public void actionPerformed(ActionEvent e) {
				JComboBox<RecognitionStrategy> cb = 
					(JComboBox<RecognitionStrategy>)e.getSource();
				RecogApp.INSTANCE.doSetStrat(cb.getItemAt(cb.getSelectedIndex()));
			}
		});
		add(stratSelect);

		JButton selectCam = new JButton("Reselect webcam");
		selectCam.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				RecogApp.INSTANCE.doSetWebcam();
			}
		});
		add(selectCam);
		
		JButton loadSelected = new JButton("Load Selected");
		loadSelected.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				RecogApp.select.loadSelected();
			}
		});
		add(loadSelected);
		
		JButton unloadAll = new JButton("Unload all");
		unloadAll.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				RecogApp.select.unloadAll();
			}
		});
		add(unloadAll);
		
		JButton launchPopout = new JButton("Card Preview");
		launchPopout.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				new PopoutCardWindow();
			}
		});
		add(launchPopout);
		
		JButton launchSetGen = new JButton("Bulk Generate Sets");
		launchSetGen.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				SetGenerator.bulkGenSets();
			}
		});
		add(launchSetGen);
		
		JButton launchDeckGen = new JButton("Deck Generator");
		launchDeckGen.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				new DeckGenerator();
			}
		});
		add(launchDeckGen);
		
		JButton toggleSetPanel = new JButton("Refresh Set Listing");
		toggleSetPanel.addActionListener(new ActionListener(){
			@Override
			public void actionPerformed(ActionEvent e) {
				RecogApp.select.refresh();
			}
		});
		add(toggleSetPanel);
	}
}
