import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamLockException;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.UIManager;

public class RecogApp extends JFrame implements KeyListener {
    private static final long serialVersionUID = 1L;
    private static RecognitionStrategy strat;
    private static AreaRecognitionStrategy areaStrat;
    private static RecognitionCanvas canvas;
    private static SettingsPanel settings;
    private static SetLoadPanel select;
    private static OperationBar task;
    private static CollectionManagerWindow collection;
    private static BrowserSourceWindow browserSource;
    public static RecogApp INSTANCE;
    private static CollectionData collectionData;  // Added collectionData instance

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }

        SavedConfig.init();
        collection = new CollectionManagerWindow();
        browserSource = new BrowserSourceWindow();
        collectionData = new CollectionData(false);  // Initialize the collection data
        new RecogApp();
    }

    public RecogApp() {
        super("Yet Another Magic Card Recognizer");
        INSTANCE = this;
        BorderLayout bl = new BorderLayout();
        this.setLayout(bl);
        task = new OperationBar();
        strat = SavedConfig.getStrat();
        SetListing.init();
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        Webcam w = WebcamUtils.getPreferredElseChooseWebcam();
        if (w == null) {
            System.exit(1);
        }

        JPanel right = new JPanel();
        right.setLayout(new BorderLayout());
        areaStrat = SavedConfig.getAreaStrat();
        canvas = new RecognitionCanvas(w, areaStrat);
        ImageIcon ico = new ImageIcon("res/YamCR.png");
        this.setIconImage(ico.getImage());
        JScrollPane scroll = new JScrollPane();
        select = new SetLoadPanel(strat);
        scroll.setViewportView(select);
        scroll.getVerticalScrollBar().setUnitIncrement(16);
        this.add(canvas, BorderLayout.CENTER);
        this.add(right, BorderLayout.EAST);
        this.add(task, BorderLayout.SOUTH);
        settings = new SettingsPanel();
        right.add(settings, BorderLayout.NORTH);
        right.add(scroll, BorderLayout.CENTER);
        right.setPreferredSize(new Dimension(320, canvas.getHeight()));
        this.pack();
        this.setVisible(true);

        try {
            w.open(true);
        } catch (WebcamLockException var7) {
            JOptionPane.showMessageDialog((Component)null, "Webcam already in use. Exiting.");
            System.exit(0);
        }

        canvas.getCanvas().addKeyListener(this);
        this.doSetBackground();

        while(true) {
            do {
                canvas.draw();
                task.repaint();
            } while(!SettingsPanel.RECOG_EVERY_FRAME);

            this.doRecog();
        }
    }

    public BufferedImage getCardImageFromID(String id) {
        return strat.resolveArt(id);
    }

    public AreaRecognitionStrategy getAreaStrategy() {
        return areaStrat;
    }

    public void doSetStrat(RecognitionStrategy strategy) {
        if (task.setTask("Reloading strategy", 1)) {
            strat.clear();
            strat = strategy;
            SavedConfig.setPreferredStrat(strat);
            task.progressTask();
        }

        select.refresh();
        settings.resetStratSelector(strat);
    }

    public void doSetAreaStrat(AreaRecognitionStrategy strategy) {
        if (task.setTask("Reloading strategy", 1)) {
            areaStrat = strategy;
            SavedConfig.setPreferredAreaStrat(areaStrat);
            canvas.setAreaStrat(areaStrat);
            task.progressTask();
        }

        settings.resetAreaStratSelector(areaStrat);
    }

    public void doSetWebcam() {
        synchronized(canvas) {
            Webcam w = WebcamUtils.chooseWebcam();
            if (w != null) {
                canvas.setWebcam(w);
                this.pack();
            }

        }
    }

    public void doSetBackground() {
        CardBoundingBoxFinder.adaptBackground(canvas.getWebcam());
    }

    public void doRecog() {
        synchronized(canvas) {
            BufferedImage img = canvas.lastDrawn();
            doRecog(img);
        }
    }

    public static void doRecog(BufferedImage img) {
        if (!task.isOperating() && img != null) {
            synchronized(strat) {
                AreaRecognitionStrategy areaSet = SavedConfig.getAreaStrat();
                ArrayList<MatchResult> matches = areaSet.recognize(canvas.lastDrawn(), strat);
                collection.addRecognizedCards(matches);

                for (MatchResult res : matches) {
                    collectionData.addEntry(new CollectionEntry(res));  // Add recognized card to collection data
                }

                if (!matches.isEmpty()) {
                    MatchResult res = matches.get(0);
                    canvas.setLastResult(res);
                    PopoutCardWindow.setDisplay(res.getData());
                    browserSource.getServer().update(res.getData(), true);
                }
            }
        }
    }

    public Graphics getCanvasGraphics() {
        return canvas.lastDrawn().getGraphics();
    }

    public OperationBar getOpBar() {
        return task;
    }

    public SetLoadPanel getLoader() {
        return select;
    }

    public CollectionManagerWindow getCollectionWindow() {
        return collection;
    }

    public BrowserSourceWindow getBrowserSourceWindow() {
        return browserSource;
    }

    public void keyPressed(KeyEvent arg0) {
        this.doRecog();
    }

    public void keyReleased(KeyEvent arg0) {
    }

    public void keyTyped(KeyEvent arg0) {
    }
}
