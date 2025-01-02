import boofcv.alg.InputSanityCheck;
import boofcv.alg.background.stationary.BackgroundStationaryBasic;
import boofcv.alg.filter.binary.BinaryImageOps;
import boofcv.alg.filter.binary.Contour;
import boofcv.alg.filter.binary.GThresholdImageOps;
import boofcv.factory.background.ConfigBackgroundBasic;
import boofcv.factory.background.FactoryBackgroundModel;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.ConfigLength;
import boofcv.struct.ConnectRule;
import boofcv.struct.image.GrayS32;
import boofcv.struct.image.GrayU8;
import boofcv.struct.image.ImageType;
import com.github.sarxos.webcam.Webcam;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

class CardBoundingBoxFinder {
    private static final float INITIAL_LEARN_RATE = 0.01f;
    private static final ImageType<GrayU8> imageType = ImageType.single(GrayU8.class);
    private static final BackgroundStationaryBasic<GrayU8> background =
            FactoryBackgroundModel.stationaryBasic(new ConfigBackgroundBasic(35, INITIAL_LEARN_RATE), imageType);

    public static ArrayList<ContourBoundingBox> process(BufferedImage in, boolean removeBackground) {
        ArrayList<ContourBoundingBox> bounds = new ArrayList<>();
        GrayU8 img = ConvertBufferedImage.convertFromSingle(in, null, GrayU8.class);
        GrayU8 binary = img.createSameShape();
        if (removeBackground) {
            background.segment(img, binary);
            binary = BinaryImageOps.dilate8(binary, 5, null);
        } else {
            GThresholdImageOps.localMean(img, binary, ConfigLength.fixed(20), 1.0, true, null, null, null);
        }

        GrayU8 filtered = BinaryImageOps.erode8(binary, 2, null);
        GrayS32 label = new GrayS32(img.width, img.height);

        double imgArea = img.height * img.width;

        List<Contour> contours = BinaryImageOps.contour(filtered, ConnectRule.EIGHT, label);
        bounds.clear();
        for (Contour contour : contours) {
            ContourBoundingBox bb = new ContourBoundingBox(contour.external);
            double ratio = bb.area() / imgArea;
            if (ratio > 0.005 && ratio < 0.5 && bb.isRoughlyRectangular()) {
                bounds.add(bb);
            }
        }
        return bounds;
    }

    /**
     * Adapted from BoofCV source
     */
    public static void mask(GrayU8 source, GrayU8 mask, GrayU8 output) {
        InputSanityCheck.checkSameShape(source, mask);
        if (output == null || output.width != source.width || output.height != source.height) {
            output = new GrayU8(source.width, source.height);
        }
        for (int y = 0; y < source.height; y++) {
            int indexA = source.startIndex + y * source.stride;
            int indexB = mask.startIndex + y * mask.stride;
            int indexOut = output.startIndex + y * output.stride;

            int end = indexA + source.width;
            for (; indexA < end; indexA++, indexB++, indexOut++) {
                byte srcval = source.data[indexA];
                byte mskval = mask.data[indexB];
                output.data[indexOut] = mskval == (byte) 1 ? srcval : (byte) 0;
            }
        }
    }

    public static void adaptBackground(BufferedImage frame) {
        GrayU8 img = ConvertBufferedImage.convertFromSingle(frame, null, GrayU8.class);
        background.updateBackground(img);
    }

    public static void adaptBackground(Webcam w) {
        background.reset();
        background.setLearnRate(INITIAL_LEARN_RATE);
        BufferedImage frame = w.getImage();
        GrayU8 img = null;
        for (int i = 0; i < 10; i++) {
            img = ConvertBufferedImage.convertFromSingle(frame, img, GrayU8.class);
            background.updateBackground(img);
            frame = w.getImage();
        }
    }
}
