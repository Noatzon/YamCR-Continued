import boofcv.abst.feature.associate.AssociateDescription;
import boofcv.abst.feature.associate.ScoreAssociation;
import boofcv.abst.feature.detdesc.DetectDescribePoint;
import boofcv.alg.enhance.EnhanceImageOps;
import boofcv.alg.misc.ImageStatistics;
import boofcv.core.image.ConvertImage;
import boofcv.factory.feature.associate.ConfigAssociateGreedy;
import boofcv.factory.feature.associate.FactoryAssociation;
import boofcv.factory.feature.detdesc.FactoryDetectDescribe;
import boofcv.io.image.ConvertBufferedImage;
import boofcv.struct.feature.AssociatedIndex;
import boofcv.struct.feature.TupleDesc_F64;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import georegression.struct.point.Point2D_F64;
import org.ddogleg.struct.DogArray;
import org.ddogleg.struct.FastAccess;

import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ImageDesc {
    private static DetectDescribePoint<GrayF32, TupleDesc_F64> detDesc =
            FactoryDetectDescribe.surfStable(
                    StaticConfigs.getHessianConf(),
                    null,
                    null,
                    GrayF32.class
            );

    private static ScoreAssociation<TupleDesc_F64> scorer =
            FactoryAssociation.defaultScore(detDesc.getDescriptionType());
    private static ConfigAssociateGreedy configAssociateGreedy = new ConfigAssociateGreedy(true, 8);
    private static AssociateDescription<TupleDesc_F64> associate =
            FactoryAssociation.greedy(configAssociateGreedy, scorer);

    private AverageHash hash;
    private AverageHash flipped;
    private DogArray<TupleDesc_F64> desc = new DogArray<>(TupleDesc_F64::new);
    private ArrayList<Point2D_F64> points = new ArrayList<>(0);
    private int size;

    public ImageDesc(BufferedImage in, BufferedImage flip_in) {
        if (!AverageHash.isInitiated()) {
            AverageHash.init(2, 2);
        }
        hash = AverageHash.avgHash(in, 2, 2);
        if (flip_in != null) {
            flipped = AverageHash.avgHash(flip_in, 2, 2);
        }
        int histogram[] = new int[256];
        int transform[] = new int[256];
        GrayU8 img = ConvertBufferedImage.convertFromSingle(in, null, GrayU8.class);
        GrayU8 norm = img.createSameShape();
        // Updated method call to align with new parameters
        ImageStatistics.histogram(img, 0, histogram);
        EnhanceImageOps.equalize(histogram, transform);
        EnhanceImageOps.applyTransform(img, transform, norm);
        GrayF32 normf = new GrayF32(img.width, img.height);
        ConvertImage.convert(norm, normf);
        desc.reset();
        size = describeImage(normf, desc, points);
    }

    public ImageDesc(BufferedImage in) {
        this(in, null);
    }

    public ImageDesc(DogArray<TupleDesc_F64> d, ArrayList<Point2D_F64> p, AverageHash h) {
        desc = d;
        hash = h;
        points = p;
        size = p.size();
    }

    public void writeOut(DataOutputStream out) throws IOException {
        out.writeInt(size);
        for (int i = 0; i < size; i++) {
            TupleDesc_F64 f = desc.get(i);
            for (double val : f.data) { // Changed from f.value to f.data
                out.writeDouble(val);
            }
            Point2D_F64 pt = points.get(i);
            out.writeDouble(pt.x);
            out.writeDouble(pt.y);
        }
        hash.writeOut(out);
    }

    public static ImageDesc readIn(ByteBuffer buf) {
        int size = buf.getInt();
        ArrayList<Point2D_F64> points = new ArrayList<>(size);
        // Updated DogArray instantiation
        DogArray<TupleDesc_F64> descs = new DogArray<>(TupleDesc_F64::new);
        for (int i = 0; i < size; i++) {
            TupleDesc_F64 f = detDesc.createDescription();
            for (int j = 0; j < f.size(); j++) {
                f.data[j] = buf.getDouble();
            }
            descs.grow().setTo(f); // Corrected the method to use grow().setTo()
            points.add(new Point2D_F64(
                    buf.getDouble(), buf.getDouble()
            ));
        }
        AverageHash hash = AverageHash.readIn(buf);
        return new ImageDesc(descs, points, hash);
    }

    public static int describeImage(GrayF32 input, DogArray<TupleDesc_F64> descs, List<Point2D_F64> points) {
        detDesc.detect(input);
        int size = detDesc.getNumberOfFeatures();
        for (int i = 0; i < size; i++) {
            descs.grow().setTo(detDesc.getDescription(i)); // Corrected the method to use grow().setTo()
            points.add(detDesc.getLocation(i));
        }
        return size;
    }

    public double compareSURF(ImageDesc i2) {
        associate.setSource(desc);
        associate.setDestination(i2.desc);
        associate.associate();

        double max = Math.max(desc.size(), i2.desc.size());
        // Corrected to use FastAccess for matches
        FastAccess<AssociatedIndex> matches = associate.getMatches();
        double score = 0;
        for (int i = 0; i < matches.size(); i++) {
            AssociatedIndex match = matches.get(i);
            score += 1 - match.fitScore;
        }
        score = score / max;
        return score;
    }

    public double compareHash(ImageDesc i2) {
        return hash.match(i2.hash);
    }

    public double compareHashWithFlip(ImageDesc i2) {
        return Math.max(hash.match(i2.hash), flipped.match(i2.hash));
    }
}
