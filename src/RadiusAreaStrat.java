
import boofcv.factory.feature.detect.line.ConfigLineRansac;
import boofcv.struct.image.GrayS16;
import boofcv.struct.image.GrayU8;
import org.ddogleg.fitting.modelset.ransac.Ransac;
import boofcv.abst.feature.detect.line.DetectLineSegment;
import boofcv.abst.feature.detect.line.DetectLineSegmentsGridRansac;
import boofcv.abst.filter.derivative.ImageGradient;
import boofcv.alg.feature.detect.line.ConnectLinesGrid;
import boofcv.alg.feature.detect.line.GridRansacLineDetector;
import boofcv.alg.feature.detect.line.gridline.Edgel;
import boofcv.alg.feature.detect.line.gridline.GridLineModelDistance;
import boofcv.alg.feature.detect.line.gridline.GridLineModelFitter;
import boofcv.alg.feature.detect.line.gridline.ImplGridRansacLineDetector_S16;
import boofcv.alg.filter.blur.GBlurImageOps;
import boofcv.factory.filter.derivative.FactoryDerivative;
import boofcv.io.image.ConvertBufferedImage;
import georegression.fitting.line.ModelManagerLinePolar2D_F32;
import georegression.struct.line.LinePolar2D_F32;
import georegression.struct.line.LineSegment2D_F32;
import georegression.struct.point.Point2D_F32;
import georegression.struct.point.Point2D_I32;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class RadiusAreaStrat extends AreaRecognitionStrategy
{

    static final int maxLines = 10;
    static final int collateFrames = 10;

    static final int BUCKET_ANGLE = 3;
    static final int NUM_BUCKETS = 180 / BUCKET_ANGLE;

    private Point2D_F32 center = new Point2D_F32(320, 240);
    private double radius = 200;

    private List<List<LineSegment2D_F32>> segments = new ArrayList<>(3);
    private List<LineSegment2D_F32> found = new ArrayList<>();

    private List<Point2D_F32> visPoints = new ArrayList<>();


    private boolean configChanged = true;
    private ConfigLineRansac config = new ConfigLineRansac();
    private DetectLineSegment<GrayU8> detector;

    private Point2D_I32[] points = new Point2D_I32[2];
    private int offx, offy = 0;
    private ContourBoundingBox bound;
    private int draggingPoint = -1;
    private int width;
    private int height;

    private ArrayList<ContourBoundingBox> candidates = new ArrayList<>();
    private ArrayList<MatchResult> result = new ArrayList<>();

    public RadiusAreaStrat()
    {
        for (int i = 0; i < collateFrames; i++)
        {
            segments.add(new ArrayList<>());
        }
    }

    @Override
    public ArrayList<MatchResult> recognize(BufferedImage in, RecognitionStrategy strat)
    {
        result.clear();
        candidates.clear();
        List<LineSegment2D_F32> frameSegments = segments.remove(collateFrames - 1);
        frameSegments.clear();
        segments.add(0, frameSegments);
        if (draggingPoint == -1)
        {
            // convert the line into a single band image
            GrayU8 input = ConvertBufferedImage.convertFromSingle(in, null, GrayU8.class);
            GrayU8 blurred = input.createSameShape();

            // Blur smooths out gradient and improves results
            int blurRad = Math.max(1, (int) (radius / 20));
            GBlurImageOps.gaussian(input, blurred, 0, blurRad, null);

            if (configChanged)
            {
                detector = this.lineRansac(config);
                configChanged = false;
            }
            frameSegments.addAll(detector.detect(blurred));
            processSegments(strat);

            for (ContourBoundingBox bound : candidates)
            {
                BufferedImage norm = ImageUtil.getScaledImage(bound.getTransformedImage(in, false));
                BufferedImage flip = ImageUtil.getScaledImage(bound.getTransformedImage(in, true));
                ImageDesc i = new ImageDesc(norm, flip);
                MatchResult mr = strat.getMatch(i, SettingsPanel.RECOG_THRESH / 100.0);
                if (mr != null)
                {
                    result.add(mr);
                }
            }
        }
        return result;
    }

    private void processSegments(RecognitionStrategy strat)
    {
        found.clear();
        for (List<LineSegment2D_F32> frameSegments : segments)
        {
            found.addAll(frameSegments);
        }
        //eliminate segments outside of the recognition area
        int i = 0;
        while (i < found.size())
        {
            LineSegment2D_F32 line = found.get(i);
            if (!isGoodSegment(line))
            {
                found.remove(i);
            } else
            {
                i++;
            }
        }

        doAngleHistogram();
    }

    private double getAngle(LineSegment2D_F32 segment)
    {
        double angle = Math.toDegrees(Math.atan2(segment.slopeY(), segment.slopeX()));
        if (angle < 0)
        {
            angle = angle + 360;
        }
        return angle;
    }

    public void doAngleHistogram()
    {
        ArrayList<Double> angles = new ArrayList<>(found.size());
        final int HALF_BUCKETS = NUM_BUCKETS / 2;
        final double HALF_ANGLE = BUCKET_ANGLE / 2.0;
        for (LineSegment2D_F32 segment : found)
        {
            angles.add(getAngle(segment));
        }
        int[] count = new int[NUM_BUCKETS];
        for (double a : angles)
        {
            double angle = (a + HALF_ANGLE) % 180;
            int i = (int) (angle) / BUCKET_ANGLE;
            count[i] += 1;
        }
        int max = 0;
        int maxix = -1;
        for (int i = 0; i < NUM_BUCKETS; i++)
        {
            if (count[i] > max)
            {
                max = count[i];
                maxix = i;
            }
        }
        if (maxix != -1)
        {
            int correspix = (maxix + HALF_BUCKETS) % NUM_BUCKETS;
            if (count[maxix] >= 2 && count[correspix] >= 2)
            {
                int ix = 0;
                ArrayList<LineSegment2D_F32> seg1 = new ArrayList<>(count[maxix]);
                ArrayList<LineSegment2D_F32> seg2 = new ArrayList<>(count[correspix]);
                float trueAngle = 0;
                while (ix < found.size())
                {
                    double a = angles.get(ix);
                    double angle = (a + HALF_ANGLE) % 180;
                    int i = (int) (angle) / BUCKET_ANGLE;
                    if (i == maxix)
                    {
                        seg1.add(found.get(ix));
                        trueAngle += Math.toRadians(a);
                        ix++;
                    } else if (i == correspix)
                    {
                        seg2.add(found.get(ix));
                        ix++;
                    } else
                    {
                        found.remove(ix);
                        angles.remove(ix);
                    }
                }
                doCollateLines(seg1, seg2, trueAngle / seg1.size());
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void doCollateLines(List<LineSegment2D_F32> seg1, List<LineSegment2D_F32> seg2, double rad)
    {
        LineSegment2D_F32[] tangents = new LineSegment2D_F32[4];
        List<PointCluster>[] intersects = new List[4];
        List<LineSegment2D_F32>[] segs = new List[]{seg1, seg2};
        LineSegment2D_F32[] foundLines = new LineSegment2D_F32[4];

        float thresh = (float) (radius / 5);

        // Init tangent lines
        for (int i = 0; i < 4; i++)
        {
            intersects[i] = new ArrayList<>();
            Point2D_F32 a, b;
            double ax = radius * Math.cos(rad) + center.x;
            double ay = radius * Math.sin(rad) + center.y;
            rad += Math.PI / 2;
            double bx = ax + 100 * Math.cos(rad);
            double by = ay + 100 * Math.sin(rad);
            a = new Point2D_F32((float) ax, (float) ay);
            b = new Point2D_F32((float) bx, (float) by);
            tangents[i] = new LineSegment2D_F32(a, b);
        }

        // For the two sets of segments, intersect with each tangent line,
        // and cluster intersection points based on radius
        for (int i = 0; i < 2; i++)
        {
            List<LineSegment2D_F32> samples = segs[i];
            for (int j = 0; j < 2; j++)
            {
                int ix = i + (2 * j);
                LineSegment2D_F32 tan = tangents[ix];

                // Add to cluster, or create a new cluster
                for (LineSegment2D_F32 s : samples)
                {
                    Point2D_F32 inter = extrapolateAndCollide(tan, s);
                    float len = s.getLength();
                    List<PointCluster> inters = intersects[ix];
                    boolean matched = false;
                    for (PointCluster cluster : inters)
                    {
                        if (cluster.testAndAdd(inter, len, thresh))
                        {
                            matched = true;
                            break;
                        }
                    }
                    if (!matched)
                    {
                        inters.add(new PointCluster(inter, len));
                    }
                }
            }

            // Sort clusters by total segment length contained
            Collections.sort(intersects[i]);
            Collections.sort(intersects[i + 2]);

            // If there are 2 or more points on each side, match the top
            // 2 of each side into 2 line segments
            if (intersects[i].size() >= 2 && intersects[i + 2].size() >= 2)
            {
                Point2D_F32 a1, a2, b1, b2;
                a1 = intersects[i].get(0).getPoint();
                b1 = intersects[i].get(1).getPoint();
                a2 = intersects[i + 2].get(0).getPoint();
                b2 = intersects[i + 2].get(1).getPoint();
                if (a1.distance(a2) > a1.distance(b2))
                {
                    Point2D_F32 temp = a2;
                    a2 = b2;
                    b2 = temp;
                }
                foundLines[i] = new LineSegment2D_F32(a1, a2);
                foundLines[i + 2] = new LineSegment2D_F32(b1, b2);
            } else
            {
                return;
            }
        }
        List<Point2D_I32> corners = new ArrayList<>();
        for (int i = 0; i < 4; i++)
        {
            LineSegment2D_F32 l1, l2;
            l1 = foundLines[i];
            l2 = foundLines[(i + 1) % 4];
            Point2D_F32 corner = extrapolateAndCollide(l1, l2);
            corners.add(new Point2D_I32((int) corner.x, (int) corner.y));
        }
        candidates.add(new ContourBoundingBox(corners));
    }

    private boolean isGoodSegment(LineSegment2D_F32 segment)
    {
        double effectiveRadius = radius * 1.1;
        boolean inRadius = segment.a.distance(center) <= effectiveRadius &&
                segment.b.distance(center) <= effectiveRadius;
        double perp = Math.toRadians(getAngle(segment) + 90);
        Point2D_F32 out = new Point2D_F32(
                (float) (center.x + Math.cos(perp)),
                (float) (center.y + Math.sin(perp))
        );
        Point2D_F32 inter = extrapolateAndCollide(segment, new LineSegment2D_F32(center, out));
        boolean farEnough = inter.distance(center) >= radius / 4;

        return inRadius && farEnough;
    }

    @Override
    public String getStratName()
    {
        return "radius";
    }

    @Override
    public String getStratDisplayName()
    {
        return "Radius Recognition";
    }

    /**
     * Extrapolate 2 line segments, and get their collision point, or null
     * if the segments are perfectly parallel
     */
    Point2D_F32 extrapolateAndCollide(LineSegment2D_F32 s1, LineSegment2D_F32 s2)
    {
        float m1, b1, m2, b2;
        double a1, a2;

        m1 = s1.slopeY() / s1.slopeX();
        b1 = s1.a.y - (s1.a.x * m1);

        m2 = s2.slopeY() / s2.slopeX();
        b2 = s2.a.y - (s2.a.x * m2);

        if (m1 == m2)
        {
            return null;
        }

        float x = (b2 - b1) / (m1 - m2);
        float y = m1 * x + b1;
        return new Point2D_F32(x, y);
    }

    @Override
    public void draw(Graphics g)
    {
        for (int i = 0; i < points.length; i++)
        {
            Point2D_I32 p = points[i];
            if (draggingPoint == i)
            {
                g.setColor(Color.RED);
            } else
            {
                g.setColor(Color.WHITE);
            }
            g.fillOval(p.x - 3, p.y - 3, 7, 7);
        }
        if (draggingPoint != -1)
        {
            g.setColor(Color.RED);
        } else
        {
            g.setColor(Color.WHITE);
        }
        g.drawOval((int) (center.x - radius), (int) (center.y - radius), (int) (radius * 2), (int) (radius * 2));

        if (SavedConfig.DEBUG)
        {
            g.setColor(Color.BLUE);
            for (LineSegment2D_F32 line : found)
            {
                g.drawLine((int) line.a.x, (int) line.a.y, (int) line.b.x, (int) line.b.y);
                //g.drawString((int) getAngle(line) + "", (int) line.a.x, (int) line.a.y);
            }
        }


        for (Point2D_F32 p : visPoints)
        {
            g.fillOval((int) p.x - 3, (int) p.y - 3, 7, 7);
        }

        for (ContourBoundingBox bound : candidates)
        {
            bound.draw(g);
        }
    }

    private void updateCircle()
    {
        center.x = points[0].x;
        center.y = points[0].y;
        radius = points[0].distance(points[1]);
        config.connectLines = true;
        config.regionSize = (int) (radius / 4);
        config.thresholdAngle = 0.16;
        config.thresholdEdge = 10;
        configChanged = true;
    }

    @Override
    public void init(int width, int height)
    {
        this.width = width;
        this.height = height;
        points[0] = new Point2D_I32(width / 2, height / 2);
        int dist = (int) (height / 2 * 0.85);
        points[1] = new Point2D_I32(width / 2 + dist, height / 2);
        updateCircle();
    }

    @Override
    public SettingsEntry getSettingsEntry()
    {
        return null;
    }

    @Override
    public void mousePressed(MouseEvent e)
    {
        Point p = e.getPoint();
        double dist = p.distance(points[0].x, points[0].y);
        if (Math.abs(radius - dist) <= 3)
        {
            draggingPoint = 1;
            return;
        }

        dist = p.distance(points[0].x, points[0].y);
        if (dist <= 3)
        {
            draggingPoint = 0;
            offx = points[1].x - points[0].x;
            offy = points[1].y - points[0].y;
            found.clear();
            return;
        }
    }

    @Override
    public void mouseReleased(MouseEvent e)
    {
        draggingPoint = -1;
    }

    @Override
    public void mouseDragged(MouseEvent e)
    {
        if (draggingPoint != -1)
        {
            Point p = e.getPoint();
            if (p.x >= 0 && p.x <= this.width)
            {
                points[draggingPoint].x = p.x;
            }
            if (p.y >= 0 && p.y <= this.height)
            {
                points[draggingPoint].y = p.y;
            }
            if (draggingPoint == 0)
            {
                points[1].x = points[0].x + offx;
                points[1].y = points[0].y + offy;
            }
        }
        updateCircle();
    }

    @Override
    public void mouseMoved(MouseEvent e)
    {

    }

    @Override
    public void mouseEntered(MouseEvent e)
    {
    }

    @Override
    public void mouseExited(MouseEvent e)
    {
    }

    @Override
    public void mouseClicked(MouseEvent e)
    {

    }

    /**
     * Derived from the BoofCV factory source code, but exposes
     * the RANSAC iterations and the max lines per grid region
     */

    private DetectLineSegmentsGridRansac<GrayU8, GrayS16> lineRansac(ConfigLineRansac config) {
        if (config == null) {
            config = new ConfigLineRansac();
        }

        ImageGradient<GrayU8, GrayS16> gradient = FactoryDerivative.sobel(GrayU8.class, GrayS16.class);

        ModelManagerLinePolar2D_F32 manager = new ModelManagerLinePolar2D_F32();
        GridLineModelDistance distance = new GridLineModelDistance((float) config.thresholdAngle);
        GridLineModelFitter fitter = new GridLineModelFitter((float) config.thresholdAngle);

        // Adjusted initialization of Ransac
        Ransac<LinePolar2D_F32, Edgel> ransac = new Ransac<>(
                123123L,
                60,
                0.25,
                manager,
                Edgel.class
        );

        GridRansacLineDetector<GrayS16> alg =
                new ImplGridRansacLineDetector_S16(config.regionSize, 2, ransac);

        ConnectLinesGrid connect = null;
        if (config.connectLines)
            connect = new ConnectLinesGrid(Math.PI * 0.01, 1, 8);

        return new DetectLineSegmentsGridRansac<>(alg, connect, gradient, config.thresholdEdge, GrayU8.class, GrayS16.class);
    }
}