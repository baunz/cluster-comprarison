package cluster;

import java.util.Collection;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.OnlineSummarizer;

/**
 * Collect statistics for distance measure calculation
 * 
 * @author j.schulte
 * 
 */
public class DelegatingDistanceMeasure implements DistanceMeasure {

    private long invocations = 0;

    private OnlineSummarizer centroidSize = new OnlineSummarizer();

    private OnlineSummarizer vectorSize = new OnlineSummarizer();

    private final DistanceMeasure distanceMeasure;

    public DelegatingDistanceMeasure(DistanceMeasure distanceMeasure) {
        super();
        this.distanceMeasure = distanceMeasure;
    }

    public double distance(Vector v1, Vector v2) {
        trace(v1, v2);
        return distanceMeasure.distance(v1, v2);
    }

    public Collection<Parameter<?>> getParameters() {
        return distanceMeasure.getParameters();
    }

    public void createParameters(String prefix, Configuration jobConf) {
        distanceMeasure.createParameters(prefix, jobConf);
    }

    public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
        trace(centroid, v);
        return distanceMeasure.distance(centroidLengthSquare, centroid, v);
    }

    private void trace(Vector centroid, Vector vector) {
        invocations++;
        centroidSize.add(centroid.getNumNondefaultElements());
        vectorSize.add(vector.getNumNondefaultElements());
    }

    public void configure(Configuration config) {
        distanceMeasure.configure(config);
    }

    @Override
    public String toString() {
        return invocations < 2 ? "" : String.format("%d invocations, median / avg  centroid size %f / %f , median / avg vector size %f / %f",
            invocations,
            centroidSize.getMedian(), centroidSize.getMean(),
            vectorSize.getMedian(), vectorSize.getMean());
    }

    public long getInvocations() {
        return invocations;
    }

}
