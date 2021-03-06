package cluster;

import static com.google.common.collect.Iterables.size;

import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.clustering.streaming.cluster.StreamingKMeans;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.neighborhood.FastProjectionSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Function;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

public class ClusterCompare {

    private List<Vector> vectors = Lists.newArrayList();

    // total points
    private int numDataPoints = (int) 400000;

    // to prevent us from needing to much ram
    private int testPoints = (int) 10000;

    // needed for regular k-means
    private int finalClusters = 10;

    // cardinality of the vectors
    private final int cardinality = 100000;

    // elements set per vector
    private int averageElementsSet = 10;

    private static final Logger LOG = LoggerFactory.getLogger(ClusterCompare.class);

    private DelegatingDistanceMeasure distanceMeasure = new DelegatingDistanceMeasure(new CosineDistanceMeasure());

    @Before
    public void before() {

        Random random = new Random();

        for (int i = 0; i < testPoints; i++) {
            Vector v = new RandomAccessSparseVector(cardinality);
            for (int j = 0; j < averageElementsSet; j++) {
                v.set(random.nextInt(cardinality), 1);
            }
            vectors.add(v);

        }

    }

    @Test
    public void testKMeans() {

        List<Cluster> seedCentroids = Lists.newArrayList();
        Random r = new Random();

        for (int i = 0; i < finalClusters; i++) {

            int nextIndex = r.nextInt(vectors.size());
            Vector seed = vectors.get(nextIndex);

            Kluster newCluster = new Kluster(seed.clone(), i, distanceMeasure);
            newCluster.observe(seed, 1);
            seedCentroids.add(newCluster);
        }

        ClusterClassifier prior =
            new ClusterClassifier(seedCentroids, new KMeansClusteringPolicy(0.001));

        final Stopwatch watch = new Stopwatch().start();
        Iterable<Vector> allDataPoints = new Iterable<Vector>() {

            private int iterationCount = 0;

            @Override
            public Iterator<Vector> iterator() {
                LOG.info(getSummary(numDataPoints * iterationCount, watch));
                LOG.info(distanceMeasure.toString());
                iterationCount++;
                return Iterables.limit(Iterables.cycle(vectors), numDataPoints).iterator();
            }
        };
        ClusterClassifier clusters = ClusterIterator.iterate(allDataPoints, prior, 10);

        LOG.info(watch.toString() + " for " + clusters.getModels().size() + " clusters");
        LOG.info(distanceMeasure.toString());

    }

    @Test
    public void testStreaming() {

        int numSketchClusters = (int) (finalClusters * Math.log(numDataPoints));
        LOG.info("Sketch clusters: " + numSketchClusters);
        StreamingKMeans streamingKMeans =
            new StreamingKMeans(new FastProjectionSearch(distanceMeasure, 3, 2), numSketchClusters);

        final Stopwatch watch = new Stopwatch().start();
        Iterable<Centroid> allDataPoints =
            Iterables.transform(Iterables.limit(Iterables.cycle(vectors), numDataPoints), new Function<Vector, Centroid>() {

                private int numPoints = 0;

                @Override
                public Centroid apply(Vector input) {
                    Centroid centroid = new Centroid(numPoints++, input, 1);
                    if (numPoints % (numDataPoints / 10) == 0) {
                        LOG.info(distanceMeasure.toString());
                        LOG.info(getSummary(numPoints, watch));
                    }
                    return centroid;
                }
            });
        UpdatableSearcher clusters = streamingKMeans.cluster(allDataPoints);
        LOG.info(watch.toString() + " for " + size(clusters) + " clusters");
        LOG.info(distanceMeasure.toString());
    }

    private String getSummary(int numPoints, Stopwatch stopwatch) {
        return String.format("numPoints %d, %f points per second, distance measure calc. per point %f", numPoints,
            ((double) numPoints / Math.max(1, stopwatch.elapsed(TimeUnit.SECONDS))),
            (double) distanceMeasure.getInvocations() / Math.max(1, numPoints));
    }
}
