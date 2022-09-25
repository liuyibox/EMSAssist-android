package org.tensorflow.lite.examples.emsassist.ml;

import com.google.common.primitives.Ints;

import java.util.List;

public class EMSBertFeature {
    public final int[] inputIds;
    public final int[] inputMask;
    public final int[] segmentIds;

    public EMSBertFeature(
            List<Integer> inputIds,
            List<Integer> inputMask,
            List<Integer> segmentIds) {
        this.inputIds = Ints.toArray(inputIds);
        this.inputMask = Ints.toArray(inputMask);
        this.segmentIds = Ints.toArray(segmentIds);
    }
}
