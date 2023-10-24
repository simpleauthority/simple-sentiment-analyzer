package dev.jacobandersen.sentimentanalyzer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

import java.util.HashMap;
import java.util.Map;

public class SentimentOutput {
    private final Map<SentimentOutputCategory, Float> outputs;

    public SentimentOutput(NDList list) {
        this.outputs = new HashMap<>();

        if (list == null) {
            throw new RuntimeException("NDList provided to SentimentOutput is null");
        }

        try (NDArray output = list.get(0).softmax(0)) {
            float[] raw = output.toFloatArray();
            outputs.put(SentimentOutputCategory.NEGATIVE, raw[0]);
            outputs.put(SentimentOutputCategory.NEUTRAL, raw[1]);
            outputs.put(SentimentOutputCategory.POSITIVE, raw[2]);
        }
    }

    public Map<SentimentOutputCategory, Float> getOutputs() {
        return outputs;
    }

    public SentimentOutputCategory getMainSentiment() {
        SentimentOutputCategory main = SentimentOutputCategory.NEGATIVE;
        float max = 0.0f;

        for (Map.Entry<SentimentOutputCategory, Float> entry : outputs.entrySet()) {
            float val = entry.getValue();
            if (val > max) {
                max = val;
                main = entry.getKey();
            }
        }

        return main;
    }

    public enum SentimentOutputCategory {
        NEGATIVE,
        NEUTRAL,
        POSITIVE
    }
}
