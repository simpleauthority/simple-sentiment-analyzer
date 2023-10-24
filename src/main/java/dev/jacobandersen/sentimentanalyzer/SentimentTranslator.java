package dev.jacobandersen.sentimentanalyzer;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class SentimentTranslator implements Translator<String, SentimentOutput> {
    private static final String TOKENIZER_NAME = "roberta-base";

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(TOKENIZER_NAME)) {
            Encoding enc = tokenizer.encode(input);
            NDManager man = ctx.getNDManager();
            return new NDList(man.create(enc.getIds()), man.create(enc.getAttentionMask()));
        }
    }

    @Override
    public SentimentOutput processOutput(TranslatorContext ctx, NDList list) {
        return new SentimentOutput(list);
    }
}
