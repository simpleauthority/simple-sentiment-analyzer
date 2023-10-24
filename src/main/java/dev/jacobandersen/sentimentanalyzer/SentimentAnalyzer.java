package dev.jacobandersen.sentimentanalyzer;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.Scanner;

public class SentimentAnalyzer {
    private static final String MODEL_ZOO_GROUP = "ai.djl.huggingface.pytorch";
    private static final String MODEL_LOADER_GROUP = "cardiffnlp/twitter-roberta-base-sentiment";

    public static void main(String[] args) {
        try {
            System.out.println("Getting ModelLoader...");
            ModelLoader loader = ModelZoo.getModelZoo(MODEL_ZOO_GROUP).getModelLoader(MODEL_LOADER_GROUP);

            System.out.println("Loading model...");
            try (ZooModel<NDList, NDList> model = loader.loadModel(Criteria.builder().setTypes(NDList.class, NDList.class).build())) {
                System.out.println("Creating predictor...");
                Predictor<String, SentimentOutput> pred = model.newPredictor(new SentimentTranslator());

                Scanner keyboard = new Scanner(System.in);
                while (true) {
                    System.out.printf("Enter the text to analyze. Send \"quit\" to exit.%n: ");

                    String line = keyboard.nextLine();
                    if (line.equalsIgnoreCase("quit")) {
                        break;
                    }

                    System.out.println("\nGenerating output...");
                    SentimentOutput output = pred.predict(line);

                    System.out.println("Results:");
                    output.getOutputs().forEach((cat, level) -> System.out.printf("\t%s: %.2f%%%n", cat.name(), (level * 100)));
                    System.out.printf("%nThe main sentiment of the text is: %s.%n%n", output.getMainSentiment().name());
                }
            }
        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException(e);
        }
    }
}
