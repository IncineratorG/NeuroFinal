package com.company;


import java.util.ArrayList;
import java.util.List;

public class Main2 {
    public static void main(String[] args) {
        int inputNodes = 3;
        int hiddenNodes = 3;
        int outputNodes = 2;
        double learningRate = 0.2;

        NeuralNetworkMultilayer neuralNetworkMultilayer = new NeuralNetworkMultilayer(inputNodes, hiddenNodes, outputNodes, learningRate);

        List<Double> inputsList = new ArrayList<>(3);
        inputsList.add(0.1);
        inputsList.add(0.3);
        inputsList.add(0.5);

        List<Double> targetsList = new ArrayList<>(2);
        targetsList.add(0.5);
        targetsList.add(0.7);

        for (int i = 0; i < 1000; ++i)
            neuralNetworkMultilayer.train(inputsList, targetsList);

        System.out.println(neuralNetworkMultilayer.query(inputsList));
    }
}
