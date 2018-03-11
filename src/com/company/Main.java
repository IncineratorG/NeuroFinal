package com.company;

import javafx.util.Pair;
import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.util.*;


public class Main {
    private static String SPLIT_CHARACTER = ",";

    public static void main(String[] args) {
        String TRAIN_FILE_PATH = "C:\\Empty files\\mnist_dataset\\mnist_train.csv";
        String TEST_FILE_PATH = "C:\\Empty files\\mnist_dataset\\mnist_test.csv";

        System.out.println("LOADING_TRAINING_DATA");
        List<Pair<Integer, List<Double>>> trainDataList = loadData(TRAIN_FILE_PATH);
        System.out.println("TRAIN_DATA_LIST_SIZE: " + trainDataList.size());

        int inputNodes = 784;
        int hiddenNodes = 300;
        int outputNodes = 10;
        double learningRate = 0.2;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);

        System.out.println("TRAINING_NETWORK");
        int epochs = 2;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            System.out.println("EPOCH: " + (epoch + 1) + " of " + epochs);

            for (int trainDataListIndex = 0; trainDataListIndex < trainDataList.size(); ++trainDataListIndex) {
                Pair<Integer, List<Double>> trainDataItem = trainDataList.get(trainDataListIndex);

                List<Double> targetsList = new ArrayList<>(outputNodes);
                for (int i = 0; i < outputNodes; ++i)
                    targetsList.add(0.01);

                targetsList.set(trainDataItem.getKey(), 0.99);

                neuralNetwork.train(trainDataItem.getValue(), targetsList);

                if ((trainDataListIndex % 1000) == 0)
                    System.out.println("EPOCH: " + (epoch + 1) + " -> TRAINED: " + trainDataListIndex);
            }
        }

        System.out.println("LOADING_TEST_DATA");
        List<Pair<Integer, List<Double>>> testDataList = loadData(TEST_FILE_PATH);
        System.out.println("TEST_DATA_LIST_SIZE: " + testDataList.size());

        System.out.println("QUERY_NETWORK");
        int correctResults = 0;
        for (int testDataListIndex = 0; testDataListIndex < testDataList.size(); ++testDataListIndex) {
            Pair<Integer, List<Double>> testDataItem = testDataList.get(testDataListIndex);

            SimpleMatrix resultMatrix = neuralNetwork.query(testDataItem.getValue());
            int resultValue = maximumMatrixRow(resultMatrix);

            if (resultValue == testDataItem.getKey())
                ++correctResults;
        }

        System.out.println("NETWORK_RESULT: " + ((double) correctResults / testDataList.size()));
    }



    static List<Pair<Integer, List<Double>>> loadData(String path) {
        File file = new File(path);

        String line = null;
        List<String> linesList = new ArrayList<>();

        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new FileReader(file));

            while ((line = bufferedReader.readLine()) != null)
                linesList.add(line);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<Pair<Integer, List<Double>>> dataList = new ArrayList<>();

        for (int lineIndex = 0; lineIndex < linesList.size(); ++lineIndex) {
            String currentLine = linesList.get(lineIndex);
            currentLine = currentLine.replaceAll(" ", "");

            List<String> currentLineContentList = Arrays.asList(currentLine.split(SPLIT_CHARACTER));

            int codedNumber = Integer.parseInt(currentLineContentList.get(0));

            List<Double> codedNumberDataList = new ArrayList<>(currentLineContentList.size() - 1);
            for (int lineContentIndex = 1; lineContentIndex < currentLineContentList.size(); ++lineContentIndex) {
                double contentNumberDataItem = Double.parseDouble(currentLineContentList.get(lineContentIndex));
                contentNumberDataItem = contentNumberDataItem / 255 * 0.99 + 0.01;

                codedNumberDataList.add(contentNumberDataItem);
            }

            Pair<Integer, List<Double>> dataItem = new Pair<>(codedNumber, codedNumberDataList);
            dataList.add(dataItem);
        }

        return dataList;
    }




    static int maximumMatrixRow(SimpleMatrix matrix) {
        double maxDoubleValue = Double.MIN_VALUE;
        int rowWithMaximumValue = Integer.MIN_VALUE;

        for (int row = 0; row < matrix.numRows(); ++row)
            for (int column = 0; column < matrix.numCols(); ++column)
                if (matrix.get(row, column) > maxDoubleValue) {
                    maxDoubleValue = matrix.get(row, column);
                    rowWithMaximumValue = row;
                }

        return rowWithMaximumValue;
    }





    static void test2() {
        int value = 1;
        SimpleMatrix hidden_errors = new SimpleMatrix(5, 1);
        for (int row = 0; row < hidden_errors.numRows(); ++row)
            for (int column = 0; column < hidden_errors.numCols(); ++column)
                hidden_errors.set(row, column, value++);

        SimpleMatrix hidden_outputs = new SimpleMatrix(5, 1);
        for (int row = 0; row < hidden_outputs.numRows(); ++row)
            for (int column = 0; column < hidden_outputs.numCols(); ++column)
                hidden_outputs.set(row, column, value++);

        SimpleMatrix inputs = new SimpleMatrix(7, 1);
        for (int row = 0; row < inputs.numRows(); ++row)
            for (int column = 0; column < inputs.numCols(); ++column)
                inputs.set(row, column, value++);

        SimpleMatrix one = new SimpleMatrix(hidden_outputs.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusHiddenOutputs = one.minus(hidden_outputs);

        SimpleMatrix tempResult = hidden_errors.elementMult(hidden_outputs).elementMult(oneMinusHiddenOutputs);
        tempResult = tempResult.mult(inputs.transpose());

        SimpleMatrix wih = new SimpleMatrix(5, 7);
        for (int row = 0; row < wih.numRows(); ++row)
            for (int column = 0; column < wih.numCols(); ++column)
                wih.set(row, column, value++);

        System.out.println(tempResult);
        System.out.println(wih);

        wih = wih.plus(tempResult);

        System.out.println(wih);
    }

    static void test() {
        //        Random random = new Random();
//
//        SimpleMatrix simpleMatrix = new SimpleMatrix(3, 4);
//        for (int row = 0; row < simpleMatrix.numRows(); ++row)
//            for (int column = 0; column < simpleMatrix.numCols(); ++column)
//                simpleMatrix.set(row, column, random.nextDouble() - 0.5);
//
//        System.out.println(simpleMatrix);




        SimpleMatrix sm1 = new SimpleMatrix(2, 2);
        sm1.set(0, 0, 1);
        sm1.set(0, 1, 2);
        sm1.set(1, 0, 3);
        sm1.set(1, 1, 4);

        SimpleMatrix sm2 = new SimpleMatrix(2, 2);
        sm2.set(0, 0, 5);
        sm2.set(0, 1, 6);
        sm2.set(1, 0, 7);
        sm2.set(1, 1, 8);

        SimpleMatrix sm3 = new SimpleMatrix(3, 1);
        sm3.set(0, 0, 1);
        sm3.set(1, 0, 2);
        sm3.set(2, 0, 3);

        SimpleMatrix sm4 = new SimpleMatrix(3, 1);
        sm4.set(0, 0, 4);
        sm4.set(1, 0, 5);
        sm4.set(2, 0, 6);



//
//        SimpleMatrix result = sm2.minus(sm1);
//        System.out.println(result);

//        SimpleMatrix resultMatrix = sm1.mult(sm2);
//
//        System.out.println(resultMatrix);
    }
}
