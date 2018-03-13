package com.company;


import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.Random;

public class NeuralNetworkMultilayer {
    private int inputNodes = 0;
    private int hiddenNodes = 0;
    private int outputNodes = 0;
    private double learningRate = 0.5;
    private SimpleMatrix wih;
    private SimpleMatrix who;
    private SimpleMatrix whh;
    private SimpleMatrix inputsMatrix;


    public NeuralNetworkMultilayer(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;

        Random random = new Random();

        wih = new SimpleMatrix(this.hiddenNodes, this.inputNodes);
        for (int row = 0; row < wih.numRows(); ++row)
            for (int column = 0; column < wih.numCols(); ++column)
                wih.set(row, column, random.nextDouble() - 0.53);

        who = new SimpleMatrix(this.outputNodes, this.hiddenNodes);
        for (int row = 0; row < who.numRows(); ++row)
            for (int column = 0; column < who.numCols(); ++column)
                who.set(row, column, random.nextDouble() - 0.53);

        whh = new SimpleMatrix(this.hiddenNodes, this.hiddenNodes);
        for (int row = 0; row < whh.numRows(); ++row)
            for (int column = 0; column < whh.numCols(); ++column)
                whh.set(row, column, random.nextDouble() - 0.53);

        inputsMatrix = new SimpleMatrix(this.inputNodes, 1);
    }


    void train(List<Double> inputsList, List<Double> targetsList) {
        SimpleMatrix inputsListMatrix = new SimpleMatrix(inputsList.size(), 1);
        for (int row = 0; row < inputsListMatrix.numRows(); ++row)
            inputsListMatrix.set(row, 0, inputsList.get(row));

        SimpleMatrix targetsListMatrix = new SimpleMatrix(targetsList.size(), 1);
        for (int row = 0; row < targetsListMatrix.numRows(); ++row)
            targetsListMatrix.set(row, 0, targetsList.get(row));

        SimpleMatrix firstHiddenInputsMatrix = wih.mult(inputsListMatrix);

        SimpleMatrix firstHiddenOutputsMatrix = new SimpleMatrix(firstHiddenInputsMatrix.numRows(), firstHiddenInputsMatrix.numCols());
        for (int row = 0; row < firstHiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < firstHiddenOutputsMatrix.numCols(); ++column)
                firstHiddenOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                firstHiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix secondHiddenInputsMatrix = whh.mult(firstHiddenOutputsMatrix);

        SimpleMatrix secondHiddenOutputsMatrix = new SimpleMatrix(secondHiddenInputsMatrix.numRows(), secondHiddenInputsMatrix.numCols());
        for (int row = 0; row < secondHiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < secondHiddenOutputsMatrix.numCols(); ++column)
                secondHiddenOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                secondHiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix finalInputsMatrix = who.mult(secondHiddenOutputsMatrix);

        SimpleMatrix finalOutputsMatrix = new SimpleMatrix(finalInputsMatrix.numRows(), finalInputsMatrix.numCols());
        for (int row = 0; row < finalOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < finalOutputsMatrix.numCols(); ++column)
                finalOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                finalInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix outputErrorsMatrixList = targetsListMatrix.minus(finalOutputsMatrix);

        SimpleMatrix errorsFinalHiddenLayerMatrix = who.transpose().mult(outputErrorsMatrixList);

        SimpleMatrix errorsSecondHiddenLayerMatrix = whh.transpose().mult(errorsFinalHiddenLayerMatrix);


        SimpleMatrix one = new SimpleMatrix(finalOutputsMatrix.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusFinalOutputs = one.minus(finalOutputsMatrix);

        SimpleMatrix elementsMultiplication = outputErrorsMatrixList.elementMult(finalOutputsMatrix).elementMult(oneMinusFinalOutputs);

        SimpleMatrix matrixMultiplication = elementsMultiplication.mult(secondHiddenOutputsMatrix.transpose());
        for (int row = 0; row < matrixMultiplication.numRows(); ++row)
            for (int column = 0; column < matrixMultiplication.numCols(); ++column)
                matrixMultiplication.set(row, column,
                        learningRate * matrixMultiplication.get(row, column)
                );

        who = who.plus(matrixMultiplication);


        one = new SimpleMatrix(secondHiddenOutputsMatrix.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusSecondHiddenOutputs = one.minus(secondHiddenOutputsMatrix);

        elementsMultiplication = errorsFinalHiddenLayerMatrix.elementMult(secondHiddenOutputsMatrix).elementMult(oneMinusSecondHiddenOutputs);

        matrixMultiplication = elementsMultiplication.mult(firstHiddenOutputsMatrix.transpose());
        for (int row = 0; row < matrixMultiplication.numRows(); ++row)
            for (int column = 0; column < matrixMultiplication.numCols(); ++column)
                matrixMultiplication.set(row, column,
                        learningRate * matrixMultiplication.get(row, column)
                );

        whh = whh.plus(matrixMultiplication);


        one = new SimpleMatrix(firstHiddenOutputsMatrix.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusFirstHiddenOutputs = one.minus(firstHiddenOutputsMatrix);

        elementsMultiplication = errorsSecondHiddenLayerMatrix.elementMult(firstHiddenOutputsMatrix).elementMult(oneMinusFirstHiddenOutputs);

        matrixMultiplication = elementsMultiplication.mult(inputsListMatrix.transpose());
        for (int row = 0; row < matrixMultiplication.numRows(); ++row)
            for (int column = 0; column < matrixMultiplication.numCols(); ++column)
                matrixMultiplication.set(row, column,
                        learningRate * matrixMultiplication.get(row, column)
                );

        wih = wih.plus(matrixMultiplication);
    }


    SimpleMatrix query(List<Double> inputsList) {
        if (inputsList.size() != inputsMatrix.numRows())
            System.out.println("INPUTS_LIST_HAS_BAD_SIZE");

        if (inputsList.size() < inputsMatrix.numRows())
            System.out.println("INPUTS_LIST_SIZE_IS_SMALLER_THEN_INPUTS_MATRIX_SIZE");

        for (int row = 0; row < inputsMatrix.numRows(); ++row)
            inputsMatrix.set(row, 0, inputsList.get(row));

        SimpleMatrix firstHiddenInputsMatrix = wih.mult(inputsMatrix);

        SimpleMatrix firstHiddenOutputsMatrix = new SimpleMatrix(firstHiddenInputsMatrix.numRows(), firstHiddenInputsMatrix.numCols());
        for (int row = 0; row < firstHiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < firstHiddenOutputsMatrix.numCols(); ++column)
                firstHiddenOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                firstHiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix secondHiddenInputsMatrix = whh.mult(firstHiddenOutputsMatrix);

        SimpleMatrix secondHiddenOutputsMatrix = new SimpleMatrix(secondHiddenInputsMatrix.numRows(), secondHiddenInputsMatrix.numCols());
        for (int row = 0; row < secondHiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < secondHiddenOutputsMatrix.numCols(); ++column)
                secondHiddenOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                secondHiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix finalInputsMatrix = who.mult(secondHiddenOutputsMatrix);

        SimpleMatrix finalOutputsMatrix = new SimpleMatrix(finalInputsMatrix.numRows(), finalInputsMatrix.numCols());
        for (int row = 0; row < finalOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < finalOutputsMatrix.numCols(); ++column)
                finalOutputsMatrix.set(row, column,
                        NeuralNetwork.ActivationFunctionClass.sigmoid(
                                finalInputsMatrix.get(row, column)
                        )
                );

        return finalOutputsMatrix;
    }


    public static class ActivationFunctionClass {
        static double sigmoid(double x) {
            return (1/( 1 + Math.pow(Math.E,(-1 * x))));
        }
    }
}
