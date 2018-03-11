package com.company;


import org.ejml.simple.SimpleMatrix;

import java.util.List;
import java.util.Random;


public class NeuralNetwork {
    private int inputNodes = 0;
    private int hiddenNodes = 0;
    private int outputNodes = 0;
    private double learningRate = 0.5;
    private SimpleMatrix wih;
    private SimpleMatrix who;
    private SimpleMatrix inputsMatrix;


    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
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

        inputsMatrix = new SimpleMatrix(this.inputNodes, 1);
    }


    void train(List<Double> inputsList, List<Double> targetsList) {
        SimpleMatrix inputsListMatrix = new SimpleMatrix(inputsList.size(), 1);
        for (int row = 0; row < inputsListMatrix.numRows(); ++row)
            inputsListMatrix.set(row, 0, inputsList.get(row));

        SimpleMatrix targetsListMatrix = new SimpleMatrix(targetsList.size(), 1);
        for (int row = 0; row < targetsListMatrix.numRows(); ++row)
            targetsListMatrix.set(row, 0, targetsList.get(row));

        SimpleMatrix hiddenInputsMatrix = wih.mult(inputsListMatrix);

        SimpleMatrix hiddenOutputsMatrix = new SimpleMatrix(hiddenInputsMatrix.numRows(), hiddenInputsMatrix.numCols());
        for (int row = 0; row < hiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < hiddenOutputsMatrix.numCols(); ++column)
                hiddenOutputsMatrix.set(row, column,
                        ActivationFunctionClass.sigmoid(
                                hiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix finalInputsMatrix = who.mult(hiddenOutputsMatrix);

        SimpleMatrix finalOutputsMatrix = new SimpleMatrix(finalInputsMatrix.numRows(), finalInputsMatrix.numCols());
        for (int row = 0; row < finalOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < finalOutputsMatrix.numCols(); ++column)
                finalOutputsMatrix.set(row, column,
                        ActivationFunctionClass.sigmoid(
                                    finalInputsMatrix.get(row, column)
                            )
                        );

        SimpleMatrix outputErrorsMatrixList = targetsListMatrix.minus(finalOutputsMatrix);

        SimpleMatrix errorsHiddenLayerMatrix = who.transpose().mult(outputErrorsMatrixList);


        // ================================
        SimpleMatrix one = new SimpleMatrix(finalOutputsMatrix.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusFinalOutputs = one.minus(finalOutputsMatrix);

        SimpleMatrix elementsMultiplication = outputErrorsMatrixList.elementMult(finalOutputsMatrix).elementMult(oneMinusFinalOutputs);

        SimpleMatrix matrixMultiplication = elementsMultiplication.mult(hiddenOutputsMatrix.transpose());
        for (int row = 0; row < matrixMultiplication.numRows(); ++row)
            for (int column = 0; column < matrixMultiplication.numCols(); ++column)
                matrixMultiplication.set(row, column,
                            learningRate * matrixMultiplication.get(row, column)
                        );

        who = who.plus(matrixMultiplication);

//        SimpleMatrix oneMinusFinalOutputsMatrix = new SimpleMatrix(finalOutputsMatrix.numRows(), finalOutputsMatrix.numCols());
//        for (int row = 0; row < oneMinusFinalOutputsMatrix.numRows(); ++row)
//            for (int column = 0; column < oneMinusFinalOutputsMatrix.numCols(); ++column)
//                oneMinusFinalOutputsMatrix.set(row, column,
//                            1.0 - finalOutputsMatrix.get(row, column)
//                        );
//
//        SimpleMatrix tempMatrix_1 = outputErrorsMatrixList.elementMult(finalOutputsMatrix);
//        tempMatrix_1 = tempMatrix_1.elementMult(oneMinusFinalOutputsMatrix);
//
//        tempMatrix_1 = tempMatrix_1.mult(hiddenOutputsMatrix.transpose());
//
//        for (int row = 0; row < tempMatrix_1.numRows(); ++row)
//            for (int column = 0; column < tempMatrix_1.numCols(); ++column)
//                tempMatrix_1.set(row, column,
//                            learningRate * tempMatrix_1.get(row, column)
//                        );
//
//        who = who.plus(tempMatrix_1);
        // ================================

        // ===============================================
        one = new SimpleMatrix(hiddenOutputsMatrix.numRows(), 1);
        for (int row = 0; row < one.numRows(); ++row)
            for (int column = 0; column < one.numCols(); ++column)
                one.set(row, column, 1.0);

        SimpleMatrix oneMinusHiddenOutputs = one.minus(hiddenOutputsMatrix);

        elementsMultiplication = errorsHiddenLayerMatrix.elementMult(hiddenOutputsMatrix).elementMult(oneMinusHiddenOutputs);

        matrixMultiplication = elementsMultiplication.mult(inputsListMatrix.transpose());
        for (int row = 0; row < matrixMultiplication.numRows(); ++row)
            for (int column = 0; column < matrixMultiplication.numCols(); ++column)
                matrixMultiplication.set(row, column,
                        learningRate * matrixMultiplication.get(row, column)
                );

        wih = wih.plus(matrixMultiplication);

//        SimpleMatrix oneMinusHiddenOutputsMatrix = new SimpleMatrix(hiddenOutputsMatrix.numRows(), hiddenOutputsMatrix.numCols());
//        for (int row = 0; row < oneMinusHiddenOutputsMatrix.numRows(); ++row)
//            for (int column = 0; column < oneMinusHiddenOutputsMatrix.numCols(); ++column)
//                oneMinusHiddenOutputsMatrix.set(row, column,
//                        1.0 - hiddenOutputsMatrix.get(row, column)
//                );
//
//        SimpleMatrix tempMatrix_2 = errorsHiddenLayerMatrix.elementMult(hiddenOutputsMatrix);
//        tempMatrix_2 = tempMatrix_2.elementMult(oneMinusHiddenOutputsMatrix);
//
//        tempMatrix_2 = tempMatrix_2.mult(hiddenOutputsMatrix.transpose());
//
//        for (int row = 0; row < tempMatrix_2.numRows(); ++row)
//            for (int column = 0; column < tempMatrix_2.numCols(); ++column)
//                tempMatrix_2.set(row, column,
//                        learningRate * tempMatrix_2.get(row, column)
//                );
//
//        System.out.println(wih);
//        System.out.println(tempMatrix_2);
//
//        wih = wih.plus(tempMatrix_2);
        // ===============================================
    }


    SimpleMatrix query(List<Double> inputsList) {
        if (inputsList.size() != inputsMatrix.numRows())
            System.out.println("INPUTS_LIST_HAS_BAD_SIZE");

        if (inputsList.size() < inputsMatrix.numRows())
            System.out.println("INPUTS_LIST_SIZE_IS_SMALLER_THEN_INPUTS_MATRIX_SIZE");

        for (int row = 0; row < inputsMatrix.numRows(); ++row)
            inputsMatrix.set(row, 0, inputsList.get(row));

        SimpleMatrix hiddenInputsMatrix = wih.mult(inputsMatrix);

        SimpleMatrix hiddenOutputsMatrix = new SimpleMatrix(hiddenInputsMatrix.numRows(), hiddenInputsMatrix.numCols());
        for (int row = 0; row < hiddenOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < hiddenOutputsMatrix.numCols(); ++column)
                hiddenOutputsMatrix.set(row, column,
                        ActivationFunctionClass.sigmoid(
                                hiddenInputsMatrix.get(row, column)
                        )
                );

        SimpleMatrix finalInputsMatrix = who.mult(hiddenOutputsMatrix);

        SimpleMatrix finalOutputsMatrix = new SimpleMatrix(finalInputsMatrix.numRows(), finalInputsMatrix.numCols());
        for (int row = 0; row < finalOutputsMatrix.numRows(); ++row)
            for (int column = 0; column < finalOutputsMatrix.numCols(); ++column)
                finalOutputsMatrix.set(row, column,
                        ActivationFunctionClass.sigmoid(
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