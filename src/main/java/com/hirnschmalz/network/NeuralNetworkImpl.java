package com.hirnschmalz.network;

import com.hirnschmalz.image.DigitImage;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.List;

/**
 * Created by tobias.
 */
public class NeuralNetworkImpl implements NeuralNetwork {
    private final Layer inputLayer;
    private final Layer hiddenLayer;
    private final Layer outputLayer;
    private final double errorThreshold;

    public NeuralNetworkImpl(int numberInputNodes, int numberHiddenNodes, int numberOutputNodes, List<DigitImage> images, double errorThreshold) {
        inputLayer = new LayerImpl(numberInputNodes, 0.2, true, 0.9);
        hiddenLayer = new LayerImpl(numberHiddenNodes, 0.2, true, 0.9);
        outputLayer = new LayerImpl(numberOutputNodes, 0.2, true, 0.9);

        inputLayer.setChildLayer(hiddenLayer);

        hiddenLayer.setParentLayer(inputLayer);
        hiddenLayer.setChildLayer(outputLayer);

        outputLayer.setParentLayer(hiddenLayer);
        this.errorThreshold = errorThreshold;

        DateFormat timePointFormatter = new SimpleDateFormat("HH:mm:ss:SSS");
        DateFormat timespanFormatter = new SimpleDateFormat("mm:ss:SSS");
        long startTime = System.currentTimeMillis();
        System.out.println("Started learn at " + timePointFormatter.format(startTime));
        learn(images);
        long endTime = System.currentTimeMillis();
        System.out.println("Finished learn at " + timePointFormatter.format(endTime));
        long elapsedTime = endTime - startTime;
        System.out.println("Elapsed time: " + timespanFormatter.format(elapsedTime));
    }

    private void learn(List<DigitImage> images) {
        int iteration = 0;

        double error = 1;

        while (error > errorThreshold) {
            error = 0;
            for (DigitImage image : images) {
                inputLayer.setNeuronValues(image.getData());
                outputLayer.setDesiredValues(new double[outputLayer.getNumberOfNodes()]);
                outputLayer.setDesiredValue(image.getLabel(), 0.9);
                feedForward();
                error = calculateError();
                backPropagate();
            }

            System.out.println("Iteration: " + ++iteration + ", Current Error: " + error);
        }
        double[][] inputLayerWeights = inputLayer.getWeights();
        double[][] outputLayerWeights = hiddenLayer.getWeights();
        System.out.println("Final error rate: " + error);
    }

    @Override
    public double calculateError() {
        double error = 0;
        for (int i = 0; i < outputLayer.getNumberOfNodes(); i++) {
            error += Math.pow(outputLayer.getNeuronValues()[i] - outputLayer.getDesiredValues()[i], 2);
        }
        return error / outputLayer.getNumberOfNodes();
    }

    @Override
    public void feedForward() {
        inputLayer.calculateNeuronValues();
        hiddenLayer.calculateNeuronValues();
        outputLayer.calculateNeuronValues();
    }

    @Override
    public void backPropagate() {
        outputLayer.calculateErrors();
        hiddenLayer.calculateErrors();
        hiddenLayer.adjustWeights();
        inputLayer.adjustWeights();
    }
}
