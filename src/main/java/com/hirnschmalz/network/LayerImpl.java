package com.hirnschmalz.network;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by tobias.
 */
public class LayerImpl implements Layer {
    private Layer childLayer;
    private Layer parentLayer;
    private final int numberOfNodes;
    private double[] neuronValues;
    private double[] errors;
    private double[][] weights;
    private double[][] weightChanges;
    private double[] biasWeights;
    private double[] biasValues;
    private double[] desiredValues;
    private final double learningRate;
    private final boolean useMomentum;
    private final double momentumFactor;

    public LayerImpl(int numberOfNodes, double learningRate, boolean useMomentum, double momentumFactor) {
        this.numberOfNodes = numberOfNodes;
        this.learningRate = learningRate;
        this.useMomentum = useMomentum;
        this.momentumFactor = momentumFactor;

        neuronValues = new double[numberOfNodes];
        desiredValues = new double[numberOfNodes];
        errors = new double[numberOfNodes];
    }

    @Override
    public void calculateNeuronValues() {
        if (parentLayer != null) {
            double x;
            for (int i = 0; i < numberOfNodes; i++) {
                x = 0;
                for (int j = 0; j < getNumberOfParentNodes(); j++) {
                    x += parentLayer.getNeuronValues()[j] * parentLayer.getWeights()[j][i];
                }
                x += parentLayer.getBiasValues()[i] * parentLayer.getBiasWeights()[i];

                if (childLayer == null) {
                    neuronValues[i] = x;
                } else {
                    neuronValues[i] = 1d / (1d + Math.exp(-x));
                }
            }
        }
    }

    @Override
    public void calculateErrors() {
        if (childLayer == null) { // output
            for (int i = 0; i < numberOfNodes; i++) {
                double x = (desiredValues[i] - neuronValues[i]) * neuronValues[i] * (1d - neuronValues[i]);
                errors[i] = (desiredValues[i] - neuronValues[i]) * neuronValues[i] * (1d - neuronValues[i]);
            }
        } else if (parentLayer == null) { // input
            Arrays.fill(errors, 0);
        } else { // hidden
            double sum;
            for (int i = 0; i < numberOfNodes; i++) {
                sum = 0;
                for (int j = 0; j < getNumberOfChildNodes(); j++) {
                    sum += childLayer.getErrors()[j] * weights[i][j];
                }
                errors[i] = sum * neuronValues[i] * (1d - neuronValues[i]);
            }
        }
    }

    @Override
    public void adjustWeights() {
        if (childLayer != null) {
            double dw;
            for (int i = 0; i < numberOfNodes; i++) {
                for (int j = 0; j < getNumberOfChildNodes(); j++) {
                    dw = learningRate * childLayer.getErrors()[j] * neuronValues[i];
                    if (useMomentum) {
                        weights[i][j] += dw + momentumFactor * weightChanges[i][j];
                        weightChanges[i][j] = dw;
                    } else {
                        weights[i][j] += dw;
                    }
                }
            }
            for (int i = 0; i < getNumberOfChildNodes(); i++) {
                biasWeights[i] += learningRate * childLayer.getErrors()[i] * biasValues[i];
            }
        }
    }

    @Override
    public Layer getChildLayer() {
        return childLayer;
    }

    @Override
    public Layer getParentLayer() {
        return parentLayer;
    }

    @Override
    public void setChildLayer(Layer childLayer) {
        this.childLayer = childLayer;

        weights = new double[numberOfNodes][getNumberOfChildNodes()];
        weightChanges = new double[numberOfNodes][getNumberOfChildNodes()];
        biasWeights = new double[getNumberOfChildNodes()];
        biasValues = new double[getNumberOfChildNodes()];

        Arrays.fill(biasValues, -1);

        for (int i = 0; i < numberOfNodes; i++) {
            for (int j = 0; j < getNumberOfChildNodes(); j++) {
                double rand = ThreadLocalRandom.current().nextDouble(-1, 1);
                weights[i][j] = rand;
            }
        }

        for (int j = 0; j < getNumberOfChildNodes(); j++) {
            biasWeights[j] = ThreadLocalRandom.current().nextDouble(-1, 1);
        }

    }

    @Override
    public void setParentLayer(Layer parentLayer) {
        this.parentLayer = parentLayer;
    }

    @Override
    public int getNumberOfParentNodes() {
        if (parentLayer == null) {
            return 0;
        }
        return parentLayer.getNumberOfNodes();
    }

    @Override
    public int getNumberOfNodes() {
        return numberOfNodes;
    }

    @Override
    public int getNumberOfChildNodes() {
        if (childLayer == null) {
            return 0;
        }
        return childLayer.getNumberOfNodes();
    }

    @Override
    public double[] getNeuronValues() {
        return neuronValues;
    }

    @Override
    public void setNeuronValues(double[] neuronValues) {
        this.neuronValues = neuronValues;
    }

    @Override
    public double[] getErrors() {
        return errors;
    }

    @Override
    public void setErrors(double[] errors) {
        this.errors = errors;
    }

    @Override
    public double[][] getWeights() {
        return weights;
    }

    @Override
    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    @Override
    public double[][] getWeightChanges() {
        return weightChanges;
    }

    @Override
    public void setWeightChanges(double[][] weightChanges) {
        this.weightChanges = weightChanges;
    }

    @Override
    public double[] getBiasWeights() {
        return biasWeights;
    }

    @Override
    public void setBiasWeights(double[] biasWeights) {
        this.biasWeights = biasWeights;
    }

    @Override
    public double[] getBiasValues() {
        return biasValues;
    }

    @Override
    public void setBiasValues(double[] biasValues) {
        this.biasValues = biasValues;
    }

    @Override
    public double[] getDesiredValues() {
        return desiredValues;
    }

    @Override
    public void setDesiredValues(double[] desiredValues) {
        this.desiredValues = desiredValues;
    }

    @Override
    public void setDesiredValue(int pos, double value) {
        this.desiredValues[pos] = value;
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public boolean getUseMomentum() {
        return useMomentum;
    }

    @Override
    public double getMomentumFactor() {
        return momentumFactor;
    }
}
