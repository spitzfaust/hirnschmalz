package com.hirnschmalz.network;

/**
 * Created by tobias.
 */
public interface Layer {
    void calculateNeuronValues();

    void calculateErrors();

    void adjustWeights();

    Layer getChildLayer();

    Layer getParentLayer();

    void setChildLayer(Layer childLayer);

    void setParentLayer(Layer parentLayer);

    int getNumberOfParentNodes();

    int getNumberOfNodes();

    int getNumberOfChildNodes();

    double[] getNeuronValues();

    void setNeuronValues(double[] neuronValues);

    double[] getErrors();

    void setErrors(double[] errors);

    double[][] getWeights();

    void setWeights(double[][] weights);

    double[][] getWeightChanges();

    void setWeightChanges(double[][] weightChanges);

    double[] getBiasWeights();

    void setBiasWeights(double[] biasWeights);

    double[] getBiasValues();

    void setBiasValues(double[] biasValues);

    double[] getDesiredValues();

    void setDesiredValues(double[] desiredValues);

    void setDesiredValue(int pos, double value);

    double getLearningRate();

    boolean getUseMomentum();

    double getMomentumFactor();
}
