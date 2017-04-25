package com.hirnschmalz.network;

/**
 * Created by tobias.
 */
public interface NeuralNetwork {
    double calculateError();
    void feedForward();
    void backPropagate();

}
