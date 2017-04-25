package com.hirnschmalz;

import com.hirnschmalz.image.DigitImage;
import com.hirnschmalz.image.DigitImageLoadingService;
import com.hirnschmalz.network.NeuralNetwork;
import com.hirnschmalz.network.NeuralNetworkImpl;

import org.jooq.lambda.Seq;

import java.io.IOException;
import java.util.List;

/**
 * Created by tobias.
 */
public class Main {
    public static void main(String[] args) throws IOException {

        final DigitImageLoadingService digitImageLoadingService = new DigitImageLoadingService("./data/train-labels.dat", "./data/train-images.dat");
        final List<DigitImage> digitImages = digitImageLoadingService.loadDigitImages();
        NeuralNetwork neuralNetwork = new NeuralNetworkImpl(784, 89, 10, digitImages, 0.005);

    }
}
