package com.hirnschmalz;

import com.hirnschmalz.image.DigitImage;
import com.hirnschmalz.image.DigitImageLoadingService;

import java.io.IOException;
import java.util.List;

/**
 * Created by tobias.
 */
public class Main {
    public static void main(String[] args) throws IOException {

        final DigitImageLoadingService digitImageLoadingService = new DigitImageLoadingService("./data/train-labels.dat", "./data/train-images.dat");
        final List<DigitImage> digitImages = digitImageLoadingService.loadDigitImages();
        for (DigitImage digitImage : digitImages) {
            System.out.print(digitImage);
            System.out.println();
        }
    }
}
