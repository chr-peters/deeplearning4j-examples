package dl4jtest;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;

import java.io.File;
import java.util.Scanner;
import java.util.Random;

/**
 * Training a CNN to distinguish between happy and sad smileys.
 */
public class Main {
    public static void main( String[] args ) throws Exception{
        // number of channels (the images are grayscale so there is only one channel)
        int numChannels = 1;
        // the images are 50x50 pixels
        int width = 50, height = 50;
        // 2 classes, happy and sad
        int numClasses = 2;
        // there are 10 examples for each class, so use a batch size of 20
        int batchSize = 20;
        // how often to pass the entire dataset through the network during training
        int numEpochs = 250;
        // make the results reproduceable
        int seed = 123;

        // get the data folders
        File trainingFolder = new File ("images/train");
        File testingFolder = new File ("images/test");

        // shuffle the images at random
        Random random = new Random();
        FileSplit train = new FileSplit(trainingFolder, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit test = new FileSplit(testingFolder, NativeImageLoader.ALLOWED_FORMATS, random);

        // create the training set
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        // read the images and assign the labels
        ImageRecordReader recordReader = new ImageRecordReader(height, width, numChannels, labelGenerator);
        recordReader.initialize(train);
        // create the DataSetIterator
        DataSetIterator trainingSet = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
            .classification(1, numClasses)
            .build();

        // create the testing set
        ImageRecordReader recordReaderTesting = new ImageRecordReader(height, width, numChannels, labelGenerator);
        recordReaderTesting.initialize(test);
        // create the DataSetIterator
        DataSetIterator testSet = new RecordReaderDataSetIterator.Builder(recordReaderTesting, 10)
            .classification(1, numClasses)
            .build();

        // normalize the data
        DataNormalization scalerTrain = new ImagePreProcessingScaler(0, 1);
        scalerTrain.fit(trainingSet);
        trainingSet.setPreProcessor(scalerTrain);
        DataNormalization scalerTest = new ImagePreProcessingScaler(0, 1);
        scalerTest.fit(testSet);
        testSet.setPreProcessor(scalerTest);

        // create the network configuration
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            // use l2 regularization to reduce overfitting
            .l2(0.0005)
            // user xavier initialization
            .weightInit(WeightInit.XAVIER)
            // use stochastic gradient descent
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            // also use momentum (first parameter)
            .updater(new Nesterovs(0.1, 0.01))
            .list()
            // the first layer is a convolution layer with a kernel size of 5x5 pixels
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                   // the number of channels are specified with the nIn method
                   .nIn(numChannels)
                   // in each step move the kernel by just one pixel
                   .stride(1, 1)
                   // number of kernels to use
                   .nOut(20)
                   // use the RELU-Function as an activation function
                   .activation(new ActivationReLU())
                   .build())
            // next use a pooling (subsampling) layer utilizing MAX-pooling
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                   .kernelSize(2, 2)
                   .stride(2, 2)
                   .build())
            // use another convolutional layer, again with a 5x5 kernel size
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                   .stride(1, 1)
                   // this time use 50 different kernels
                   .nOut(50)
                   .activation(new ActivationReLU())
                   .build())
            // use one more subsampling layer before the densely connected network starts
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                   .kernelSize(2, 2)
                   .stride(2, 2)
                   .build())
            // hidden layer in the densely connected network
            .layer(4, new DenseLayer.Builder()
                   .activation(new ActivationReLU())
                   // use 500 hidden neurons in this layer
                   .nOut(500)
                   .build())
            // output layer of the network using NegativeLogLikelihood as loss function
            .layer(5, new OutputLayer.Builder(new LossNegativeLogLikelihood())
                   // use as many output neurons as there are classes
                   .nOut(numClasses)
                   // use the softmax function in the last layer so the outputs can be interpreted as probabilities
                   .activation(new ActivationSoftmax())
                   .build())
            // the images are represented as vectors, thus the input type is convolutionalFlat
            .setInputType(InputType.convolutionalFlat(width,height, 1))
            .backprop(true)
            .pretrain(false)
            .build();

        // now create the neural network from the configuration
        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
        // initialize the network
        neuralNetwork.init();

        // set up a local web-UI to monitor the training available at localhost:9000
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        // additionally print the score to stdout every 10 iterations
        neuralNetwork.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
        uiServer.attach(statsStorage);

        // now train the network for the desired number of epochs
        for (int curEpoch = 0; curEpoch < numEpochs; curEpoch++) {
            neuralNetwork.fit(trainingSet);
        }

        // evaluate the trained model and print the stats
        Evaluation evaluation = neuralNetwork.evaluate(testSet);
        System.out.println(evaluation.stats());
        
        // wait for input to tear down the web-UI
        Scanner sc = new Scanner(System.in);
        System.out.println("Press enter to end the application and destroy the web-UI.");
        sc.nextLine();
        
        System.exit(0);
    }
}
