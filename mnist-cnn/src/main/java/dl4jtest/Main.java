package dl4jtest;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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

import java.util.Scanner;

/**
 * Digit classification on the MNIST-Dataset using a convolutional neural network.
 */
public class Main {
    public static void main( String[] args ) throws Exception{
        // number of channels (the images are grayscale so there is only one channel)
	int numChannels = 1;
	// 10 classes, one for each digit
	int numClasses = 10;
	// batchSize for the stochastic gradient descent algorithm
	int batchSize = 64;
	// how often to pass the entire dataset through the network during training
	int numEpochs = 1;
	// make the results reproduceable
	int seed = 123;

	// fetch the MNIST training dataset
	DataSetIterator trainingSet = new MnistDataSetIterator(batchSize, true, seed);
	// fetch the MNIST testing dataset	
	DataSetIterator testSet = new MnistDataSetIterator(batchSize, false, seed);

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
	    // in the MNIST dataset, the 28x28 grayscale images are represented as vectors, thus the input type is convolutionalFlat
	    .setInputType(InputType.convolutionalFlat(28, 28, 1))
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
