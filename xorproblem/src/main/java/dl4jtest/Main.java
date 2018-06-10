package dl4jtest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;

public class Main  {
    public static void main( String[] args ) {
	
        // create the input data
	INDArray input = Nd4j.zeros(4, 2);
	input.putScalar(new int [] {0, 0}, 0);
	input.putScalar(new int [] {0, 1}, 0);
	input.putScalar(new int [] {1, 0}, 1);
	input.putScalar(new int [] {1, 1}, 0);
	input.putScalar(new int [] {2, 0}, 0);
	input.putScalar(new int [] {2, 1}, 1);
	input.putScalar(new int [] {3, 0}, 1);
	input.putScalar(new int [] {3, 1}, 1);

	// create the desired network output
	// class 0: (1 0), class 1: (0, 1)
	INDArray labels = Nd4j.zeros(4, 2);
	labels.putScalar(new int[] {0, 0}, 1);
	labels.putScalar(new int[] {0, 1}, 0);
	labels.putScalar(new int[] {1, 0}, 0);
	labels.putScalar(new int[] {1, 1}, 1);
	labels.putScalar(new int[] {2, 0}, 0);
	labels.putScalar(new int[] {2, 1}, 1);
	labels.putScalar(new int[] {3, 0}, 1);
	labels.putScalar(new int[] {3, 1}, 0);

	// create the dataset from the input and the labels
	DataSet dataset = new DataSet(input, labels);

	// create the neural network configuration
	NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
	builder.iterations(5000);
	builder.learningRate(0.1);
	builder.seed(123);
	builder.useDropConnect(false);
	builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
	builder.biasInit(0);
	builder.miniBatch(false);

	// number of hidden neurons
	int hiddenNeurons = 4;

	// create the hidden layer
	DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
	// 2 neurons in the previous layer (input layer)
	hiddenLayerBuilder.nIn(2);
	// there will be 4 outgoing connections (thus there are 4 neurons in the current layer)
	hiddenLayerBuilder.nOut(hiddenNeurons);
	hiddenLayerBuilder.activation(new ActivationSigmoid());
	hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
	hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

	// create the output layer (use L2 as loss function)
	OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(new LossL2());
	outputLayerBuilder.nIn(hiddenNeurons);
	outputLayerBuilder.nOut(2);
	outputLayerBuilder.activation(new ActivationSoftmax());
	outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
	outputLayerBuilder.dist(new UniformDistribution(0, 1));

	// connect the configurations
	NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
	listBuilder.layer(0, hiddenLayerBuilder.build());
	listBuilder.layer(1, outputLayerBuilder.build());
	listBuilder.pretrain(false);

	MultiLayerConfiguration networkConfiguration = listBuilder.build();
	
	// create and initialize the neural network
	MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(networkConfiguration);
	neuralNetwork.init();

	// print the error on every 100 iterations
	neuralNetwork.setListeners(new ScoreIterationListener(100));

	// train the network on the data
	neuralNetwork.fit(dataset);

	// create output for every training sample
	INDArray output = neuralNetwork.output(dataset.getFeatureMatrix());

	// evaluate the results, 2 means 2 classes
	Evaluation evaluation = new Evaluation(2);
	evaluation.eval(dataset.getLabels(), output);
	System.out.println("Confusion Matrix:");
	System.out.println("=================");
	System.out.println(evaluation.confusionToString());
	System.out.println(evaluation.stats());

	System.exit(0);
    }
}
