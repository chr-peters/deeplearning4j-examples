package dl4jtest;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Main {
    public static void main( String[] args ) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .updater(new Nesterovs(0.006, 0.9))
            .l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder() //create hidden layer with xavier initialization
                   .nIn(numRows * numColumns)
                   .nOut(1000)
                   .activation(Activation.RELU)
                   .weightInit(WeightInit.XAVIER)
                   .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create output layer
                   .nIn(1000)
                   .nOut(outputNum)
                   .activation(Activation.SOFTMAX)
                   .weightInit(WeightInit.XAVIER)
                   .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
	
        //print the score with every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        System.out.println("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        System.out.println(eval.stats());

        System.exit(0);
    }
}
