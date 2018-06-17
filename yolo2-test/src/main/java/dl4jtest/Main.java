package dl4jtest;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.model.YOLO2;
import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.deeplearning4j.zoo.util.Labels;
import org.deeplearning4j.zoo.util.ClassPrediction;
import org.deeplearning4j.nn.conf.inputs.InputType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import org.datavec.image.loader.NativeImageLoader;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import java.io.File;
import java.util.List;

/**
 * Testing the pretrained YOLO2-model on an image with an apple and a slice of pizza.
 * According to the documentation the model was trained on the COCO dataset.
 *
 * Other test images are available in the images folder. The size of these images does not
 * matter since they will be scaled to fit the 608*608 format that YOLO2 uses.
 */
public class Main {
    public static void main( String[] args ) throws Exception{
	// this is the YOLO2-model that was trained on the COCO dataset
	YOLO2 model = YOLO2.builder().build();
        ComputationGraph initializedModel = (ComputationGraph) model.initPretrained();

	// get the input image we want to perform the detection on
	// this model supports images of 608x608 pixels with 3 channels (RGB)
	NativeImageLoader loader = new NativeImageLoader(608, 608, 3);
	// enter the path to the image here
	INDArray image = loader.asMatrix(new File("images/apple_and_pizza.jpg"));

	// for YOLO2, the pixel values have to be between 0 and 1
	DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

	// now get the detected objects using YoloUtils
	INDArray outputs = initializedModel.outputSingle(image);
	List<DetectedObject> objects = YoloUtils.getPredictedObjects(Nd4j.create(model.getPriorBoxes()), outputs, 0.6, 0.4);

	// get the COCO labels to convert the network output to actual readable strings
	Labels labels = new COCOLabels();

	// create a frame to visualize the results
	CanvasFrame frame = new CanvasFrame("YOLO2 test");

	// draw the rectangle and the text on imgMat
	Mat mat = loader.asMat(image);
	Mat imgMat = new Mat();
	// this conversion is done because it will lead to errors otherwise
	mat.convertTo(imgMat, CV_8U, 255, 0);
	
	for (DetectedObject curObj: objects) {
	    // get the coordinates of the bounding box
	    double[] xy1 = curObj.getTopLeftXY();
	    double[] xy2 = curObj.getBottomRightXY();

	    // the YOLO2 grid is 19x19, so calculate the box-coordinates in the original picture
	    int x1 = (int) Math.round(608 * xy1[0] / 19);
	    int y1 = (int) Math.round(608 * xy1[1] / 19);
	    int x2 = (int) Math.round(608 * xy2[0] / 19);
	    int y2 = (int) Math.round(608 * xy2[1] / 19);
	    
	    // draw the box
	    rectangle(imgMat, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
	    
	    // get the predictions the network made
	    ClassPrediction prediction = labels.decodePredictions(curObj.getClassPredictions(), 1).get(0).get(0);

	    // write the text
	    putText(imgMat, String.format("%.2f", prediction.getProbability()*100)+"% "+prediction.getLabel(), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
	}

	// this is used to display the image on the frame
	OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
	
	frame.showImage(converter.convert(imgMat));
	frame.waitKey();
	frame.dispose();

	System.exit(0);
    }
}
