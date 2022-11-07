import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Test1
{
    public static void main(String args[])
    {
        try {
           J48 j48Classifier = new J48();

           String breastCancerDataset = "../data/breast-cancer.arff";
           BufferedReader bufferedReader = new BufferedReader(new FileReader(breastCancerDataset));
           Instances datasetInstances = new Instances(bufferedReader);
           datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

           Evaluation evaluation = new Evaluation(datasetInstances);
           evaluation.crossValidateModel(j48Classifier, datasetInstances, 10, new Random(1));
           System.out.println(evaluation.toSummaryString("\nResults", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
