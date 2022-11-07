import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Test2
{
    public static void main(String args[])
    {
        try {
           J48 j48Classifier = new J48();

           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances datasetInstances = source.getDataSet();
           datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

           Evaluation evaluation = new Evaluation(datasetInstances);
           evaluation.crossValidateModel(j48Classifier, datasetInstances, 10, new Random(1));
           System.out.println(evaluation.toSummaryString("\nResults", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
