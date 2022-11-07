import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Test4
{
    public static void main(String args[])
    {
        try {
           J48 j48Classifier = new J48();

           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances datasetInstances = source.getDataSet();
           datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);
 
           String[] options = new String[2];
           options[0] = "-R";
           options[1] = "1"; 
           Remove remove = new Remove();
           remove.setOptions(options);
           remove.setInputFormat(datasetInstances);
           Instances newDatasetInstances = Filter.useFilter(datasetInstances, remove);

           Evaluation evaluation = new Evaluation(newDatasetInstances);
           evaluation.crossValidateModel(j48Classifier, newDatasetInstances, 10, new Random(1));
           System.out.println(evaluation.toSummaryString("\nResults", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
