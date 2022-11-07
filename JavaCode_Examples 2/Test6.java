import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.Filter;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Test6
{
    public static void main(String args[])
    {
        try {
           J48 j48Classifier = new J48();

           DataSource source = new DataSource("../data/iris.arff");
           Instances datasetInstances = source.getDataSet();
           datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);

           Normalize filter = new Normalize();
           filter.setInputFormat(datasetInstances);  
           Instances newDatasetInstances = Filter.useFilter(datasetInstances, filter);

           Evaluation evaluation = new Evaluation(newDatasetInstances);
           evaluation.crossValidateModel(j48Classifier, newDatasetInstances, 10, new Random(1));
           System.out.println(evaluation.toSummaryString("\nResults", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
