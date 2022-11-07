import java.util.Random;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.meta.FilteredClassifier;

public class Test5
{
    public static void main(String args[])
    {
        try {
           DataSource source = new DataSource("../data/breast-cancer.arff");

           Instances train = source.getDataSet();
           train.setClassIndex(train.numAttributes() - 1);
           Instances test = source.getDataSet();
           test.setClassIndex(test.numAttributes() - 1);
 
           // filter
           Remove rm = new Remove();
           rm.setAttributeIndices("1");  // remove 1st attribute
 
           // classifier
           J48 j48 = new J48();
           j48.setUnpruned(true);
 
           // meta-classifier
           FilteredClassifier fc = new FilteredClassifier();
           fc.setFilter(rm);
           fc.setClassifier(j48);
 
           // train and make predictions
           fc.buildClassifier(train);
           System.out.println("ID\t" + "Actual\t\t\t" + "Predicted");
           for (int i = 0; i < test.numInstances(); i++) {
               double pred = fc.classifyInstance(test.instance(i));
               System.out.print((i+1) + "\t");
               System.out.print(test.classAttribute().value((int) test.instance(i).classValue()) + "\t");
               System.out.println(test.classAttribute().value((int) pred));
           }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
