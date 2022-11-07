import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Test7
{
    public static void main(String args[])
    {
        try {
           J48 cls = new J48();

           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances train = source.getDataSet();
           train.setClassIndex(train.numAttributes() - 1);

           Instances test = source.getDataSet();
           test.setClassIndex(test.numAttributes() - 1);

           cls.buildClassifier(train);
           Evaluation eval = new Evaluation(train);
           eval.evaluateModel(cls, test);
           System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
