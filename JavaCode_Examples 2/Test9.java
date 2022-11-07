import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;

public class Test9
{
    public static void main(String args[])
    {
        try {
           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances data= source.getDataSet();
           data.setClassIndex(data.numAttributes() - 1);

           AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
           CfsSubsetEval eval = new CfsSubsetEval();
           GreedyStepwise search = new GreedyStepwise();
           search.setSearchBackwards(true);
           J48 base = new J48();
           classifier.setClassifier(base);
           classifier.setEvaluator(eval);
           classifier.setSearch(search);

           Evaluation evaluation = new Evaluation(data);
           evaluation.crossValidateModel(classifier, data, 10, new Random(1));
           System.out.println(evaluation.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
