import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;

public class Test11
{
    public static void main(String args[])
    {
        try {
           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances data = source.getDataSet();
           data.setClassIndex(data.numAttributes() - 1);

           AttributeSelection attsel = new AttributeSelection();
           CfsSubsetEval eval = new CfsSubsetEval();
           GreedyStepwise search = new GreedyStepwise();
           search.setSearchBackwards(true);
           attsel.setEvaluator(eval);
           attsel.setSearch(search);
           attsel.SelectAttributes(data);

           int[] indices = attsel.selectedAttributes();
           System.out.println(Utils.arrayToString(indices));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
