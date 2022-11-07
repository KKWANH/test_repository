import java.util.Random;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;

public class Test10
{
    public static void main(String args[])
    {
        try {
           DataSource source = new DataSource("../data/breast-cancer.arff");
           Instances data = source.getDataSet();
           data.setClassIndex(data.numAttributes() - 1);

           AttributeSelection filter = new AttributeSelection();  
           CfsSubsetEval eval = new CfsSubsetEval();
           GreedyStepwise search = new GreedyStepwise();
           search.setSearchBackwards(true);
           filter.setEvaluator(eval);
           filter.setSearch(search);
           filter.setInputFormat(data);

           Instances newData = Filter.useFilter(data, filter);
           System.out.println(newData);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
