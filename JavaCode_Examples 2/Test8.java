import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

public class Test8
{
    public static void main(String args[])
    {
        try {
            String[] options = new String[2];
            options[0] = "-t";
            options[1] = "../data/breast-cancer.arff";
            System.out.println(Evaluation.evaluateModel(new J48(), options));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
