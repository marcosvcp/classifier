package com.classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

public class ImagesSetup {


    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";
    private static final int CAPACITY = 257;
    private static final int INDEX = 256;

    public static void main(String[] a) throws Exception {
        String pathname; pathname = "/media/sda3/teste/digitos/digitos_3.jpg";
        File f = new File(pathname);
        try {
            System.out.println(Arrays.toString(buildHistogram(f)));
        } catch (Exception e) {
            e.printStackTrace();
        }
        FastVector wekaAttributes = new FastVector(CAPACITY);
        double[] histogram = buildHistogram(f);
        for (int i = 0; i < histogram.length ; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttributes.addElement(attr);
        }
        FastVector classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);
        Attribute attr = new Attribute("classes",classes);

        wekaAttributes.addElement(attr);


        Classifier cModel = new NaiveBayes();
        Instances trainingSet = createTrainingSet(wekaAttributes, histogram);
        cModel.buildClassifier(trainingSet);

        Evaluation eTest = new Evaluation(trainingSet);
        //TODO Mudar testingSet para ser usado na base de dados
        Instances testingSet = trainingSet;
        eTest.evaluateModel(cModel,testingSet);

        String strSummary = eTest.toSummaryString();
        System.out.println(strSummary);
    }

    private static Instances createTrainingSet(FastVector wekaAttributes, double[] histogram) {
        Instances isTrainingSet = new Instances("Rel", wekaAttributes, 1);
        isTrainingSet.setClassIndex(INDEX);

        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length ; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX), DIGITOS);
        isTrainingSet.add(imageInstance);
        return isTrainingSet;

    }

//////////////// helper code /////////////////////////

    private static final double LUMINANCE_RED = 0.299D;
    private static final double LUMINANCE_GREEN = 0.587D;
    private static final double LUMINANCE_BLUE = 0.114;
    private static final int HIST_WIDTH = 256;
    private static final int HIST_HEIGHT = 100;

    /**
     * Parses pixels out of an image file, converts the RGB values to
     * its equivalent grayscale value (0-255), then constructs a
     * histogram of the percentage of counts of grayscale values.
     *
     * @param infile - the image file.
     * @return - a histogram of grayscale percentage counts.
     */
    protected static double[] buildHistogram(File infile) throws Exception {
        BufferedImage input = ImageIO.read(infile);
        int width = input.getWidth();
        int height = input.getHeight();
        List<Integer> graylevels = new ArrayList<Integer>();
        double maxWidth = 0.0D;
        double maxHeight = 0.0D;
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                Color c = new Color(input.getRGB(row, col));
                int graylevel = (int) (LUMINANCE_RED * c.getRed() +
                        LUMINANCE_GREEN * c.getGreen() +
                        LUMINANCE_BLUE * c.getBlue());
                graylevels.add(graylevel);
                maxHeight++;
                if (graylevel > maxWidth) {
                    maxWidth = graylevel;
                }
            }
        }
        double[] histogram = new double[HIST_WIDTH];
        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
            int idx = graylevel;
            histogram[idx] +=
                    Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / maxHeight;
        }
        return histogram;
    }
}