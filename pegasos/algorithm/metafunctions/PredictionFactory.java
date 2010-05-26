package org.apache.mahout.classifier.svm.algorithm.metafunctions;

public class PredictionFactory {
  /**
   * 
   * @param type
   *          the type of classifier
   * @return
   */
  public static Prediction getInstance(int type) {
    // Classification training
    switch (type) {
      case 0:
        return new PredictionLargeScaleDataset();
      case 1:
        // regression training
        return new PredictionLargeScaleRegression();
      case 2:
        return new PredictionMultiClassOneVsOne();
      case 3:
        return new PredictionMultiClassOneVsOthers();
      default:
        throw new RuntimeException();
    }
  }
}
