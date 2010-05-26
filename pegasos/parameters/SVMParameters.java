/*
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  under the License.
 */
package org.apache.mahout.classifier.svm.parameters;

import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class SVMParameters {

  private static final Logger log = LoggerFactory.getLogger(SVMParameters.class);
  
  // For HDFS
  public static final String HADOOP_MODLE_PATH = "para.mahout.classifier.svm.model.path";
  public static final String HADOOP_MAX_ITERATION = "para.mahout.classifier.svm.max.iteration";
  public static final String HADOOP_SAMPLE_NUMBER = "para.mahout.classifier.svm.sample.number";
  public static final String HADOOP_CLASS_NUMBER = "para.mahout.classifier.svm.class.number";
  public static final String HADOOP_MULTI_CLASS_TYPE = "para.mahout.classifier.svm.multiclass.type";
  public static final String HADOOP_LAMBDA = "para.mahout.classifier.svm.lambda";
  public static final String HADOOP_K = "para.mahout.classifier.svm.k";
  public static final String HDFS_SERVER = "para.mahout.classifier.svm.hdfs.server";
  public static final String HADOOP_STARTING_CLASS_INDEX = "para.mahout.classifier.svm.starting.class.index";
  public static final String DEFAULT_HDFS_SERVER = "hdfs://localhost:12009";
  // For HBASE
  public static final String DEFAULT_HBASE_SERVER = "localhost:60000";
  // Label for WeightVector dump to file
  public static final String CLASSIFICATION_TYPE = "ClassificationType";
  public static final String LABELS = "Labels";
  public static final String DIMENSION = "Dimension";
  public static final String SNORM = "Snorm";
  public static final String CLASS_NUMBER = "ClassNum";
  public static final String A = "A";
  public static final String W = "W";
  
  // Parameters
  private double lambda = 0.001;
  private int maxIter = (int) (100 / getLambda());
  private int examplesPerIter = 1;
  private int averageIteration;
  private String modelFileName;
  
  // Output variables
  private long trainTime;
  private long calcObjTime;
  private double objValue;
  private double normValue;
  private double lossValue;
  private double zeroOneError;
  private double testLoss;
  private double testError;
  private double testAccuracy;
  
  // additional parameters
  private int etaRuleType;
  private double etaConstant;
  private int projectionRule;
  private double projectionConstant;
  private int trainSampleNumber; // currently, can support 21 billion samples.
  private List<Integer> randomPool;
  private List<Integer> randomPoolsorted;
  
  // For Map/Reduce framework
  private String hdfsServerAddr = "hdfs://localhost:12009";
  private double epsilon = 0.1;
  private int classificationType; // default 0: classification, 1: regression,
                                  // 2: multi-classification(one vs. one) 3:
                                  // multi-classification(one vs. others)
  private String trainFile;
  private String outFile;
  private int classNum; // class number
  private long mapSplitSize = 1000000;
  private int numberReducers = 1; // reducer's number
  private String maxHeapSize = "-Xmx500M";
  private long taskTimeout = 1080;
  private String testFile;
  private int startingClassIndex;
  private int validateExampleNumber;
  
  public void report() {
    StringBuffer output = new StringBuffer();
    output.append(String.valueOf(this.getNormValue()) + " = Norm of solution");
    output.append("\n");
    output.append(String.valueOf(this.getLossValue())
                       + "  = avg Loss of solution");
    output.append("\n");
    output.append(String.valueOf(this.getZeroOneError())
                       + "  = avg zero-one error of solution");
    output.append("\n");
    output.append(String.valueOf(this.getObjValue())
                       + " = primal objective of solution");
    output.append("\n");
    output.append(String.valueOf(this.getTestLoss()) + " = avg Loss over test");
    output.append("\n");
    output.append(String.valueOf(this.getTestError())
                       + " = avg zero-one error over test");
    output.append("\n");
    output.append(String.valueOf((1 - this.getTestError()) * 100)
                       + "% = Testing Accuracy ");
    output.append("\n");
    log.info(output.toString());
  }
  
  public void trainReport() {
    StringBuffer output = new StringBuffer();
    output.append(String.valueOf(this.getNormValue()) + " = Norm of solution");
    output.append("\n");
    output.append(String.valueOf(this.getLossValue())
                       + "  = avg Loss of solution");
    output.append("\n");
    output.append(String.valueOf(this.getZeroOneError())
                       + "  = avg zero-one error of solution");
    output.append("\n");
    output.append(String.valueOf(this.getObjValue())
                       + " = primal objective of solution");
    output.append("\n");
    log.info(output.toString());

  }
  
  public void trainRegressionReport() {
    StringBuffer output = new StringBuffer();
    output.append(String.valueOf(this.getNormValue()) + " = Norm of solution");
    output.append("\n");
    output.append(String.valueOf(this.getLossValue())
                       + "  = avg Loss of solution");
    output.append("\n");
    output.append(String.valueOf(this.getObjValue())
                       + " = primal objective of solution");
    output.append("\n");
    log.info(output.toString());
  }
  
  public void testReport() {
    StringBuffer output = new StringBuffer();
    output.append(String.valueOf(this.getTestLoss()) + " = avg Loss over test");
    output.append("\n");
    output.append(String.valueOf(this.getTestError())
                       + " = avg zero-one error over test");
    output.append("\n");
    output.append(String.valueOf((1 - this.getTestError()) * 100)
                       + "% = Testing Accuracy ");
    output.append("\n");
    log.info(output.toString());
  }
  
  public void testRegressionReport() {
    log.info(String.valueOf(this.getTestLoss()) + " = avg Loss over test");
  }
  
  public void report(int type) {
    if (getTrainFile() != null) {
      switch (type) {
        case 0:
          // binary classification
          trainReport();
          break;
        case 1:
          // regression training
          trainRegressionReport();
          break;
        case 2:
          // multiple classification
          trainReport();
          break;
        case 3:
          // multiple classification
          trainReport();
          break;
        default:
          trainReport();
          break;
      }
    } else if (getTestFile() != null) {
      switch (type) {
        case 0:
          // binary classification
          testReport();
          break;
        case 1:
          // regression
          testRegressionReport();
          break;
        case 2:
          // multiple classification one-vs-one
          testReport();
          break;
        case 3:
          // multiple classification one-vs-others.
          testReport();
          break;
        default:
          trainReport();
          break;
      }
    }
  }

  public void setMaxIter(int maxIter) {
    this.maxIter = maxIter;
  }

  public int getMaxIter() {
    return maxIter;
  }

  public void setExamplesPerIter(int examplesPerIter) {
    this.examplesPerIter = examplesPerIter;
  }

  public int getExamplesPerIter() {
    return examplesPerIter;
  }

  public void setAverageIteration(int averageIteration) {
    this.averageIteration = averageIteration;
  }

  public int getAverageIteration() {
    return averageIteration;
  }

  public void setModelFileName(String modelFileName) {
    this.modelFileName = modelFileName;
  }

  public String getModelFileName() {
    return modelFileName;
  }

  public void setTrainTime(long trainTime) {
    this.trainTime = trainTime;
  }

  public long getTrainTime() {
    return trainTime;
  }

  public void setCalcObjTime(long calcObjTime) {
    this.calcObjTime = calcObjTime;
  }

  public long getCalcObjTime() {
    return calcObjTime;
  }

  public void setObjValue(double objValue) {
    this.objValue = objValue;
  }

  public double getObjValue() {
    return objValue;
  }

  public void setNormValue(double normValue) {
    this.normValue = normValue;
  }

  public double getNormValue() {
    return normValue;
  }

  public void setLossValue(double lossValue) {
    this.lossValue = lossValue;
  }

  public double getLossValue() {
    return lossValue;
  }

  public void setZeroOneError(double zeroOneError) {
    this.zeroOneError = zeroOneError;
  }

  public double getZeroOneError() {
    return zeroOneError;
  }

  public void setTestLoss(double testLoss) {
    this.testLoss = testLoss;
  }

  public double getTestLoss() {
    return testLoss;
  }

  public void setTestError(double testError) {
    this.testError = testError;
  }

  public double getTestError() {
    return testError;
  }

  public void setTestAccuracy(double testAccuracy) {
    this.testAccuracy = testAccuracy;
  }

  public double getTestAccuracy() {
    return testAccuracy;
  }

  public void setEtaRuleType(int etaRuleType) {
    this.etaRuleType = etaRuleType;
  }

  public int getEtaRuleType() {
    return etaRuleType;
  }

  public void setEtaConstant(double etaConstant) {
    this.etaConstant = etaConstant;
  }

  public double getEtaConstant() {
    return etaConstant;
  }

  public void setProjectionRule(int projectionRule) {
    this.projectionRule = projectionRule;
  }

  public int getProjectionRule() {
    return projectionRule;
  }

  public void setProjectionConstant(double projectionConstant) {
    this.projectionConstant = projectionConstant;
  }

  public double getProjectionConstant() {
    return projectionConstant;
  }

  public void setTrainSampleNumber(int trainSampleNumber) {
    this.trainSampleNumber = trainSampleNumber;
  }

  public int getTrainSampleNumber() {
    return trainSampleNumber;
  }

  public void setRandomPool(List<Integer> randomPool) {
    this.randomPool = randomPool;
  }

  public List<Integer> getRandomPool() {
    return randomPool;
  }

  public void setRandomPoolsorted(List<Integer> randomPoolsorted) {
    this.randomPoolsorted = randomPoolsorted;
  }

  public List<Integer> getRandomPoolsorted() {
    return randomPoolsorted;
  }

  public void setHdfsServerAddr(String hdfsServerAddr) {
    this.hdfsServerAddr = hdfsServerAddr;
  }

  public String getHdfsServerAddr() {
    return hdfsServerAddr;
  }

  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }

  public double getEpsilon() {
    return epsilon;
  }

  public void setClassificationType(int classificationType) {
    this.classificationType = classificationType;
  }

  public int getClassificationType() {
    return classificationType;
  }

  public void setTrainFile(String trainFile) {
    this.trainFile = trainFile;
  }

  public String getTrainFile() {
    return trainFile;
  }

  public void setOutFile(String outFile) {
    this.outFile = outFile;
  }

  public String getOutFile() {
    return outFile;
  }

  public void setClassNum(int classNum) {
    this.classNum = classNum;
  }

  public int getClassNum() {
    return classNum;
  }

  public void setMapSplitSize(long mapSplitSize) {
    this.mapSplitSize = mapSplitSize;
  }

  public long getMapSplitSize() {
    return mapSplitSize;
  }

  public void setNumberReducers(int numberReducers) {
    this.numberReducers = numberReducers;
  }

  public int getNumberReducers() {
    return numberReducers;
  }

  public void setMaxHeapSize(String maxHeapSize) {
    this.maxHeapSize = maxHeapSize;
  }

  public String getMaxHeapSize() {
    return maxHeapSize;
  }

  public void setTaskTimeout(long taskTimeout) {
    this.taskTimeout = taskTimeout;
  }

  public long getTaskTimeout() {
    return taskTimeout;
  }

  public void setTestFile(String testFile) {
    this.testFile = testFile;
  }

  public String getTestFile() {
    return testFile;
  }

  public void setStartingClassIndex(int startingClassIndex) {
    this.startingClassIndex = startingClassIndex;
  }

  public int getStartingClassIndex() {
    return startingClassIndex;
  }

  public void setValidateExampleNumber(int validateExampleNumber) {
    this.validateExampleNumber = validateExampleNumber;
  }

  public int getValidateExampleNumber() {
    return validateExampleNumber;
  }

  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  public double getLambda() {
    return lambda;
  }
}
