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
package org.apache.mahout.classifier.svm.algorithm.metafunctions;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.datastore.GeneralWriter;
import org.apache.mahout.classifier.svm.datastore.LibsvmFormatParser;
import org.apache.mahout.classifier.svm.datastore.NullInputString;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sequential Multi-classification, one-vs.-others.
 */
public class PredictionMultiClassOneVsOthers extends Prediction {
  
  private static final Logger log = LoggerFactory
      .getLogger(PredictionMultiClassOneVsOne.class);
  
  /**
   * Sequential Multi-classification, one-vs.-others.
   * 
   * @param testDataset
   *          Test data set
   * @param para
   * @throws IOException
   */
  @Override
  public void prediction(DataSetHandler testDataset, SVMParameters para) throws IOException {
    String modelPath = para.getModelFileName();
    Map<String,WeightVector> weightList = new HashMap<String,WeightVector>();
    
    if (null != para.getHdfsServerAddr()) {
      WeightVector.getBatchModels(para.getHdfsServerAddr(), modelPath, weightList);
    } else {
      WeightVector.getBatchModels(modelPath, weightList);
    }
    
    Set<Double> labelList = null;
    // get label set
    for (Entry<String,WeightVector> entry : weightList.entrySet()) {
      labelList = entry.getValue().getLabels();
      break;
    }
    Integer[] labels = new Integer[labelList.size()];
    int k = 0;
    for (Double l : labelList) {
      labels[k++] = l.intValue();
    }
    
    BufferedReader in = null;
    para.setTestLoss(0.0);
    para.setTestError(0.0);
    String str = null;
    int rowNum = 0;
    int trueLabel = -Integer.MAX_VALUE;
    SequentialAccessSparseVector row = null;
    List<Integer> predictedLabelList = new ArrayList<Integer>();
    int predictedLabel = 0;
    try {
      in = new BufferedReader(new FileReader(testDataset.getFileName()));
      log.info("Sequential Testing ...");
      
      while ((str = in.readLine()) != null) {
        row = new SequentialAccessSparseVector(Integer.MAX_VALUE, 10);
        trueLabel = (int) LibsvmFormatParser.str2Vector(str, row);
        
        // empty line or do NOT have label, continue.
        if (row.size() < 1 || trueLabel == Double.MAX_VALUE) {
          continue;
        }
        
        predictedLabel = oneLineClassifier(weightList, row, labels);
        predictedLabelList.add(predictedLabel);
        
        para.setTestError(para.getTestError()
                             + ((predictedLabel == trueLabel) ? 0 : 1));
        
        rowNum++;
        
        if (0 == rowNum % 1000) {
          log.info("Echo line: " + rowNum);
        } else if (0 == rowNum % 10000) {
          if (null != para.getOutFile()) {
            GeneralWriter.writer(para.getOutFile(), predictedLabelList);
            predictedLabelList.clear();
          }
        }
      }
      
      in.close();
    } catch (Exception e) {
      log.error("read file error: " + e.getMessage());
      return;
    }
    
    if (0 != rowNum) {
      para.setTestError(para.getTestError() / rowNum);
    }
    
    if (null != para.getOutFile()) {
      GeneralWriter.writer(para.getOutFile(), predictedLabelList);
    }
    
  }
  
  /**
   * Multiply classifier Multi-classification. Given a list of trained models,
   * and a sample, this function will predict the label and return whether the
   * prediction is correct or not.
   * 
   * @param weightList
   *          List of trained weights for each class.
   * @param row
   *          one feature vector for prediction
   * @param label
   * @param labels
   * @return return correct -> 0, incorrect -> 1
   */
  @Override
  public int oneLineClassifier(Map<String,WeightVector> weightList,
                               SequentialAccessSparseVector row,
                               Integer[] labels,
                               int label) {
    
    // classifier's number.
    int classiferNum = weightList.size();
    double[] results = new double[classiferNum];
    String key = null;
    
    // compute the one-to-all probability
    for (int i = 0; i < labels.length; i++) {
      key = "W" + labels[i];
      results[i] = weightList.get(key).times(row);
    }
    
    // vote
    double[] vote = new double[classiferNum];
    for (int i = 0; i < classiferNum; i++) {
      if (results[i] > 0) { // this class
        vote[i] += results[i];
      } else {
        for (int j = 0; j < classiferNum; j++) { // otherwise divied the result
          // to other class
          if (j != i) {
            vote[j] += Math.abs(results[i]) / (classiferNum - 1);
          }
        }
      }
    }
    
    int voteMaxIndex = 0;
    for (int i = 1; i < classiferNum; i++) {
      if (vote[i] > vote[voteMaxIndex]) {
        voteMaxIndex = i;
      }
    }
    
    // correct return 0, otherwise return 1.
    if (labels[voteMaxIndex].intValue() == label) {
      return 0;
    } else {
      return 1;
    }
  }
  
  /**
   * Multiply classifier Multi-classification. Given a list of trained models,
   * and a sample, this function will predict the label and return whether the
   * prediction is correct or not.
   * 
   * @param weightList
   *          List of trained weights for each class.
   * @param row
   *          one feature vector for prediction
   * @param labels
   * @return return predicted label
   */
  @Override
  public int oneLineClassifier(Map<String,WeightVector> weightList,
                               SequentialAccessSparseVector row,
                               Integer[] labels) {
    
    // classifier's number.
    int classiferNum = weightList.size();
    double[] results = new double[classiferNum];
    
    String key = null;
    // compute the one-to-all probability
    for (int i = 0; i < labels.length; i++) {
      key = "W" + labels[i];
      results[i] = weightList.get(key).times(row);
    }
    
    // vote
    double[] vote = new double[classiferNum];
    for (int i = 0; i < classiferNum; i++) {
      if (results[i] > 0) { // this class
        vote[i] += results[i];
      } else {
        for (int j = 0; j < classiferNum; j++) { // otherwise divied the result
          // to other class
          if (j != i) {
            vote[j] += Math.abs(results[i]) / (classiferNum - 1);
          }
        }
      }
    }
    
    int voteMaxIndex = 0;
    for (int i = 1; i < classiferNum; i++) {
      if (vote[i] > vote[voteMaxIndex]) {
        voteMaxIndex = i;
      }
    }
    
    // correct return 0, otherwise return 1.
    return labels[voteMaxIndex].intValue();
  }
  
  /**
   * Multiply classifier Multi-classification. Given a list of trained models,
   * and a sample, this function will predict the label and return whether the
   * prediction is correct or not.
   * 
   * @param weightList
   *          List of trained weights for each class.
   * @param value
   *          one test sample
   * @return return correct -> 0, incorrect -> 1
   */
  @Override
  public String oneLineClassifier(Map<String,WeightVector> weightList,
                                  Text value) {
    // classifier's number.
    int classiferNum = weightList.size();
    int label = -1;
    double[] results = new double[classiferNum];
    SequentialAccessSparseVector row = new SequentialAccessSparseVector(
        Integer.MAX_VALUE, 10);
    
    if (value.getLength() < 1) {
      return null;
    } else {
      try {
        label = (int) LibsvmFormatParser.str2Vector(value.toString(), row);
      } catch (NullInputString e) {
        log.error(e.getMessage());
      }
    }
    
    // if the line does not follow the format of libsvm.
    if (row.size() < 1) {
      return null;
    }
    
    String key = null;
    
    // compute the one-to-all probability
    for (int i = 0; i < classiferNum; i++) {
      key = "W" + i;
      results[i] = weightList.get(key).times(row);
    }
    
    // vote
    double[] vote = new double[classiferNum];
    for (int i = 0; i < classiferNum; i++) {
      if (results[i] > 0) { // this class
        vote[i] += results[i];
      } else {
        for (int j = 0; j < classiferNum; j++) { // otherwise divied the result
          // to other class
          if (j != i) {
            vote[j] += Math.abs(results[i]) / (classiferNum - 1);
          }
        }
      }
    }
    
    int voteMaxIndex = 0;
    for (int i = 1; i < classiferNum; i++) {
      if (vote[i] > vote[voteMaxIndex]) {
        voteMaxIndex = i;
      }
    }
    
    String result = voteMaxIndex + "_";
    // correct return 0, otherwise return 1.
    if (voteMaxIndex == label) {
      result += 0;
    } else {
      result += 1;
    }
    
    return result;
  }
}
