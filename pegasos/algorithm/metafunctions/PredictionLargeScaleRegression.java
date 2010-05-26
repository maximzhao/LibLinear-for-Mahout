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
import java.util.List;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.datastore.GeneralWriter;
import org.apache.mahout.classifier.svm.datastore.LibsvmFormatParser;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sequential Regression BinaryClassificationTesting process Under same
 * situation with LargeScaleSequentialTesting above.
 * 
 */
public class PredictionLargeScaleRegression extends Prediction {
  
  private static final Logger log = LoggerFactory
      .getLogger(PredictionLargeScaleRegression.class);
  
  /**
   * Sequential Regression BinaryClassificationTesting process Under same
   * situation with LargeScaleSequentialTesting above.
   * 
   * @param dataset
   *          Data set
   * @param para
   *          parameters
   * @throws IOException
   */
  @Override
  public void prediction(DataSetHandler dataset, SVMParameters para) throws IOException {
    // Read the w from model file
    WeightVector w = new WeightVector(para.getModelFileName());
    
    BufferedReader in = null;
    para.setTestLoss(0.0);
    para.setTestError(0.0);
    String str = null;
    int rowNum = 0;
    double curLoss = 0.0;
    double label = 0;
    SequentialAccessSparseVector row = null;
    List<Double> predictedLabelList = new ArrayList<Double>();
    Double predictedLabel = 0.0;
    try {
      in = new BufferedReader(new FileReader(dataset.getFileName()));
      log.info("Sequential Testing (one point denotes 1000 samples!");
      
      while ((str = in.readLine()) != null) {
        row = new SequentialAccessSparseVector(Integer.MAX_VALUE, 10);
        
        label = LibsvmFormatParser.str2Vector(str, row);
        
        // empty line or do NOT have label, continue.
        if (row.size() < 1 || label == Double.MAX_VALUE) {
          continue;
        }
        
        predictedLabel = w.times(row);
        predictedLabelList.add(predictedLabel);
        
        curLoss = Math.pow(label - predictedLabel, 2);
        para.setTestLoss(para.getTestLoss() + curLoss);
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
      para.setTestLoss(para.getTestLoss() / rowNum);
    }
    
    if (null != para.getOutFile()) {
      GeneralWriter.writer(para.getOutFile(), predictedLabelList);
    }
  }
}
