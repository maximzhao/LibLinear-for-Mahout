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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.datastore.GeneralWriter;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;

/**
 * BinaryClassificationTesting process (For Binary Classification, useless now.)
 * It will read all prediction data set into memory, which could be problematic
 * in the case of large-scale data set prediction.
 * 
 * Such prediction only output accuracy of the trained model works on prediction
 * dataset. This function cannot support of predicting the label given un-seen
 * dataset.
 * 
 */
public class PredictionBinaryClassification extends Prediction {
  
  /**
   * Prediction process
   * 
   * @param testDataset
   * @param para
   * @throws IOException
   */
  @Override
  public void prediction(DataSetHandler testDataset, SVMParameters para) throws IOException {
    WeightVector w = new WeightVector(para.getModelFileName());
    
    // Calculate test_loss and test_error
    para.setTestLoss(0.0);
    para.setTestError(0.0);
    List<Integer> predictedLabelList = new ArrayList<Integer>();
    for (int i = 0; i < testDataset.getLabels().size(); ++i) {
      Double predictedLabel = w.times(testDataset.getDataset().getRow(i));
      predictedLabelList.add(predictedLabel > 0 ? 1 : -1);
      double curLoss = 1 - testDataset.getLabels().get(i) * predictedLabel;
      if (curLoss < 0.0) {
        curLoss = 0.0;
      }
      
      para.setTestLoss(para.getTestLoss() + curLoss);
      
      if (curLoss >= 1.0) {
        para.setTestError(para.getTestError() + 1.0);
      }
    }
    
    if (0 != testDataset.getLabels().size()) {
      para.setTestLoss(para.getTestLoss() / testDataset.getLabels().size());
      para.setTestError(para.getTestError() / testDataset.getLabels().size());
    }
    
    if (null != para.getOutFile()) {
      GeneralWriter.writer(para.getOutFile(), predictedLabelList);
    }
  }
}
