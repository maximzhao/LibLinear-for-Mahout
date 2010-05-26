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

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;

/**
 * Regression BinaryClassificationTesting process (it will load all data to
 * memory) In the same situation with BinaryClassificationTesting above.
 * (useless)
 */
public class PredictionRegression extends Prediction {
  
  /**
   * Regression BinaryClassificationTesting process (it will load all data to
   * memory) In the same situation with BinaryClassificationTesting above.
   * (useless)
   * 
   * @param testDataset
   *          the prediction dataset
   * @param para
   *          parameters
   * @throws IOException
   */
  @Override
  public void prediction(DataSetHandler testDataset, SVMParameters para) throws IOException {
    WeightVector w = new WeightVector(para.getModelFileName());
    
    // Calculate test_loss and test_error
    para.setTestLoss(0.0);
    for (int i = 0; i < testDataset.getLabels().size(); ++i) {
      double curLoss = Math.pow(testDataset.getLabels().get(i)
                                - w.times(testDataset.getDataset().getRow(i)), 2);
      
      para.setTestLoss(para.getTestLoss() + curLoss);
      
    }
    
    if (0 != testDataset.getLabels().size()) {
      para.setTestLoss(para.getTestLoss() / testDataset.getLabels().size());
    }
  }
}
