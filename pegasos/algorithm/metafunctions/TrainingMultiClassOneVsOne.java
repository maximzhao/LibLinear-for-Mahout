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

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SparseMatrix;

/**
 * SVM Sequential Multi-classification Training process, which uses linear
 * kernel. Implementing the One-vs.-one scheme.
 */
public class TrainingMultiClassOneVsOne extends Training {
  
  
  /**
   * SVM Sequential Multi-classification Training process, which uses linear
   * kernel. Implementing the One-vs.-one scheme.
   * 
   * @param trainDataset
   *          Training data set
   * @param para
   *          Parameters
   * @return weight vector
   */
  @Override
  public WeightVector training(DataSetHandler trainDataset, SVMParameters para) {
    
    // the main loop
    int pos = 0;
    String modelPath = para.getModelFileName();
    Training binaryClassifier = new TrainingBinaryClassification();
    
    // set the model name, shut down write one file for each model.
    para.setModelFileName(null);
    
    // get the all labels from data set.
    Double[] labelList = new Double[trainDataset.getUniqueLables().size()];
    int labelIndex = 0;
    for (Double a : trainDataset.getUniqueLables()) {
      labelList[labelIndex++] = a;
    }
    
    // ensure the random pool is null;
    para.setRandomPool(null);
    List<WeightVector> weightList = new ArrayList<WeightVector>();
    
    int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
    
    DataSetHandler twoClassData = new DataSetHandler("nofile");
    twoClassData.getUniqueLables().addAll(trainDataset.getUniqueLables());
    // currently, we only consider the small number of class.
    // one-vs-one. i-j classifier.
    for (int i = 0; i < para.getClassNum(); i++) {
      for (int j = i + 1; j < para.getClassNum(); j++) {
        
        // create a subset of samples.
        twoClassData.getLabels().clear();
        twoClassData.setDataset(new SparseMatrix(cardinality));
        
        pos = 0;
        // got related samples from pre-fetched data set.
        for (int index = 0; index < trainDataset.getLabels().size(); index++) {
          if (labelList[i].equals(trainDataset.getLabels().get(index))) {
            twoClassData.getLabels().put(pos, 1.0);
            twoClassData.getDataset().assignRow(pos, trainDataset.getDataset()
                .getRow(index));
            pos++;
          } else if (labelList[j].equals(trainDataset.getLabels().get(index))) {
            twoClassData.getLabels().put(pos, -1.0);
            twoClassData.getDataset().assignRow(pos, trainDataset.getDataset()
                .getRow(index));
            pos++;
          }
        }
        
        // train the model.
        weightList.add(binaryClassifier.training(twoClassData, para));
      }
    }
    // write the vector to file
    WeightVector.batchDumpModels(modelPath, weightList, para,
      trainDataset.getUniqueLables());
    return null;
  }
}
