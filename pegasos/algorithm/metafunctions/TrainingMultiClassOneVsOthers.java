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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SVM Sequential Multi-classification Training process, which uses linear
 * kernel. Implementing the One-vs.-others scheme
 * 
 */
public class TrainingMultiClassOneVsOthers extends Training {
  
  private static final Logger log = LoggerFactory
      .getLogger(TrainingMultiClassOneVsOthers.class);
  
  /**
   * SVM Sequential Multi-classification Training process, which uses linear
   * kernel. Implementing the One-vs.-others scheme.
   * 
   * @param trainDataset
   * @param para
   * @return
   */
  @Override
  public WeightVector training(DataSetHandler trainDataset, SVMParameters para) {
    
    Training binaryClassifier = new TrainingBinaryClassification();
    
    String modelPath = para.getModelFileName();
    Double[] labelList = new Double[trainDataset.getUniqueLables().size()];
    int labelIndex = 0;
    for (Double a : trainDataset.getUniqueLables()) {
      labelList[labelIndex++] = a;
    }
    
    // ensure the random pool is null;
    para.setRandomPool(null);
    List<WeightVector> weightList = new ArrayList<WeightVector>();
    
    // one- vs.-others training
    Map<Integer,Double> originalIndex = new HashMap<Integer,Double>();
    
    originalIndex.putAll(trainDataset.getLabels());
    
    // the main loop for multi-class training
    for (int i = 0; i < para.getClassNum(); i++) {
      // process the labels.
      for (int index = 0; index < originalIndex.size(); index++) {
        if (labelList[i].equals(originalIndex.get(index))) {
          trainDataset.getLabels().put(index, 1.0);
        } else {
          trainDataset.getLabels().put(index, -1.0);
        }
      }
      
      // train the model.
      weightList.add(binaryClassifier.training(trainDataset, para));
      log.info("Training: " + i);
    }
    
    // write the vector to file
    WeightVector.batchDumpModels(modelPath, weightList, para,
      trainDataset.getUniqueLables());
    
    return null;
  }
}
