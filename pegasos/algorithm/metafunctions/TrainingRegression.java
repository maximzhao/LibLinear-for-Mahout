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

import java.util.Random;
import java.util.Vector;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;

/**
 * SVM Regression with Linear Kernel
 * 
 */
public class TrainingRegression extends Training {
  
  @Override
  public WeightVector training(DataSetHandler trainDataset, SVMParameters para) {
    
    double eta = 0.0;
    
    // create the weight
    WeightVector w = new WeightVector(Integer.MAX_VALUE);
    Random rand = new Random();
    
    int r = 0;
    // main loop
    for (int i = 0; i < para.getMaxIter(); ++i) {
      // learning rate
      // pegasos eta rule
      if (0 == para.getEtaRuleType()) {
        eta = 1 / (para.getLambda() * (i + 2));
      } else if (1 == para.getEtaRuleType()) {
        eta = para.getEtaConstant() / Math.sqrt(i + 2);
        w.setAtoOne();
      } else {
        eta = para.getEtaConstant();
      }
      
      Vector<Integer> gradIndex = new Vector<Integer>();
      Vector<Double> gradWeight = new Vector<Double>();
      
      // calc sub-gradients
      for (int j = 0; j < para.getExamplesPerIter(); ++j) {
        // choose example randomly
        if (para.getRandomPool() != null) {
          r = para.getRandomPool().get(i);
        } else {
          r = rand.nextInt(trainDataset.getLabels().size());
        }
        
        // calculate prediction
        double prediction = w.times(trainDataset.getDataset().getRow(r));
        
        double curLoss = trainDataset.getLabels().get(r) - prediction;
        int lossSign = curLoss > 0 ? 1 : -1;
        // |y - < w, x> | != 0 -> A+ set.
        if (0.0 != curLoss) {
          gradIndex.add(r);
          // sign(y - < w, x >)
          gradWeight.add(eta * lossSign / para.getExamplesPerIter());
        }
      }
      
      // scale w
      w.scale(1.0 - eta * para.getLambda());
      
      // and add sub-gradients
      for (int j = 0; j < gradIndex.size(); ++j) {
        w.add(trainDataset.getDataset().getRow(gradIndex.get(j)), gradWeight.get(j)
            .doubleValue());
      }
      
      double norm2 = 0.0;
      
      // Project if needed
      if (0 == para.getProjectionRule()) {
        norm2 = w.getSnorm();
        if (norm2 > 1.0 / para.getLambda()) {
          w.scale(Math.sqrt(1.0 / (para.getLambda() * norm2)));
        }
      } else if (1 == para.getProjectionRule()) {
        norm2 = w.getSnorm();
        if (norm2 > (para.getProjectionConstant() * para.getProjectionConstant())) {
          w.scale(para.getProjectionConstant() / Math.sqrt(norm2));
        }
      } // else -- no projection
    }
    
    // Calculate objective value
    para.setNormValue(w.getSnorm());
    para.setObjValue(para.getNormValue() * para.getLambda() / 2.0);
    para.setLossValue(0.0);
    
    // use the random label
    if (para.getRandomPool() != null) {
      for (int i = 0; i < trainDataset.getLabels().size(); ++i) {
        double curLoss = Math.pow(trainDataset.getLabels().get(para.getRandomPool()
            .get(i))
                                  - w.times(trainDataset.getDataset()
                                      .getRow(para.getRandomPool().get(i))), 2);
        para.setLossValue(para.getLossValue() + curLoss);
      }
    } else {
      for (int i = 0; i < trainDataset.getLabels().size(); ++i) {
        double curLoss = Math.pow(trainDataset.getLabels().get(i)
                                  - w.times(trainDataset.getDataset().getRow(i)), 2);
        para.setLossValue(para.getLossValue() + curLoss);
      }
    }
    para.setLossValue(para.getLossValue() / trainDataset.getLabels().size());
    
    // save the model to file
    if (para.getModelFileName() != null) {
      w.writeToFile(para.getModelFileName());
    }
    return w;
  }
}
