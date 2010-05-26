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
import java.util.Random;
import java.util.Vector;

import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;

/**
 * SVM with linear kernel (Pegasos) BinaryClassificationTraining
 */
public class TrainingBinaryClassificationwithValidation extends Training {
    
  /**
   * SVM (Pegasos) BinaryClassificationTraining
   * 
   * @param trainDataset
   *          train dataset
   * @param para
   *          contains parameters for training and some useful results
   * @return trained w
   */
  @Override
  public WeightVector training(DataSetHandler trainDataset, SVMParameters para) {
    
    // create the weight
    WeightVector w = new WeightVector(Integer.MAX_VALUE);
    WeightVector bestW = new WeightVector(Integer.MAX_VALUE);
    double bestObj = 1.0;
    List<Integer> validateIndices = new ArrayList<Integer>();
    Random rand = new Random();
    double eta = 0.0;
    for (int i = 0; i < para.getValidateExampleNumber(); i++) {
      if (para.getRandomPool() != null) {
        validateIndices.add(para.getRandomPool().get(i));
      } else {
        validateIndices.add(rand.nextInt(trainDataset.getLabels().size()));
      }
    }
    
    // Choose s random indices
    int s = 5; // corresponds to confidence of 0.9933
    int blockSize = para.getMaxIter() / s;
    
    List<Integer> candidates = new ArrayList<Integer>();
    for (int i = 0; i < s - 1; i++) {
      candidates.add(blockSize * i + rand.nextInt(blockSize));
    }
    candidates.add(para.getMaxIter() - 1);
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
        // choose random example
        if (para.getRandomPool() != null) {
          r = para.getRandomPool().get(i);
        } else {
          r = rand.nextInt(trainDataset.getLabels().size());
        }
        
        // calculate prediction
        double prediction = w.times(trainDataset.getDataset().getRow(r));
        
        // calculate loss
        double curLoss = 1 - trainDataset.getLabels().get(r) * prediction;
        if (curLoss < 0.0) {
          curLoss = 0.0;
        }
        
        // and add to the gradient
        if (curLoss > 0.0) {
          gradIndex.add(r);
          gradWeight.add(eta * trainDataset.getLabels().get(r)
                         / para.getExamplesPerIter());
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
      
      // and validate
      if (candidates.contains(i)) {
        double obj = 0.0;
        int index = 0;
        for (int j = 0; j < validateIndices.size(); j++) {
          index = validateIndices.get(j);
          double curLoss = 1 - trainDataset.getLabels().get(index)
                           * w.times(trainDataset.getDataset().getRow(index));
          if (curLoss < 0.0) {
            curLoss = 0.0;
          }
          obj += curLoss;
        }
        
        obj /= validateIndices.size();
        obj += para.getLambda() / 2.0 * w.getSnorm();
        
        if (obj <= bestObj) {
          bestW = w;
          bestObj = obj;
        }
      }
    }
    
    // Calculate objective value
    para.setNormValue(bestW.getSnorm());
    para.setObjValue(para.getNormValue() * para.getLambda() / 2.0);
    para.setLossValue(0.0);
    para.setZeroOneError(0.0);
    // use the random label
    if (para.getRandomPool() != null) {
      for (int i = 0; i < trainDataset.getLabels().size(); ++i) {
        
        double curLoss = 1
                         - trainDataset.getLabels().get(para.getRandomPool().get(i))
                         * bestW.times(trainDataset.getDataset()
                             .getRow(para.getRandomPool().get(i)));
        if (curLoss < 0.0) {
          curLoss = 0.0;
        }
        para.setLossValue(para.getLossValue() + curLoss);
        para.setObjValue(para.getObjValue() + curLoss);
        if (curLoss >= 1.0) {
          para.setZeroOneError(para.getZeroOneError() + 1.0);
        }
      }
    } else {
      for (int i = 0; i < trainDataset.getLabels().size(); ++i) {
        double curLoss = 1 - trainDataset.getLabels().get(i)
                         * bestW.times(trainDataset.getDataset().getRow(i));
        if (curLoss < 0.0) {
          curLoss = 0.0;
        }
        para.setLossValue(para.getLossValue() + curLoss);
        para.setObjValue(para.getObjValue() + curLoss);
        if (curLoss >= 1.0) {
          para.setZeroOneError(para.getZeroOneError() + 1.0);
        }
      }
    }
    
    para.setLossValue(para.getLossValue() / trainDataset.getLabels().size());
    para.setObjValue(para.getObjValue() / trainDataset.getLabels().size());
    para.setZeroOneError(para.getZeroOneError() / trainDataset.getLabels().size());
    
    // save the model to file
    if (para.getModelFileName() != null && null == para.getHdfsServerAddr()) {
      bestW.writeToFile(para.getModelFileName(), para.getClassNum(),
        trainDataset.getUniqueLables());
    }
    return bestW;
  }
}
