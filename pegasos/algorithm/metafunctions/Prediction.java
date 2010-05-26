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
import java.util.Map;

import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;

/**
 * 
 * 
 */
public abstract class Prediction {
  
  public abstract void prediction(DataSetHandler dataset, SVMParameters para) throws IOException;
  
  public int oneLineClassifier(Map<String,WeightVector> weightList,
                               SequentialAccessSparseVector row,
                               Integer[] labels,
                               int label) {
    return 0;
  }
  
  public int oneLineClassifier(Map<String,WeightVector> weightList,
                               SequentialAccessSparseVector row,
                               Integer[] labels) {
    return 0;
  }
  
  public String oneLineClassifier(Map<String,WeightVector> weightList,
                                  Text value) {
    return null;
  }
}
