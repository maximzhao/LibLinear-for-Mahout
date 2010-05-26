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
package org.apache.mahout.classifier.svm.datastore;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * General file system Writer
 * 
 */
public class GeneralWriter {
  
  public GeneralWriter() {
    
  }
  
  private static final Logger log = LoggerFactory
      .getLogger(GeneralWriter.class);
    
  /**
   * 
   * @param fileName
   * @param labels
   */
  public static void writer(String fileName, List labels) {
    try {
      BufferedWriter wr = new BufferedWriter(new FileWriter(fileName, true));
      for (int i = 0; i < labels.size(); i++) {
        wr.write(labels.get(i) + "\r\n");
      }
      wr.close();
    } catch (Exception e) {
      log.error("Exception" + e.getMessage());
    }
  }
}
