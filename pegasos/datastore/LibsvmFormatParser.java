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

import java.util.regex.Pattern;

import org.apache.mahout.math.Vector;

/**
 * 
 */
public class LibsvmFormatParser {
  
  public LibsvmFormatParser() {
    
  }
  
  /**
   * String to SparseVector converter for sample string to vector converting
   * 
   * @param str
   *          sample's string ( label index:value ... )
   * @param row
   *          SparseVector.
   * @return label label of this sample, if the sample does NOT have label,
   *         return -Double.MAX_VALUE
   * @throws NullInputString
   */
  public static double str2Vector(String str, Vector row) throws NullInputString {
    
    if (null == str) {
      throw new NullInputString();
    }
    
    String temp = null;
    String[] array = null;
    String[] labelTest = null;
    int key = 0;
    double v = 0.0;
    double label = 0;
    int i = 0;
    
    Pattern splitter = Pattern.compile("\\s++");
    if (0 == str.indexOf("#")) {
      return 0;
    } else if (2 == str.split("#").length) {
      temp = str.split("#")[0];
    } else {
      temp = str;
    }
    
    temp.trim();
    // handle matlab data set :( e.g. 2: .4
    array = splitter.split(temp.replace(": .", ":0."));
    
    // skip the empty header.
    while (array[i].isEmpty()) {
      i++;
    }
    
    labelTest = array[i].replace("+", "").replace(" ", "").split(":");
    if (1 == labelTest.length) {
      label = Double.parseDouble(labelTest[0]);
    } else { // do NOT have label
      label = -Double.MAX_VALUE;
      i--;
    }
    
    for (int j = i + 1; j < array.length; j++) {
      key = Integer.parseInt(array[j].split(":")[0]);
      v = Double.parseDouble(array[j].split(":")[1]);
      row.setQuick(key, v);
    }
    return label;
  }
}
