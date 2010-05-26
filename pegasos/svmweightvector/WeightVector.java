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
package org.apache.mahout.classifier.svm.svmweightvector;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.svm.datastore.HDFSReader;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 *
 */
public class WeightVector {
  // for multi-classification
  
  private static final Logger log = LoggerFactory.getLogger(WeightVector.class);
  private int classNum;
  private Set<Double> labels;
  private int d;
  private double myA;
  private double mySnorm;
  private Vector myVector;
  
  /**
   * Construction function
   */
  public WeightVector() {
    this.classNum = 0;
    this.labels = null;
    this.d = 0;
    this.myA = 1.0;
    this.mySnorm = 0.0;
    this.myVector = null;
  }
  
  /**
   * Constructing a WeightVector by given dimension
   * 
   * @param d
   *          dimension
   */
  public WeightVector(int d) {
    this.classNum = 0;
    this.labels = null;
    this.d = d;
    this.myA = 1.0;
    this.mySnorm = 0.0;
    this.myVector = new SequentialAccessSparseVector(d, 10);
  }
  
  /**
   * Constructing a WeightVector from file
   * 
   * @param fileName
   *          File stored weight vector
   * @throws IOException
   *           File not found or cannot be read
   */
  public WeightVector(String fileName) throws IOException {
    BufferedReader modelFile = new BufferedReader(new FileReader(fileName));
    Map<String,String> keyValue = new OpenHashMap<String,String>();
    String line = null;
    String[] vectorString = null;
    String[] labelList = null;
    
    // read all line from file
    while ((line = modelFile.readLine()) != null) {
      String[] words = line.trim().split("#");
      if (words.length > 1) {
        keyValue.put(words[0], words[1]);
      }
    }
    
    // process the variables.
    this.classNum = Integer.parseInt(keyValue.get(SVMParameters.CLASS_NUMBER));
    Integer.parseInt(keyValue
        .get(SVMParameters.CLASSIFICATION_TYPE));
    this.d = Integer.parseInt(keyValue.get(SVMParameters.DIMENSION));
    this.mySnorm = Double.parseDouble(keyValue.get(SVMParameters.SNORM));
    
    // label
    labelList = keyValue.get(SVMParameters.LABELS).trim().split(" ");
    if (labelList.length > 1) {
      this.labels = new TreeSet<Double>();
      for (int i = 0; i < labelList.length; i++) {
        this.labels.add(new Double(labelList[i]));
      }
    }
    
    // A
    this.myA = Double.parseDouble(keyValue.get(SVMParameters.A));
    
    // weight vector
    vectorString = keyValue.get(SVMParameters.W).split(" ");
    this.myVector = new SequentialAccessSparseVector(this.d, 10);
    for (int i = 0; i < vectorString.length; i++) {
      String[] iv = vectorString[i].split(":");
      this.myVector
          .setQuick(Integer.parseInt(iv[0]), Double.parseDouble(iv[1]));
    }
    
    modelFile.close();
    
  }
  
  /**
   * Constructing a WeightVector from HDFS
   * 
   * @param serverAddr
   * @param fileName
   * @throws IOException
   */
  public WeightVector(String serverAddr, String fileName) throws IOException {
    HDFSReader rd = new HDFSReader();
    rd.setServerAddress(serverAddr);
    Path file = new Path(fileName);
    
    List<String> lines = rd.readLines(file, 0, 3);
    Iterator<String> elt = lines.iterator();
    String line = null;
    String[] vectorString = null;
    Map<String,String> keyValue = new OpenHashMap<String,String>();
    String[] labelList = null;
    
    // read all line from file
    while (elt.hasNext()) {
      line = elt.next().toString();
      String[] words = line.trim().split("#");
      if (words.length > 1) {
        keyValue.put(words[0], words[1]);
      }
    }
    
    // process the variables.
    this.classNum = Integer.parseInt(keyValue.get(SVMParameters.CLASS_NUMBER));
    Integer.parseInt(keyValue
        .get(SVMParameters.CLASSIFICATION_TYPE));
    this.d = Integer.parseInt(keyValue.get(SVMParameters.DIMENSION));
    this.mySnorm = Double.parseDouble(keyValue.get(SVMParameters.SNORM));
    
    // label
    labelList = keyValue.get(SVMParameters.LABELS).trim().split(" ");
    if (labelList.length > 1) {
      this.labels = new TreeSet<Double>();
      for (int i = 0; i < labelList.length; i++) {
        this.labels.add(new Double(labelList[i]));
      }
    }
    
    // A
    this.myA = Double.parseDouble(keyValue.get(SVMParameters.A));
    
    // weight vector
    vectorString = keyValue.get(SVMParameters.W).split(" ");
    this.myVector = new SequentialAccessSparseVector(this.d, 10);
    for (int i = 0; i < vectorString.length; i++) {
      String[] iv = vectorString[i].split(":");
      this.myVector
          .setQuick(Integer.parseInt(iv[0]), Double.parseDouble(iv[1]));
    }
  }
  
  public void setAtoOne() {
    this.myVector.times(this.myA);
    this.myA = 1.0;
  }
  
  public void setVector(double[] value) {
    for (int i = 0; i < value.length; i++) {
      if (value[i] > 0) {
        this.myVector.setQuick(i, value[i]);
      }
    }
  }
  
  public void setAtoX(double x) {
    this.myA = x;
  }
  
  public void setSnorm(double x) {
    this.mySnorm = x;
  }
  
  public double getSnorm() {
    return this.mySnorm;
  }
  
  public int getDimension() {
    return this.d;
  }
  
  public int getclassNum() {
    return this.classNum;
  }
  
  public Set<Double> getLabels() {
    return this.labels;
  }
  
  public double getA() {
    return this.myA;
  }
  
  public Vector getVector() {
    return this.myVector;
  }
  
  // get weight
  public double get(int index) {
    if (index < this.d) {
      return this.myVector.get(index) * this.myA;
    } else {
      return 0.0;
    }
  }
  
  public void scale(double s) {
    this.mySnorm *= (s * s);
    if (0.0 != s) {
      this.myA *= s;
    } else {
      this.myA = 1.0;
      this.myVector.times(0.0);
    }
  }
  
  public void add(Vector x, double s) {
    double pred = 0.0;
    double xNorm = 0.0;
    Iterator<Vector.Element> iter = x.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      double value = elt.get() * s;
      xNorm += value * value;
      pred += 2.0 * this.myVector.getQuick(elt.index()) * value;
      this.myVector.setQuick(elt.index(), this.myVector.getQuick(elt.index())
                                          + value / this.myA);
    }
    this.mySnorm += xNorm + this.myA * pred;
  }
  
  public void add(WeightVector x, double s) {
    this.mySnorm = 0.0;
    Iterator<Vector.Element> iter = this.myVector.iterateAll();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      this.myVector.set(elt.index(), elt.get() * this.myA + x.get(elt.index())
                                     * s);
      this.mySnorm += this.myVector.get(elt.index())
                      * this.myVector.get(elt.index());
    }
    this.myA = 1.0;
  }
  
  // careful u * v [* this.myA]
  // public double times( SparseVector u, WeightVector v)
  // {
  // double outcome = 0.0;
  // // for test
  // outcome = u.times(v.myVector.times(this.myA)).zSum();
  // return outcome;
  // }
  // public double times( Vector u)
  // {
  // double outcome = 0.0;
  // outcome = u.times(this.myVector.times(this.myA)).zSum();
  // return outcome;
  // }
  // careful u * v [* this.myA]
  public double times(Vector u, WeightVector v) {
    double result = 0.0;
    Iterator<Vector.Element> elt = u.iterateNonZero();
    while (elt.hasNext()) {
      Vector.Element it = elt.next();
      result += it.get() * v.myVector.getQuick(it.index()) * this.myA;
    }
    return result;
  }
  
  public double times(WeightVector v, Vector u) {
    return times(u, v);
  }
  
  public double times(Vector u) {
    double result = 0.0;
    Iterator<Vector.Element> elt = u.iterateNonZero();
    while (elt.hasNext()) {
      Vector.Element it = elt.next();
      result += it.get() * this.myVector.getQuick(it.index()) * this.myA;
    }
    return result;
  }
  
  /**
   * Dump the weight vector to string for MapReduce framework.
   * 
   * @param classifierType
   *          The type of Classifier
   * @param classNumber
   *          The number of categories.
   * @param classLabel
   *          The unique Label of such data set.
   * @return The string of model
   */
  public String dumpToString(int classifierType, int classNumber, String classLabel) {
    
    StringBuffer temp = new StringBuffer();
    temp.append(SVMParameters.CLASSIFICATION_TYPE + "#" + classifierType
                + "\r\n");
    temp.append(SVMParameters.CLASS_NUMBER + "#" + classNumber + "\r\n");
    temp.append(SVMParameters.LABELS + "#" + 0 + "\r\n");
    temp
        .append(SVMParameters.DIMENSION + "#" + String.valueOf(this.d) + "\r\n");
    temp.append(SVMParameters.A + classLabel + "#" + String.valueOf(this.myA)
                + "\r\n");
    temp.append(SVMParameters.SNORM + "#" + String.valueOf(this.mySnorm)
                + "\r\n");
    temp.append(SVMParameters.W + classLabel + "#");
    Iterator<Vector.Element> iter = this.myVector.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      temp.append(elt.index() + ":" + elt.get() + " ");
    }
    temp.append("\r\n");
    return temp.toString();
  }
  
  /**
   * Output the weight vector
   */
  public void printNonZero() {
    Iterator<Vector.Element> elt = this.myVector.iterateNonZero();
    StringBuffer out = new StringBuffer();
    while (elt.hasNext()) {
      Vector.Element it = elt.next();
      out.append(it.index() + ":" + it.get() * this.myA + " ");
    }
    out.append("\n");
    log.info(out.toString());
  }
  
  /**
   * Dump the weight vector to file
   * 
   * @param fileName
   *          File name
   */
  public void writeToFile(String fileName) {
    // finally, print the model to the model_file
    if (fileName != null) {
      try {
        BufferedWriter modelFile = new BufferedWriter(new FileWriter(fileName));
        modelFile.write(SVMParameters.CLASSIFICATION_TYPE + "#" + 0 + "\r\n");
        modelFile.write(SVMParameters.CLASS_NUMBER + "#" + 0 + "\r\n");
        modelFile.write(SVMParameters.LABELS + "#" + 0 + "\r\n");
        modelFile.write(SVMParameters.DIMENSION + "#" + String.valueOf(this.d)
                        + "\r\n");
        modelFile.write(SVMParameters.A + "#" + String.valueOf(this.myA)
                        + "\r\n");
        modelFile.write(SVMParameters.SNORM + "#"
                        + String.valueOf(this.mySnorm) + "\r\n");
        StringBuffer temp = new StringBuffer();
        Iterator<Vector.Element> iter = this.myVector.iterateNonZero();
        while (iter.hasNext()) {
          Vector.Element elt = iter.next();
          temp.append(elt.index() + ":" + elt.get() + " ");
        }
        modelFile.write(SVMParameters.W + "#" + temp.toString() + "\r\n");
        modelFile.flush();
        modelFile.close();
      } catch (Exception e) {
        log.error("Write Model File Error:" + e.getMessage());
      }
    }
  }
  
  /**
   * Dump the weight vector to file. Currently, only binary classification use
   * this function.
   * 
   * @param fileName
   * @param classNumber
   *          the number of categories
   * @param uniqueLables
   *          a set of unique labels of such data set.
   */
  public void writeToFile(String fileName,
                          int classNumber,
                          Set<Double> uniqueLables) {
    // finally, print the model to the model_file
    if (fileName != null) {
      try {
        BufferedWriter modelFile = new BufferedWriter(new FileWriter(fileName));
        modelFile.write(SVMParameters.CLASSIFICATION_TYPE + "#" + 0 + "\r\n");
        modelFile.write(SVMParameters.CLASS_NUMBER + "#" + classNumber + "\r\n");
        String labelstr = SVMParameters.LABELS + "#";
        for (Double l : uniqueLables) {
          labelstr += " " + l.intValue();
        }
        labelstr += "\r\n";
        modelFile.write(labelstr);
        modelFile.write(SVMParameters.DIMENSION + "#" + String.valueOf(this.d)
                        + "\r\n");
        modelFile.write(SVMParameters.A + "#" + String.valueOf(this.myA)
                        + "\r\n");
        modelFile.write(SVMParameters.SNORM + "#"
                        + String.valueOf(this.mySnorm) + "\r\n");
        StringBuffer temp = new StringBuffer();
        Iterator<Vector.Element> iter = this.myVector.iterateNonZero();
        while (iter.hasNext()) {
          Vector.Element elt = iter.next();
          temp.append(elt.index() + ":" + elt.get() + " ");
        }
        modelFile.write(SVMParameters.W + "#" + temp.toString() + "\r\n");
        modelFile.flush();
        modelFile.close();
      } catch (IOException e) {
        log.error("Write Model File Error:" + e.getMessage());
      }
    }
  }
  
  /**
   * Dump batch models to one file. Multi-classification use such file to dump
   * models.
   * 
   * @param fileName
   *          Name of file
   * @param weightList
   *          List of WeightVectors
   * @param para
   *          Parameters
   * @param uniqueLables
   *          A set of unique labels in this data set
   */
  public static void batchDumpModels(String fileName,
                                     List<WeightVector> weightList,
                                     SVMParameters para,
                                     Set<Double> uniqueLables) {
    if (fileName != null) {
      try {
        BufferedWriter modelFile = new BufferedWriter(new FileWriter(fileName));
        StringBuffer lines = new StringBuffer();
        lines.append(SVMParameters.CLASSIFICATION_TYPE + "#"
                     + para.getClassificationType() + "\r\n");
        lines.append(SVMParameters.CLASS_NUMBER + "#" + para.getClassNum() + "\r\n");
        String labels = SVMParameters.LABELS + "#";
        Integer[] labelList = new Integer[para.getClassNum()];
        int idx = 0;
        for (Double l : uniqueLables) {
          labels += " " + l.intValue();
          labelList[idx++] = l.intValue();
        }
        labels += "\r\n";
        lines.append(labels);
        lines.append(SVMParameters.DIMENSION + "#" + weightList.get(0).d
                     + "\r\n");
        
        lines.append(SVMParameters.SNORM + "#" + weightList.get(0).mySnorm
                     + "\r\n");
        
        int pos = 0;
        if (2 == para.getClassificationType()) { // one-vs.-one output
          for (int i = 0; i < para.getClassNum(); i++) {
            for (int j = i + 1; j < para.getClassNum(); j++) {
              lines.append(SVMParameters.A + labelList[i] + "_" + labelList[j]
                           + "#" + weightList.get(i).myA + "\r\n");
              StringBuffer temp = new StringBuffer();
              Iterator<Vector.Element> iter = weightList.get(pos).myVector
                  .iterateNonZero();
              while (iter.hasNext()) {
                Vector.Element elt = iter.next();
                temp.append(elt.index() + ":" + elt.get() + " ");
              }
              lines.append(SVMParameters.W + labelList[i] + "_" + labelList[j]
                           + "#" + temp.toString() + "\r\n");
              pos++;
            }
          }
        } else if (3 == para.getClassificationType()) { // one-vs.-other output
          for (int i = 0; i < weightList.size(); i++) {
            lines.append(SVMParameters.A + labelList[i] + "#"
                         + weightList.get(i).myA + "\r\n");
            StringBuffer temp = new StringBuffer();
            Iterator<Vector.Element> iter = weightList.get(i).myVector
                .iterateNonZero();
            while (iter.hasNext()) {
              Vector.Element elt = iter.next();
              temp.append(elt.index() + ":" + elt.get() + " ");
            }
            lines.append(SVMParameters.W + labelList[i] + "#" + temp.toString()
                         + "\r\n");
          }
        }
        modelFile.write(lines.toString());
        modelFile.flush();
        modelFile.close();
      } catch (IOException e) {
        System.err
            .println("Write Batch Models to File Error:" + e.getMessage());
      }
    }
  }
  
  /**
   * Constructing WeightVector list from one file.
   * 
   * @param fileName
   *          Model file
   * @param weightList
   *          Weight list for all models.
   * @throws IOException
   */
  public static void getBatchModels(String fileName,
                                    Map<String,WeightVector> weightList) throws IOException {
    BufferedReader modelFile = new BufferedReader(new FileReader(fileName));
    Map<String,String> keyValue = new OpenHashMap<String,String>();
    String line;
    String[] vectorString;
    int d;
    double mySnorm;
    Set<Double> labels = null;
    int classNum;
    List<String> weightKeyList = new ArrayList<String>();
    
    // read all line from file
    while ((line = modelFile.readLine()) != null) {
      String[] words = line.trim().split("#");
      if (words.length > 1) {
        keyValue.put(words[0], words[1]);
        if (words[0].contains("W")) {
          weightKeyList.add(words[0]);
        }
      }
    }
    
    // process the variables.
    classNum = Integer.parseInt(keyValue.get(SVMParameters.CLASS_NUMBER));
//    classifierType = Integer.parseInt(keyValue
//        .get(SVMParameters.CLASSIFICATION_TYPE));
    d = Integer.parseInt(keyValue.get(SVMParameters.DIMENSION));
    mySnorm = Double.parseDouble(keyValue.get(SVMParameters.SNORM));
    
//    // label
//    labelList = keyValue.get(SVMParameters.LABELS).trim().split(" ");
//    if (labelList.length > 1) {
//      labels = new TreeSet<Double>();
//      for (int i = 0; i < labelList.length; i++) {
//        labels.add(new Double(labelList[i]));
//      }
//    }

    // label
    if (weightKeyList.size() > 1) {
      labels = new TreeSet<Double>();
      for (int i = 0; i < weightKeyList.size(); i++) {
        String[] dim = weightKeyList.get(i).replace("W", "").split("_");
        if (dim.length > 1) {
          // for W0_1
          labels.add(new Double(dim[0]));
          labels.add(new Double(dim[1]));
        } else {
          labels.add(new Double(dim[0]));
        }
      }
    }
    
    // it is not depends on the one-vs.-one or one-vs.-others.
    for (int modelIndex = 0; modelIndex < weightKeyList.size(); modelIndex++) {
      WeightVector w = new WeightVector();
      w.classNum = classNum;
      w.d = d;
      w.mySnorm = mySnorm;
      w.labels = labels;
      
      // A
      w.myA = Double.parseDouble(keyValue.get(weightKeyList.get(modelIndex)
          .replace("W", "A")));
      
      // weight vector
      vectorString = keyValue.get(weightKeyList.get(modelIndex)).split(" ");
      w.myVector = new SequentialAccessSparseVector(d, 10);
      for (int i = 0; i < vectorString.length; i++) {
        String[] iv = vectorString[i].split(":");
        w.myVector.setQuick(Integer.parseInt(iv[0]), Double.parseDouble(iv[1]));
      }
      weightList.put(weightKeyList.get(modelIndex), w);
    }
    modelFile.close();
  }
  
  /**
   * Constructing WeightVector list from one file, which stored on HDFS.
   * 
   * @param hostName
   *          HDFS server address and port number
   * @param fileName
   *          Name of model file or folder
   * @param weightList
   *          Weight list for all models
   * @throws IOException
   */
  public static void getBatchModels(String hostName,
                                    String fileName,
                                    Map<String,WeightVector> weightList) throws IOException {
    HDFSReader rd = new HDFSReader();
    rd.setServerAddress(hostName);
    Path file = new Path(fileName);
    Map<String,String> keyValue = new OpenHashMap<String,String>();
    String line;
    String[] vectorString;
    int d;
    double mySnorm;
    Set<Double> labels = null;
    int classNum;
    List<String> weightKeyList = new ArrayList<String>();
    
    // read all line from file
    List<String> lines;
    if (rd.isDir(file)) {
      lines = rd.readADirectory(file);
    } else {
      lines = rd.readAllLines(file);
    }
    
    Iterator<String> elt = lines.iterator();
    while (elt.hasNext()) {
      line = elt.next();
      String[] words = line.trim().split("#");
      if (words.length > 1) {
        keyValue.put(words[0], words[1]);
        if (words[0].contains("W")) {
          weightKeyList.add(words[0]);
        }
      }
    }
    
    // process the variables.
    classNum = Integer.parseInt(keyValue.get(SVMParameters.CLASS_NUMBER));
    // classifierType =
    // Integer.parseInt(keyValue.get(SVMParameters.CLASSIFICATION_TYPE));
    d = Integer.parseInt(keyValue.get(SVMParameters.DIMENSION));
    mySnorm = Double.parseDouble(keyValue.get(SVMParameters.SNORM));
    
    // label
    if (weightKeyList.size() > 1) {
      labels = new TreeSet<Double>();
      for (int i = 0; i < weightKeyList.size(); i++) {
        String[] dim = weightKeyList.get(i).replace("W", "").split("_");
        if (dim.length > 1) {
          // for W0_1
          labels.add(new Double(dim[0]));
          labels.add(new Double(dim[1]));
        } else {
          labels.add(new Double(dim[0]));
        }
      }
    }
    
    // it is not depends on the one-vs.-one or one-vs.-others.
    for (int modelIndex = 0; modelIndex < weightKeyList.size(); modelIndex++) {
      WeightVector w = new WeightVector();
      w.classNum = classNum;
      w.d = d;
      w.mySnorm = mySnorm;
      w.labels = labels;
      
      // A
      w.myA = Double.parseDouble(keyValue.get(weightKeyList.get(modelIndex)
          .replace("W", "A")));
      
      // weight vector
      vectorString = keyValue.get(weightKeyList.get(modelIndex)).split(" ");
      w.myVector = new SequentialAccessSparseVector(d, 10);
      for (int i = 0; i < vectorString.length; i++) {
        String[] iv = vectorString[i].split(":");
        w.myVector.setQuick(Integer.parseInt(iv[0]), Double.parseDouble(iv[1]));
      }
      weightList.put(weightKeyList.get(modelIndex), w);
    }
    
  }
}
