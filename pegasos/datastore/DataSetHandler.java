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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Due to the Pegasos solver randomly fetches Maximum Iteration samples. One
 * optimized method is that pre-fetch needed samples to Memory.
 * 
 * All the strategies have been determined in this class regarding the
 * parameters. Whether use file batch fetching or not :)
 * 
 * TODO when to use batch fetching? A intuitive idea is when size of train
 * samples > max iteration, typically 100, 000 samples
 * 
 * 
 */
public class DataSetHandler {
  
  private static final Logger log = LoggerFactory
      .getLogger(DataSetHandler.class);
  private Matrix dataset;
  private Map<Integer,Double> labels = new HashMap<Integer,Double>();
  // for counting the unique labels the data set has.
  private Set<Double> uniqueLables = new TreeSet<Double>();
  private String fileName;
  
  public DataSetHandler(String fileName) {
    this.fileName = fileName;
  }
  
  public String getFileName() {
    return this.fileName;
  }
  
 
  /**
   * This is the interface for getting data from general file system, HDFS etc.
   * 
   * All the strategies have been determined here regarding the parameters.
   * Determine whether use file batch fetching or not :)
   * 
   * TODO when to use batch fetching? A intuitive idea is when size of train
   * samples > max iteration
   * 
   * @param para
   * @throws IOException
   */
  public void getData(SVMParameters para) throws IOException {
    // For small data set
    if (para.getMaxIter() > para.getTrainSampleNumber()) {
      
      // hdfs file or normal file system.
      if (para.getHdfsServerAddr() != null) {
        getDataFromHDFS(para.getHdfsServerAddr());
      } else {
        getDataFromFile();
      }
      
    } else { // For large data set.
      // Produce all random number and store them as a pool.
      para.setRandomPool(new ArrayList<Integer>(para.getMaxIter()
                                               * para.getExamplesPerIter()));
      para.setRandomPoolsorted(new ArrayList<Integer>(para.getMaxIter()
                                                     * para.getExamplesPerIter()));
      
      Random rand = new Random();
      for (int i = 0; i < Math.abs(para.getClassNum() - 1) * para.getMaxIter()
                          * para.getExamplesPerIter(); i++) {
        int seed = rand.nextInt(para.getTrainSampleNumber());
        para.getRandomPool().add(seed);
        para.getRandomPoolsorted().add(seed);
      }
      
      if (para.getHdfsServerAddr() != null) {
        getBatchDataFromHDFS(para.getHdfsServerAddr(), para.getRandomPoolsorted());
      } else {
        getBatchDataFromFile(para.getRandomPoolsorted());
      }
    }
    
    // Set the class number.
    if (0 == para.getClassNum()) {
      para.setClassNum(this.getUniqueLables().size());
    }
  }
  
  /**
   * Converter: Text data from Reducer -> SparseMatrix
   * 
   * @param values
   * @return
   */
  public Boolean getData(Iterable<Text> values) {
    
    int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
    this.setDataset(new SparseMatrix(cardinality));
    String str;
    int rowNum = 0;
    Vector row;
    for (Text value : values) {
      str = value.toString();
      if (0 == str.indexOf("#")) {
        continue;
      }
      // convert string line to feature vector,
      row = str2Vector(str, rowNum);
      this.getDataset().assignRow(rowNum++, row);
    }
    return true;
  }
  
  /**
   * Get data set from file, it provides file (SVM-light) -> SparseMatrix
   * service. Thanks for Jake Mannix's suggestion, I set the cardinality to
   * Integer.MAX_VALUE without setting size to so huge number.
   * 
   * @return get data successfully or not
   * @throws IOException
   */
  public Boolean getDataFromFile() throws IOException {
    BufferedReader in = null;
    try {
      // set the sparse vector to integer.max_value, but the size is not so.
      // thanks for Jake Mannix's suggestion
      int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
      this.setDataset(new SparseMatrix(cardinality));
      in = new BufferedReader(new FileReader(this.fileName));
      String str;
      int rowNum = 0;
      Vector row;
      while ((str = in.readLine()) != null) {
        if (0 == str.indexOf("#")) {
          continue;
        }
        // convert string line to feature vector,
        row = str2Vector(str, rowNum);
        this.getDataset().assignRow(rowNum++, row);
      }
    } catch (Exception e) {
      log.error("Read File Error: " + e.getMessage());
      return false;
    } finally {
      in.close();
    }
    return true;
  }
  
  /**
   * Fetch a block of samples from file, only suitable for the case that the
   * samples in data set > max iteration
   * 
   * @param randomPool
   *          pre-fetched random line index.
   * @return
   * @throws IOException
   */
  public Boolean getBatchDataFromFile(List<Integer> randomPool) throws IOException {
    BufferedReader in = null;
    try {
      
      int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
      this.setDataset(new SparseMatrix(cardinality));
      in = new BufferedReader(new FileReader(this.fileName));
      String str;
      int rowNum = 0;
      Collections.sort(randomPool);
      Vector row;
      log.info("Using batch fetching!");
      
      while ((str = in.readLine()) != null) {
        // check whether this line is sampled
        if (0 == str.indexOf("#")) {
          continue;
        }
        if (randomPool.contains(rowNum)) {
          row = str2Vector(str, rowNum);
          this.getDataset().assignRow(rowNum, row);
        } else if (rowNum >= randomPool.get(randomPool.size() - 1)) {
          break;
        }
        rowNum++;
        if (0 == rowNum % 10000) {
          log.info("Loaded data lines: " + rowNum);
        }
      }
    } catch (Exception e) {
      log.error("Read File Error: " + e.getMessage());
      return false;
    } finally {
      in.close();
    }
    return true;
  }
  
  /**
   * Fetch samples from file (HDFS).
   * 
   * @param serverAddress
   *          HDFS server's address
   * @return
   * @throws IOException
   */
  public Boolean getDataFromHDFS(String serverAddress) throws IOException {
    
    Configuration conf = new Configuration();
    HDFSConfig.setSeverAddress(conf, serverAddress);
    Path file = new Path(this.fileName);
    FileSystem fs = FileSystem.get(conf);
    // LineReader lr = new LineReader(fs.open(filePath), 64000000);
    LineReader lr = new LineReader(fs.open(file));
    Text line = new Text();
    log.info("Data from HDFS:)");
    
    try {
      int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
      this.setDataset(new SparseMatrix(cardinality));
      String str = null;
      int rowNum = 0;
      Vector row = null;
      while (true) {
        lr.readLine(line);
        if (line.getLength() < 1) {
          break;
        }
        str = line.toString();
        if (0 == str.indexOf("#")) {
          continue;
        }
        row = str2Vector(str, rowNum);
        this.getDataset().assignRow(rowNum++, row);
        
        if (0 == rowNum % 10000) {
          log.info("Loaded data lines: " + rowNum);
        }
      }
    } catch (Exception e) {
      log.error("Read HDFS File Error: " + e.getMessage());
      return false;
    } finally {
      lr.close();
    }
    return true;
  }
  
  /**
   * Fetch a block of samples from file (HDFS) , only suitable for the case that
   * the samples in data set > max iteration
   * 
   * @param serverAddress
   *          hdfs server's address
   * @param randomPool
   *          pre-fetched random line index.
   * @return
   * @throws IOException
   */
  public Boolean getBatchDataFromHDFS(String serverAddress,
                                      List<Integer> randomPool) throws IOException {
    Configuration conf = new Configuration();
    HDFSConfig.setSeverAddress(conf, serverAddress);
    Path file = new Path(this.fileName);
    // set buff to 64MB
    // LineReader lr = new LineReader(fs.open(filePath), 64000000);
    LineReader lr = null;
    
    try {
      int[] cardinality = {Integer.MAX_VALUE, Integer.MAX_VALUE};
      this.setDataset(new SparseMatrix(cardinality));
      FileSystem fs = FileSystem.get(conf);
      lr = new LineReader(fs.open(file));
      Text line = new Text();
      String str;
      int rowNum = 0;
      Vector row;
      
      Collections.sort(randomPool);
      while (true) {
        lr.readLine(line);
        if (line.getLength() < 1) {
          break;
        }
        str = line.toString();
        if (0 == str.indexOf("#")) {
          continue;
        }
        if (randomPool.contains(rowNum)) {
          row = str2Vector(str, rowNum);
          this.getDataset().assignRow(rowNum, row);
        } else if (rowNum >= randomPool.get(randomPool.size() - 1)) {
          break;
        }
        rowNum++;
      }
    } catch (Exception e) {
      log.error("Read HDFS File Error: " + e.getMessage());
      return false;
    } finally {
      lr.close();
    }
    return true;
  }
  
  /**
   * String line to vector for svm-light or libsvm format data set
   * 
   * @param strLine
   *          a line of data in string.
   * @param lineNum
   *          line number.
   * 
   * @return Vector sparse vector of one row of features.
   * 
   */
  public Vector str2Vector(String strLine, int lineNum) {
    double label = 0.0;
    Vector row = new RandomAccessSparseVector(Integer.MAX_VALUE, 10);
    try {
      label = LibsvmFormatParser.str2Vector(strLine, row);
    } catch (NullInputString ex) {
      log.error("NULLInput: " + ex.getMessage());
    }
    this.getLabels().put(lineNum, label);
    this.getUniqueLables().add(label);
    return row;
  }

  public void setDataset(Matrix dataset) {
    this.dataset = dataset;
  }

  public Matrix getDataset() {
    return dataset;
  }

  public void setLabels(Map<Integer,Double> labels) {
    this.labels = labels;
  }

  public Map<Integer,Double> getLabels() {
    return labels;
  }

  public void setUniqueLables(Set<Double> uniqueLables) {
    this.uniqueLables = uniqueLables;
  }

  public Set<Double> getUniqueLables() {
    return uniqueLables;
  }
}
