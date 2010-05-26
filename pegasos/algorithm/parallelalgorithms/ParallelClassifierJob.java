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
package org.apache.mahout.classifier.svm.algorithm.parallelalgorithms;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.classifier.svm.mapreduce.MapReduceUtil;
import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class ParallelClassifierJob {

  private static final Logger log = LoggerFactory.getLogger(ParallelClassifierJob.class);
  /** Job name */
  public static final String JOB_NAME = "job.mahout.classifier.svm.paralleltesting";
  
  /**
   * The number of reducers must be 0 since it only has mappers.
   * 
   * @param job
   * @throws java.io.IOException
   */
  public static void setJobParameters(Job job) throws IOException {
    MapReduceUtil.setJobStaticParameters(job, ParallelClassifierJob.class,
      JOB_NAME, TextInputFormat.class, TextOutputFormat.class,
      ParallelClassifierJob.InnerMapper.class,
      ParallelClassifierJob.InnerCombiner.class,
      ParallelClassifierJob.InnerReducer.class, Text.class, Text.class);
  }
  
  /**
   * Sets the parameters related to this mapper.
   * 
   * <p>
   * <ol>
   * <li></li>
   * </ol>
   * 
   * @param conf
   * @param hdfsServer
   * @param modelName
   *          model name
   */
  public static void setMapperParameters(Configuration conf,
                                         String hdfsServer,
                                         String modelName) {
    // set the columns to be updated
    if (null != modelName) {
      conf.set(SVMParameters.HADOOP_MODLE_PATH, modelName);
    }
    
    conf.set(SVMParameters.HDFS_SERVER, hdfsServer);
    
  }
  
  /**
   * Generates track info from track list.
   * 
   */
  public static final class InnerMapper extends
      Mapper<LongWritable,Text,Text,Text> {
    
    Text outKey = new Text();
    Text outValue = new Text();
    String str;
    int rowNum;
    String temp;
    String[] array;
    int key;
    double value;
    double curLoss;
    int label;
    SequentialAccessSparseVector row;
    WeightVector w;
    double testLoss;
    double testError;
    String modelFile;
    String defaultModelName = "SVM.model";
    String hdfsServer;
    
    @Override
    public void setup(Context context) throws IOException {
      log.info("[mapper]: setup");
      modelFile = context.getConfiguration().get(
        SVMParameters.HADOOP_MODLE_PATH, this.defaultModelName);
      log.info("[para]: para.mapper.solr.server.url = "
                         + this.modelFile);
      hdfsServer = context.getConfiguration().get(SVMParameters.HDFS_SERVER,
        SVMParameters.DEFAULT_HDFS_SERVER);
      //
      this.w = new WeightVector(hdfsServer, modelFile);
    }
    
    @Override
    public void map(LongWritable mapKey, Text mapValue, Context context) throws IOException,
                                                                  InterruptedException {
      // log.info("Sequential Testing (one point denotes 1000
      // samples!");
      
      // check whether this line is sampled
      if (0 == mapValue.toString().indexOf("#") || mapValue.getLength() < 1) {
        return;
      } else if (2 == mapValue.toString().split("#").length) {
        temp = mapValue.toString().split("#")[0];
      } else {
        temp = mapValue.toString();
      }
      
      this.row = new SequentialAccessSparseVector(Integer.MAX_VALUE, 10);
      
      temp.trim();
      array = temp.split(" ");
      label = Integer.parseInt(array[0].replace("+", ""));
      
      for (int j = 1; j < array.length; j++) {
        this.key = Integer.parseInt(array[j].split(":")[0]);
        this.value = Double.parseDouble(array[j].split(":")[1]);
        this.row.setQuick(this.key, this.value);
      }
      this.curLoss = 1 - label * w.times(row);
      
      context.getCounter("map", "Total.successful.tested.line").increment(1);
      
      outValue.set(String.valueOf(this.curLoss));
      outKey.set("testing");
      context.write(outKey, outValue);
    }
  }
  
  /**
   * Reducer. key should not be empty, otherwise, nothing will be written. value
   * can be empty.
   */
  public static class InnerCombiner extends Reducer<Text,Text,Text,Text> {
    
    private Text outValue = new Text();
    double curLoss;
    double testLoss;
    double testError;
    int lineCounter;
    
    /**
     * 
     * @param key
     *          should be "testing"
     * @param values
     *          the
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
                                                                           InterruptedException {
      for (Text value : values) {
        curLoss = Double.parseDouble(value.toString());
        if (curLoss < 0.0) {
          curLoss = 0.0;
        }
        
        this.testLoss += curLoss;
        
        if (curLoss >= 1.0) {
          this.testError += 1.0;
        }
        lineCounter++;
      }
      if (0 != lineCounter) {
        this.testError /= lineCounter;
        this.testLoss /= lineCounter;
      }
      outValue.set(this.testLoss + " " + this.testError);
      context.write(key, outValue); // use value as key
      context.getCounter("combiner", "total_line").increment(1);
    }
  }
  
  /**
   * Reducer. key should not be empty, otherwise, nothing will be written. value
   * can be empty.
   */
  public static final class InnerReducer extends Reducer<Text,Text,Text,Text> {
    
    private Text outValue = new Text();
    double curLoss;
    double testLoss;
    double testError;
    int lineCounter;
    String[] lossError;
    
    /**
     * 
     * @param key
     *          should be "testing"
     * @param values
     *          the
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
                                                                           InterruptedException {
      for (Text value : values) {
        lossError = value.toString().split(" ");
        if (2 == lossError.length) {
          this.testLoss += Double.parseDouble(lossError[0]);
          this.testError += Double.parseDouble(lossError[1]);
        }
        lineCounter++;
      }
      if (0 != lineCounter) {
        this.testLoss /= lineCounter;
        this.testError /= lineCounter;
      }
      outValue.set(this.testLoss + " " + this.testError);
      context.write(key, outValue); // use value as key
      context.getCounter("reduce", "total_line").increment(1);
    }
  }
}
