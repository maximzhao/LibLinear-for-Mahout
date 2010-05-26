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
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.classifier.svm.mapreduce.MapReduceUtil;
import org.apache.mahout.classifier.svm.svmweightvector.WeightVector;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.Prediction;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.PredictionFactory;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 *
 */
public class ParallelMultiClassPredictionJob {
  
  private static final Logger log = LoggerFactory
      .getLogger(ParallelMultiClassPredictionJob.class);
  // public static void main(String[] args) throws IOException,
  // InterruptedException, Exception {
  // Job job = new Job(new Configuration());
  //
  // setJobParameters(job);
  // setMapperParameters(job.getConfiguration(), "/user/maximzhao/proteinova/",
  // "hdfs://localhost:12009", 3, 3);
  //
  //
  // // prepare the input
  // List<String> in = new ArrayList();
  // in.add("2 21:1.00 42:1.00 51:1.00
  // 74: .50 83: .50 86:1.00 122: .50 123: .50 146:1.00 166: .50
  // 167: .50 172:1.00 207: .50 208: .50 213: .50 222: .50 233:1.00 255: .50
  // 271: .50 282: .50 293: .50 313:1.00 317:1.00 344: .50 352: .50 ");
  // MapReduceTestUtil.TestJob(job, in, Text.class, Text.class);
  //
  // }
  /** Job name */
  public static final String JOB_NAME = "job.mahout.classifier.svm.multi-classfier";
  
  /**
   * The number of reducers must be 0 since it only has mappers.
   * 
   * @param job
   * @throws java.io.IOException
   */
  public static void setJobParameters(Job job) throws IOException {
    MapReduceUtil.setJobStaticParameters(job, ParallelMultiClassPredictionJob.class,
      JOB_NAME, TextInputFormat.class, TextOutputFormat.class,
      ParallelMultiClassPredictionJob.InnerMapper.class, null, null,
      Text.class, Text.class);
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
   * @param modelPath
   *          model name
   * @param hdfsServer
   * @param classNum
   * @param classifierType
   */
  public static void setMapperParameters(Configuration conf,
                                         String modelPath,
                                         String hdfsServer,
                                         Integer classNum,
                                         Integer classifierType) {
    // set the columns to be updated
    if (null != modelPath) {
      conf.set(SVMParameters.HADOOP_MODLE_PATH, modelPath);
    }
    
    if (null != hdfsServer) {
      conf.set(SVMParameters.HDFS_SERVER, hdfsServer);
    }
    
    if (null != classNum) {
      conf.setInt(SVMParameters.HADOOP_CLASS_NUMBER, classNum);
    }
    
    if (null != classifierType) {
      conf.setInt(SVMParameters.CLASSIFICATION_TYPE, classNum);
    }
  }
  
  /**
   * Generates track info from track list.
   * 
   */
  public static final class InnerMapper extends
    Mapper<LongWritable,Text,Text,Text> {
    
    Text outKey = new Text();
    Text outValue = new Text();
    SequentialAccessSparseVector row;
    SVMParameters para = new SVMParameters();
    Map<String,WeightVector> w = new HashMap<String,WeightVector>();
    String result;
    String defaultModelName = "/user";
    
    @Override
    public void setup(Context context) throws IOException {
      log.info("[mapper]: setup");
      
      para.setModelFileName(context.getConfiguration().get(
        SVMParameters.HADOOP_MODLE_PATH, this.defaultModelName));
      para.setHdfsServerAddr(context.getConfiguration().get(
        SVMParameters.HDFS_SERVER, "hdfs://localhost:12009"));
      para.setClassNum(context.getConfiguration().getInt(
        SVMParameters.HADOOP_CLASS_NUMBER, 3));
      para.setClassificationType(context.getConfiguration().getInt(
        SVMParameters.CLASSIFICATION_TYPE, 3));
      
      // read all weight vectors.
      WeightVector.getBatchModels(para.getHdfsServerAddr(), para.getModelFileName(),
        this.w);
    }
    
    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException,
                                                                  InterruptedException {
      log.info("Multi-classification!");
      
      Prediction tester = PredictionFactory.getInstance(para.getClassificationType());
      result = tester.oneLineClassifier(w, value);
      
      context.getCounter("map", "Total.successful.tested.line").increment(1);
      outValue.set(String.valueOf(result));
      outKey.set(key.toString());

      context.write(outKey, outValue);
    }
  }
  // /**
  // * Reducer. key should not be empty, otherwise, nothing will be written.
  // * value can be empty.
  // */
  // public static class InnerCombiner extends Reducer<Text, Text, Text, Text> {
  //
  // private Text outValue = new Text();
  // double curLoss = 0.0;
  // double testLoss = 0.0;
  // double testError = 0.0;
  // int lineCounter = 0;
  //
  // /**
  // *
  // * @param key should be "prediction"
  // * @param values the
  // * @param context
  // * @throws IOException
  // * @throws InterruptedException
  // */
  // @Override
  // public void reduce(Text key, Iterable<Text> values, Context context) throws
  // IOException, InterruptedException {
  // this.testError = 0;
  // for (Text value : values) {
  // this.testError += Double.parseDouble(value.toString());
  // lineCounter++;
  // }
  // if (0 != lineCounter) {
  // this.testError /= lineCounter;
  // }
  // outValue.set(String.valueOf(this.testError));
  // context.write(key, outValue); // use value as key
  // context.getCounter("combiner", "total_line").increment(1);
  // }
  // }
  // /**
  // * Reducer. key should not be empty, otherwise, nothing will be written.
  // * value can be empty.
  // */
  // public static final class InnerReducer extends Reducer<Text, Text, Text,
  // Text> {
  //
  // private Text outValue = new Text();
  // double curLoss = 0.0;
  // double testLoss = 0.0;
  // double testError = 0.0;
  // int lineCounter = 0;
  // String[] lossError = null;
  //
  // /**
  // *
  // * @param key should be "prediction"
  // * @param values the
  // * @param context
  // * @throws IOException
  // * @throws InterruptedException
  // */
  // @Override
  // public void reduce(Text key, Iterable<Text> values, Context context) throws
  // IOException, InterruptedException {
  // this.testError = 0;
  // for (Text value : values) {
  // this.testError += Double.parseDouble(value.toString());
  // lineCounter++;
  // }
  // if (0 != lineCounter) {
  // this.testError /= lineCounter;
  // }
  // outValue.set(String.valueOf(1 - this.testError));
  // context.write(key, outValue); // use value as key
  // context.getCounter("reduce", "total_line").increment(1);
  // }
  // }
}
