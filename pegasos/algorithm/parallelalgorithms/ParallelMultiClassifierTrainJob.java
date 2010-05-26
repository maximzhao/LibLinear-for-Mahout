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
import java.util.Random;
import java.util.regex.Pattern;

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
import org.apache.mahout.classifier.svm.algorithm.metafunctions.Training;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.TrainingFactory;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class ParallelMultiClassifierTrainJob {
  
  /** Job name */
  public static final String JOB_NAME = "job.mahout.classifier.svm.parallel.multiple.classification";
  private static final Logger log = LoggerFactory
      .getLogger(ParallelMultiClassifierTrainJob.class);
  
  // public static void main(String[] args) throws IOException,
  // InterruptedException, Exception {
  //
  // int sampleNum = 1000000;
  // int categories = 10;
  // Job job = new Job(new Configuration());
  // setJobParameters(job);
  // setMapperParameters(job.getConfiguration(), 100000, sampleNum, categories,
  // 3, 0);
  // setReducerParameters(job.getConfiguration(), 0.001, 1,
  // "/user/maximzhao/multiclassification", "hdfs://localhost:12009" );
  //
  // // prepare the input
  // List<String> in = new ArrayList();
  // for (int i = 0; i < sampleNum; i++) {
  // in.add(String.valueOf(i % categories) + " 21:1.00 42:1.00
  // 63:1.00 84:1.00 105:1.00 126:1.00 147:1.00 168:1.00 177:1.00 200: .50
  // 209: .50 212:1.00 248: .50 249: .50 272:1.00 292: .50 293: .50
  // 298:1.00 333: .50 334: .50 339: .50 348: .50 ");
  // }
  //
  // MapReduceTestUtil.TestJob(job, in, Text.class,Text.class);
  //
  // }
  
  /**
   * The number of reducers must be 0 since it only has mappers.
   * 
   * @param job
   * @throws java.io.IOException
   */
  public static void setJobParameters(Job job) throws IOException {
    MapReduceUtil.setJobStaticParameters(job,
      ParallelMultiClassifierTrainJob.class, JOB_NAME,
      TextInputFormat.class, TextOutputFormat.class,
      ParallelMultiClassifierTrainJob.InnerMapper.class,
      // ParallelMultiClassifierTrainJobber.InnerCombiner.class,
      null, ParallelMultiClassifierTrainJob.InnerReducer.class, Text.class,
      Text.class);
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
   * @param maxIteration
   * @param samplesSize
   *          the size of training samples.
   * @param classNum
   *          the number of classes
   * @param classificationType
   * @param startingClassIndex
   */
  public static void setMapperParameters(Configuration conf,
                                         int maxIteration,
                                         long samplesSize,
                                         int classNum,
                                         int classificationType,
                                         int startingClassIndex) {
    // set the columns to be updated
    conf.setInt(SVMParameters.HADOOP_MAX_ITERATION, maxIteration);
    conf.setLong(SVMParameters.HADOOP_SAMPLE_NUMBER, samplesSize);
    conf.setInt(SVMParameters.HADOOP_CLASS_NUMBER, classNum);
    conf.setInt(SVMParameters.HADOOP_MULTI_CLASS_TYPE, classificationType);
    conf.setInt(SVMParameters.HADOOP_STARTING_CLASS_INDEX, startingClassIndex);
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
   * @param lambda
   * @param k
   * @param modelFile
   *          model files store path
   * @param hdfsServer
   *          hdfs server address
   */
  public static void setReducerParameters(Configuration conf,
                                          double lambda,
                                          int k,
                                          String modelFile,
                                          String hdfsServer) {
    // set the columns to be updated
    conf.setFloat(SVMParameters.HADOOP_LAMBDA, (float) lambda);
    conf.setInt(SVMParameters.HADOOP_K, k);
    conf.set(SVMParameters.HADOOP_MODLE_PATH, modelFile);
    conf.set(SVMParameters.HDFS_SERVER, hdfsServer);
  }
  
  /**
   * Generates track info from track list.
   * 
   */
  public static final class InnerMapper extends
      Mapper<LongWritable,Text,Text,Text> {
    
//    Text outKey = new Text();
//    Text outValue = new Text();
    String temp;
    String[] array;
    int label;
    SequentialAccessSparseVector row;
    Random rand = new Random();
    // scale the number of samples of each class for random emitting.
    float scaler = (float) 1.1;
    float sameLabelProbabilityRange;
    float otherLabelProbabilityRange;
    String outputString;
    float randValue;
    int classNUM;
    float lowboundary;
    int maxIteration;
    long samplesSize;
    int mapNum;
    int multiClassType;
    int startingClassIndex;
    String[] output;
    Pattern splitter = Pattern.compile("[ ]+");
    
    @Override
    public void setup(Context context) throws IOException {
      log.info("[mapper]: setup");
      this.maxIteration = context.getConfiguration().getInt(
        SVMParameters.HADOOP_MAX_ITERATION, 100000);
      this.samplesSize = context.getConfiguration().getLong(
        SVMParameters.HADOOP_SAMPLE_NUMBER, 100000);
      this.classNUM = context.getConfiguration().getInt(
        SVMParameters.HADOOP_CLASS_NUMBER, 10);
      this.multiClassType = context.getConfiguration().getInt(
        SVMParameters.HADOOP_MULTI_CLASS_TYPE, 0);
      this.startingClassIndex = context.getConfiguration().getInt(
        SVMParameters.HADOOP_STARTING_CLASS_INDEX, 0);
      
      this.sameLabelProbabilityRange = (float) this.scaler * this.classNUM
                                       * this.maxIteration
                                       / (2 * this.samplesSize);
      this.otherLabelProbabilityRange = this.sameLabelProbabilityRange
                                        / (this.classNUM - 1);
      
    }
    
    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException,
                                                                  InterruptedException {
      
      // check whether this line is sampled
      if (0 == value.toString().indexOf("#") || value.getLength() < 1) {
        return;
      } else if (2 == value.toString().split("#").length) {
        temp = value.toString().split("#")[0];
      } else {
        temp = value.toString();
      }
      
      this.row = new SequentialAccessSparseVector(Integer.MAX_VALUE, 10);
      
      temp.trim();
      array = splitter.split(temp);
      label = (int) Double.parseDouble(array[0].replace("+", ""));
      
      this.lowboundary = 0;
      this.outputString = null;

      
      if (3 == this.multiClassType) { // one-versus-others
        /** use random sampling techniques. */
        if (this.sameLabelProbabilityRange < 1) { // < 1
          // emit labels according to the random range.
          for (int i = startingClassIndex; i < this.classNUM
                                               + startingClassIndex; i++) {
            randValue = this.rand.nextFloat();
            // for current lab pick it as correct rang of pobability.
            if (label == i) {
              if (randValue < this.sameLabelProbabilityRange) {
                outputString += i + "@ +1" + temp.replaceFirst(array[0], "")
                                + "_"; // label +1
              }
            } else if (randValue < this.otherLabelProbabilityRange) {
              outputString += i + "@ -1" + temp.replaceFirst(array[0], "")
                              + "_"; // label -1 for other categories
            }
          }
        } else { // emit all samples.
          for (int i = startingClassIndex; i < this.classNUM
                                               + startingClassIndex; i++) {
            if (label == i) {
              outputString += i + "@ +1" + temp.replaceFirst(array[0], "")
                              + "_";
            } else {
              outputString += i + "@ -1" + temp.replaceFirst(array[0], "")
                              + "_";
            }
          }
        }
        
        if (null == outputString) {
          return;
        }

        
        output = outputString.replace("null", "").split("_");
        for (int i = 0; i < output.length; i++) {
          Text outKey = new Text();
          Text outValue = new Text();
          outValue.set(output[i].split("@")[1]);
          outKey.set(output[i].split("@")[0]);
          context.write(outKey, outValue);
          context.getCounter("map", "Total.emitted.lines").increment(1);
        }
      } else if (2 == this.multiClassType) {
        // one-against-one emit
        for (int i = startingClassIndex; i < label; i++) {
          outputString = i + "_" + label + "@-1"
                         + temp.replaceFirst(array[0], "");
          
          if (null == outputString) {
            context.getCounter("map", "Emit.null.samples.string").increment(1);
            return;
          }
          
          context.getCounter("map", "Total.emitted.lines").increment(1);
          Text outKey = new Text();
          Text outValue = new Text();
          outValue.set(outputString.split("@")[1]);
          outKey.set(outputString.split("@")[0]);
          context.write(outKey, outValue);
        }
        
        for (int i = label + 1; i < this.classNUM + startingClassIndex; i++) {
          outputString = label + "_" + i + "@1"
                         + temp.replaceFirst(array[0], "");
          
          if (null == outputString) {
            context.getCounter("map", "Emit.null.samples.string").increment(1);
            return;
          }
          
          context.getCounter("map", "Total.emitted.lines").increment(1);
          Text outKey = new Text();
          Text outValue = new Text();
          outValue.set(outputString.split("@")[1]);
          outKey.set(outputString.split("@")[0]);
          context.write(outKey, outValue);
        }
      } else {
        log
            .error("Exception: Only support multi-classification. Use -s 2 (one-vs.-one) or -s 3 (one-vs.-others)");
      }
      
    }
  }
  
  /**
   * Reducer. key should not be empty, otherwise, nothing will be written. value
   * can be empty.
   */
  public static final class InnerReducer extends Reducer<Text,Text,Text,Text> {
    
    private Text outValue = new Text();
    SVMParameters para;
    Training classifier;
    
    @Override
    public void setup(Context context) throws IOException {
      log.info("[reducer]: setup");
      para = new SVMParameters();
      para.setLambda(context.getConfiguration().getFloat(
        SVMParameters.HADOOP_LAMBDA, (float) 0.001));
      para.setMaxIter(context.getConfiguration().getInt(
        SVMParameters.HADOOP_MAX_ITERATION, (int) (100 / para.getLambda())));
      para.setExamplesPerIter(context.getConfiguration().getInt(
        SVMParameters.HADOOP_K, 1));
      para.setModelFileName(context.getConfiguration().get(
        SVMParameters.HADOOP_MODLE_PATH, "/user/maximzhao/multiclassification"));
      para.setHdfsServerAddr(context.getConfiguration().get(
        SVMParameters.HDFS_SERVER, "hdfs://localhost:12009"));
    }
    
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
    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
                                                                        InterruptedException {
      context.getCounter("reduce", "Total classes need to be processed!")
          .increment(1);
      DataSetHandler trainDataset = new DataSetHandler("noNeedFileName");
      if (!trainDataset.getData(values)) {
        throw new IOException();
      }
      
      // Train a binary classifier for each pair.
      classifier = TrainingFactory.getInstance(0);
      WeightVector weight = classifier.training(trainDataset, para);
      
      context.getCounter("reduce", "Total processed number of classes !")
          .increment(1);
      
      // output
      outValue.set(weight.dumpToString(para.getClassificationType(), para.getClassNum(),
        key.toString()));
      context.write(key, outValue);
    }
  }
}
