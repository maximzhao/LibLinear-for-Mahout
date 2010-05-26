/*
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
package org.apache.mahout.classifier.svm.mapreduce;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * MapReduce Utilization for Hadoop .2.0.
 * 
 */
public class MapReduceUtil {
  
  public static final String NULL_OUTPUT_FILE_FORMAT_CLASS_NAME = 
                                "org.apache.hadoop.mapreduce.lib.output.NullOutputFormat";
  
  public MapReduceUtil() {

  }
  
  /**
   * for hadoop .1.9.
   * 
   * Gets a job driver.
   * 
   * @param jobClass
   * @param jobName
   * @param numReducers
   * @param inputPath
   * @param outputPath
   * @param inputFormat
   * @param outputFormat
   * @param mapper
   * @param combiner
   * @param reducer
   * @param outputkey
   * @param outputValue
   * @return
   * @throws java.io.IOException
   * @deprecated to be removed soon
   */
  public static Job getJobDriver(Class<?> jobClass,
                                 String jobName,
                                 int numReducers,
                                 String inputPath,
                                 String outputPath,
                                 Class<? extends InputFormat> inputFormat,
                                 Class<? extends OutputFormat> outputFormat,
                                 Class<? extends Mapper> mapper,
                                 Class<? extends Reducer> combiner,
                                 Class<? extends Reducer> reducer,
                                 Class<? extends WritableComparable> outputkey,
                                 Class<? extends Writable> outputValue) throws IOException {
    
    Job job = new Job(new Configuration());
    
    // job
    job.setJarByClass(jobClass);
    job.setJobName(jobName);
    
    // number reducers
    if (numReducers > -1) {
      job.setNumReduceTasks(numReducers);
    }
    
    // file path
    if (null != inputPath) {
      FileInputFormat.addInputPaths(job, inputPath);
    }
    if (null != outputPath) {
      FileOutputFormat.setOutputPath(job, new Path(outputPath));
    }
    
    // file format
    if (null != inputFormat) {
      job.setInputFormatClass(inputFormat);
    }
    if (null != outputFormat) {
      job.setOutputFormatClass(outputFormat);
    }
    
    // mapper, combiner, redcuer, partitioner
    if (null != mapper) {
      job.setMapperClass(mapper);
    }
    if (null != combiner) {
      job.setCombinerClass(combiner);
    }
    if (null != reducer) {
      job.setReducerClass(reducer);
    }
    
    // map, output key and value class
    if (null != outputkey) {
      job.setMapOutputKeyClass(outputkey);
    }
    if (null != outputValue) {
      job.setMapOutputValueClass(outputValue);
    }
    
    // output key and value class
    return job;
  }
  
  /**
   * Submit a job to hadoop cluster.
   * 
   * @param job
   * @return
   * @throws java.io.IOException
   * @throws java.lang.InterruptedException
   * @throws java.lang.ClassNotFoundException
   */
  public static boolean submitJob(Job job) throws IOException,
                                          InterruptedException,
                                          ClassNotFoundException {
    return job.waitForCompletion(true);
  }
  
  /**
   * Set the general parameters related to a job.
   * 
   * @param job
   * @param inputPath
   * @param outputPath
   * @param maxSplitSize
   * @param numReducers
   * @param maxHeapSize
   * @param taskTimeOut
   * @throws java.io.IOException
   * @throws java.lang.InterruptedException
   * @throws java.lang.ClassNotFoundException
   */
  public static void setJobParameters(Job job,
                                      String inputPath,
                                      String outputPath, // path
                                      long maxSplitSize,
                                      int numReducers, // mappers and reducers
                                      String maxHeapSize,
                                      long taskTimeOut //
  ) throws IOException, InterruptedException, ClassNotFoundException {
    // step 1 set job dynamic parameters
    MapReduceUtil.setJobDynamicParameters(job, inputPath, outputPath,
      maxSplitSize, numReducers);
    
    // step 2 set task dynamic parameters
    MapReduceUtil.setTaskDynamicParameters(job, taskTimeOut, maxHeapSize);
  }
  
  /**
   * Sets the static parameters (input, output, map, combine, reduce) related to
   * a job.
   * 
   * @param job
   * @param jobClass
   * @param jobName
   * @param inputFormat
   * @param outputFormat
   * @param mapper
   * @param combiner
   * @param reducer
   * @param mapOutputKey
   * @param mapOutputValue
   * @throws java.io.IOException
   */
  public static void setJobStaticParameters(Job job,
                                            Class<?> jobClass,
                                            String jobName,
                                            Class<? extends InputFormat> inputFormat,
                                            Class<? extends OutputFormat> outputFormat,
                                            Class<? extends Mapper> mapper,
                                            Class<? extends Reducer> combiner,
                                            Class<? extends Reducer> reducer,
                                            Class<? extends WritableComparable> mapOutputKey,
                                            Class<? extends Writable> mapOutputValue) throws IOException {
    
    // job (name and class)
    job.setJobName(jobName);
    job.setJarByClass(jobClass);
    
    // format (input and output)
    if (null != inputFormat) {
      job.setInputFormatClass(inputFormat);
    }
    if (null != outputFormat) {
      job.setOutputFormatClass(outputFormat);
    }
    
    // mapper
    if (null != mapper) {
      job.setMapperClass(mapper);
    }
    if (null != mapOutputKey) {
      job.setMapOutputKeyClass(mapOutputKey);
    }
    if (null != mapOutputValue) {
      job.setMapOutputValueClass(mapOutputValue);
    }
    
    // combiner
    if (null != combiner) {
      job.setCombinerClass(combiner);
    }
    
    // reducer
    if (null != reducer) {
      job.setReducerClass(reducer);
    }
    // job.setOutputKeyClass(jobClass);
    // partitioner
  }
  
  public static void setJobPartitioner(Job job,
                                       Class<? extends Partitioner> partitioner) {
    job.setPartitionerClass(partitioner);
  }
  
  /**
   * Sets the dynamic parameters related to a job.
   * 
   * <ol>
   * <li>The input paths can be a string composed of many paths which are
   * separated by comma (',').</li>
   * 
   * <li>The maximum split size must be positive, which determines the number of
   * mappers.</li>
   * <li>The number of reducers must be nonegative.</li>
   * </ol>
   * 
   * @param job
   * @param inputPaths
   * @param outputPath
   * @param maxSplitSize
   * @param numReducers
   * @throws java.io.IOException
   * @throws ClassNotFoundException
   */
  public static void setJobDynamicParameters(Job job,
                                             String inputPaths,
                                             String outputPath,
                                             long maxSplitSize,
                                             int numReducers) throws IOException,
                                                             ClassNotFoundException {
    // input path
    if (null != inputPaths) {
      FileInputFormat.addInputPaths(job, inputPaths);
    } else {
      throw new IOException("[hadoop][job] input path is not specified");
    }
    
    // output path
    if (null != outputPath) {
      FileOutputFormat.setOutputPath(job, new Path(outputPath));
    } else if (job.getOutputFormatClass().getName().equals(
      NULL_OUTPUT_FILE_FORMAT_CLASS_NAME))
    ;
    else {
      throw new IOException("[hadoop][job] output path is not specified");
    }
    
    // maximum split size which determines the number of mappers
    if (maxSplitSize > 0) {
      job.getConfiguration().set("mapred.min.split.size", "0");
      job.getConfiguration().set("mapred.max.split.size", maxSplitSize + "");
    } else {
      throw new IOException("[hadoop][job] maximum split size must be positive");
    }
    
    // the number of reducers
    if (numReducers >= 0) {
      job.setNumReduceTasks(numReducers);
    } else {
      throw new IOException(
          "[hadoop][job] number of reducers must be nonnegative");
    }
  }
  
  /**
   * Sets the dynamic parameters related to a task.
   * 
   * Note that the unit of taskTimeOut is minute.
   * 
   * @param job
   * @param taskTimeOut
   *          The unit is minute
   * @param maxHeapSize
   *          such as -Xmx700M
   * @throws java.io.IOException
   */
  public static void setTaskDynamicParameters(Job job,
                                              long taskTimeOut,
                                              String maxHeapSize) throws IOException {
    // time out of a task
    if (taskTimeOut > 0) {
      job.getConfiguration().set("mapred.task.timeout",
        (taskTimeOut * 60 * 1000) + ""); // unit is minutes
    } else {
      throw new IOException("[hadoop][job] taks timeout must be positive");
    }
    
    // You can specify other Java options for each map or reduce task here,
    // but most likely you will want to adjust the heap size.
    if (null != maxHeapSize) {
      job.getConfiguration().set("mapred.child.java.opts", maxHeapSize);
    }
  }
  
  /**
   * 
   * It is seldom used.
   * 
   * @param job
   * @param maxMapTasks
   * @param maxReduceTasks
   * @param maxMappersPerTaskTracker
   * @param maxReducersPerTaskTracker
   */
  public static void setTaskStaticParameters(Job job,
                                             long maxMapTasks,
                                             long maxReduceTasks,
                                             long maxMappersPerTaskTracker,
                                             long maxReducersPerTaskTracker) {
    // As a rule of thumb, use 10x the number of slaves (i.e., number of
    // tasktrackers).
    job.getConfiguration().setLong("mapred.map.tasks", maxMapTasks);
    
    // As a rule of thumb, use 2x the number of slave processors (i.e., number
    // of tasktrackers).
    job.getConfiguration().setLong("mapred.reduce.tasks", maxMapTasks);
    
    // The maximum number of map tasks that will be run simultaneously by a task
    // tracker.
    job.getConfiguration().setLong("mapred.tasktracker.map.tasks.maximum",
      maxMappersPerTaskTracker);
    
    // The maximum number of reduce tasks that will be run simultaneously by a
    // task tracker.
    job.getConfiguration().setLong("mapred.tasktracker.reduce.tasks.maximum",
      maxReducersPerTaskTracker);
  }
  
  /**
   * It may not copy all the things in a configuration. Be careful when using
   * it.
   * 
   * @param conf
   * @return
   */
  public static Configuration copyConfiguration(Configuration conf) {
    Configuration nconf = new Configuration();
    Iterator<Entry<String,String>> it = conf.iterator();
    Entry<String,String> e = null;
    while (it.hasNext()) {
      e = it.next();
      nconf.set(e.getKey(), e.getKey());
    }
    return nconf;
  }
  
  public static void checkParameters(Configuration oldConf,
                                     Class<? extends Mapper> innerMapper,
                                     Class<? extends Mapper> innerCombiner,
                                     Class<? extends Reducer> innerReducer) throws IOException,
                                                                           InterruptedException,
                                                                           NoSuchMethodException,
                                                                           IllegalAccessException,
                                                                           IllegalArgumentException,
                                                                           InvocationTargetException,
                                                                           ClassNotFoundException,
                                                                           InstantiationException {
    Configuration conf = copyConfiguration(oldConf); // get a copy of it
    
    // important, for reducer use
    conf.setStrings("io.serializations",
      "org.apache.hadoop.io.serializer.WritableSerialization");
    
  }
}
