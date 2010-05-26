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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.mahout.classifier.svm.mapreduce.MapReduceUtil;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class ParallelMultiClassifierTrainDriver {
  
  /** Job name */
  public static final String JOB_NAME = ParallelMultiClassifierTrainJob.JOB_NAME;
  private static final Logger log = LoggerFactory
      .getLogger(ParallelMultiClassifierTrainDriver.class);
  
  public static void main(String[] args) throws IOException,
                                        InterruptedException,
                                        ClassNotFoundException,
                                        OptionException {
    // args = new String [] {"-if","infile","-of","outfile","m",
    // "-nm","10","--nr","11"};
   log.info("[job] " + JOB_NAME);
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option trainFileOpt = obuilder
        .withLongName("trainFile")
        .withRequired(true)
        .withArgument(
          abuilder.withName("trainFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Training data set file").withShortName("if").create();
    
    Option outputFileOpt = obuilder.withLongName("output").withRequired(true)
        .withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("Out put file name: ").withShortName("of").create();
    
    Option lambdaOpt = obuilder.withLongName("lambda").withRequired(false)
        .withArgument(
          abuilder.withName("lambda").withMinimum(1).withMaximum(1).create())
        .withDescription("Regularization parameter (default = 0.01) ")
        .withShortName("l").create();
    
    Option iterOpt = obuilder.withLongName("iter").withRequired(false)
        .withArgument(
          abuilder.withName("iter").withMinimum(1).withMaximum(1).create())
        .withDescription("Number of iterations (default = 10/lambda) ")
        .withShortName("i").create();
    
    Option kOpt = obuilder.withLongName("k").withRequired(false).withArgument(
      abuilder.withName("k").withMinimum(1).withMaximum(1).create())
        .withDescription("Size of block for stochastic gradient (default = 1)")
        .withShortName("v").create();
    
    Option sampleNumOpt = obuilder
        .withLongName("trainSampleNum")
        .withRequired(false)
        .withArgument(
          abuilder.withName("trainSampleNum").withMinimum(1).withMaximum(1)
              .create())
        .withDescription(
          "Number of Samples in traindata set, for large-scale dataset optimization (default = 0) ")
        .withShortName("tsn").create();
    
    Option classNumOpt = obuilder.withLongName("classNum").withRequired(true)
        .withArgument(
          abuilder.withName("classNum").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The number of classes (Categories in multi-classification) ")
        .withShortName("c").create();
    
    Option startingClassIndexOpt = obuilder.withLongName("startingClassIndex")
        .withRequired(false).withArgument(
          abuilder.withName("startingClassIndex").withMinimum(1).withMaximum(1)
              .create()).withDescription(
          "The starting index of class (default = 0) or 1")
        .withShortName("sci").create();
    
    Option hdfsServerOpt = obuilder.withLongName("HDFSServer").withRequired(
      false).withArgument(
      abuilder.withName("HDFSServer").withMinimum(1).withMaximum(1).create())
        .withDescription("HDFS Server's Address (default = null) ")
        .withShortName("hdfs").create();
    
    Option svmTypeOpt = obuilder
        .withLongName("svmType")
        .withRequired(false)
        .withArgument(
          abuilder.withName("svmType").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "0 -> Binary Classfication, 1 -> Regression, "
              + "2 -> Multi-Classification (one-vs.-one), 3 -> Multi-Classification (one-vs.-others) ")
        .withShortName("s").create();
    
    Option modelFileOpt = obuilder.withLongName("modelFile").withRequired(true)
        .withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of model file (default = noModelFile) ")
        .withShortName("m").create();
    
    // hadoop system setting.
    Option mapSplitSizeOpt = obuilder.withLongName("mapSplitSize")
        .withRequired(false).withArgument(
          abuilder.withName("mapSplitSize").withMinimum(1).withMaximum(1)
              .create()).withDescription("Max map Split size ").withShortName(
          "ms").create();
    
    Option maxHeapSizeOpt = obuilder.withLongName("maxHeapSize").withRequired(
      false).withArgument(
      abuilder.withName("maxHeapSize").withMinimum(1).withMaximum(1).create())
        .withDescription("Max Heap Size: ").withShortName("mhs").create();
    
    Option numberofReducersOpt = obuilder.withLongName("numberofReducers")
        .withRequired(false).withArgument(
          abuilder.withName("numberofReducers").withMinimum(1).withMaximum(1)
              .create()).withDescription("Number of Reducers: (defaults = 0)")
        .withShortName("nor").create();
    
    Option taskTimeoutOpt = obuilder.withLongName("taskTimeout").withRequired(
      false).withArgument(
      abuilder.withName("taskTimeout").withMinimum(1).withMaximum(1).create())
        .withDescription("Task Time out ( Minutes ) : ").withShortName("ttt")
        .create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription(
      "Print out help").withShortName("h").create();
    
    Group group = gbuilder.withName("Options").withOption(trainFileOpt)
        .withOption(outputFileOpt).withOption(lambdaOpt).withOption(iterOpt)
        .withOption(kOpt).withOption(svmTypeOpt).withOption(classNumOpt)
        .withOption(hdfsServerOpt).withOption(modelFileOpt).withOption(
          startingClassIndexOpt).withOption(sampleNumOpt).withOption(
          mapSplitSizeOpt).withOption(maxHeapSizeOpt)
        .withOption(taskTimeoutOpt).withOption(numberofReducersOpt).withOption(
          helpOpt).create();
    
    SVMParameters para = new SVMParameters();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      para.setTrainFile(cmdLine.getValue(trainFileOpt).toString());
      para.setOutFile(cmdLine.getValue(outputFileOpt).toString());
      
      // lambda
      if (cmdLine.hasOption(lambdaOpt)) {
        para.setLambda(Double
            .parseDouble(cmdLine.getValue(lambdaOpt).toString()));
      }
      // iteration
      if (cmdLine.hasOption(iterOpt)) {
        para.setMaxIter(Integer.parseInt(cmdLine.getValue(iterOpt).toString()));
      }
      // k
      if (cmdLine.hasOption(kOpt)) {
        para.setExamplesPerIter(Integer.parseInt(cmdLine.getValue(kOpt)
            .toString()));
      }
      // class number
      para.setClassNum(Integer
          .parseInt(cmdLine.getValue(classNumOpt).toString()));
      // number of samples in training data set.
      if (cmdLine.hasOption(sampleNumOpt)) {
        para.setTrainSampleNumber(Integer.parseInt(cmdLine
            .getValue(sampleNumOpt).toString()));
      }
      
      if (cmdLine.hasOption(startingClassIndexOpt)) {
        para.setStartingClassIndex(Integer.parseInt(cmdLine.getValue(
          startingClassIndexOpt).toString()));
      }
      // models' path
      para.setModelFileName(cmdLine.getValue(modelFileOpt).toString());
      // hdfs server address
      if (cmdLine.hasOption(hdfsServerOpt)) {
        para.setHdfsServerAddr(cmdLine.getValue(hdfsServerOpt).toString());
      }
      // multi classification classificationType
      if (cmdLine.hasOption(svmTypeOpt)) {
        para.setClassificationType(Integer.parseInt(cmdLine.getValue(svmTypeOpt)
            .toString()));
      }
      // MapReduce system setting.
      if (cmdLine.hasOption(mapSplitSizeOpt)) {
        para.setMapSplitSize(Long.parseLong(cmdLine.getValue(mapSplitSizeOpt)
            .toString()));
      }
      if (cmdLine.hasOption(numberofReducersOpt)) {
        para.setNumberReducers(Integer.parseInt(cmdLine.getValue(
          numberofReducersOpt).toString()));
      }
      if (cmdLine.hasOption(maxHeapSizeOpt)) {
        para.setMaxHeapSize(cmdLine.getValue(maxHeapSizeOpt).toString());
      }
      if (cmdLine.hasOption(taskTimeoutOpt)) {
        para.setTaskTimeout(Long.parseLong(cmdLine.getValue(taskTimeoutOpt)
            .toString()));
      }
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
    
    // set parameters for the mapper, combiner, reducer
    
    // creat a job
    Job job = new Job(new Configuration());
    
    // step 1.1 set job static parameters
    ParallelMultiClassifierTrainJob.setJobParameters(job);
    
    // step 1.2 set mapper parameters
    ParallelMultiClassifierTrainJob.setMapperParameters(job
        .getConfiguration(), para.getMaxIter(), para.getTrainSampleNumber(),
      para.getClassNum(), para.getClassificationType(), para.getStartingClassIndex());
    
    ParallelMultiClassifierTrainJob.setReducerParameters(job
        .getConfiguration(), (float) para.getLambda(), para.getExamplesPerIter(),
      para.getModelFileName(), para.getHdfsServerAddr());
    
    // set general parameters related to a job
    MapReduceUtil.setJobParameters(job, para.getTrainFile(), para.getOutFile(),
      para.getMapSplitSize(), para.getNumberReducers(), para.getMaxHeapSize(),
      para.getTaskTimeout());
    
    // submit a job
    log.info("job completed: " + MapReduceUtil.submitJob(job));
  }
}
