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
public class ParallelClassifierDriver {
  
  /** Job name */
  public static final String JOB_NAME = ParallelClassifierJob.JOB_NAME;
  private static final Logger log = LoggerFactory
      .getLogger(ParallelClassifierDriver.class);
  
  public static void main(String[] args) throws IOException,
                                        InterruptedException,
                                        ClassNotFoundException,
                                        OptionException {
    
    // example args:
    // -if /user/maximzhao/dataset/rcv1_test.binary -of
    // /user/maximzhao/rcv.result
    // -m /user/maximzhao/rcv1.model -nor 1 -ms 241572968 -mhs -Xmx500M -ttt
    // 1080
    log.info("[job] " + JOB_NAME);
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option testFileOpt = obuilder.withLongName("testFile").withRequired(true)
        .withArgument(
          abuilder.withName("testFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of test data file (default = noTestFile)")
        .withShortName("if").create();
    
    Option outputFileOpt = obuilder.withLongName("output").withRequired(true)
        .withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("Out put file name: ").withShortName("of").create();
    
    Option hdfsServerOpt = obuilder.withLongName("HDFSServer").withRequired(
      false).withArgument(
      abuilder.withName("HDFSServer").withMinimum(1).withMaximum(1).create())
        .withDescription("HDFS Server's Address (default = null) ")
        .withShortName("hdfs").create();
    
    Option modelFileOpt = obuilder
        .withLongName("modelFile")
        .withRequired(true)
        .withArgument(
          abuilder.withName("modelFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of model file (default = noModelFile) ")
        .withShortName("m").create();
    
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
    
    Group group = gbuilder.withName("Options").withOption(modelFileOpt)
        .withOption(testFileOpt).withOption(mapSplitSizeOpt).withOption(
          hdfsServerOpt).withOption(outputFileOpt).withOption(maxHeapSizeOpt)
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
      
      para.setTestFile(cmdLine.getValue(testFileOpt).toString());
      para.setOutFile(cmdLine.getValue(outputFileOpt).toString());
      para.setModelFileName(cmdLine.getValue(modelFileOpt).toString());
      
      // hdfs server address
      if (cmdLine.hasOption(hdfsServerOpt)) {
        para.setHdfsServerAddr(cmdLine.getValue(hdfsServerOpt).toString());
      }
      
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
    ParallelClassifierJob.setJobParameters(job);
    
    // step 1.2 set mapper parameters
    ParallelClassifierJob.setMapperParameters(job.getConfiguration(),
      para.getHdfsServerAddr(), para.getModelFileName());
    
    // set general parameters related to a job
    MapReduceUtil.setJobParameters(job, para.getTestFile(), para.getOutFile(),
      para.getMapSplitSize(), para.getNumberReducers(), para.getMaxHeapSize(),
      para.getTaskTimeout());
    
    // submit a job
    log.info("job completed: " + MapReduceUtil.submitJob(job));
  }
}
