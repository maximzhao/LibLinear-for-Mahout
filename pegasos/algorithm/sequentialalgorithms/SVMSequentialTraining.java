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
package org.apache.mahout.classifier.svm.algorithm.sequentialalgorithms;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.Training;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.TrainingFactory;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * BinaryClassificationTraining part for SVM Pegasos
 * BinaryClassificationTraining e.g. -tr path-to-training-dataset -m
 * path-to-model-file -
 */
public class SVMSequentialTraining {
  
  private static final Logger log = LoggerFactory
      .getLogger(SVMSequentialTraining.class);
  
  public static void main(String[] args) throws IOException, OptionException {
    if (args.length < 1) {
      args = new String[] {
                           "-tr",
                           "../examples/src/test/resources/svmdataset/train.dat",
                           "-m",
                           "../examples/src/test/resources/svmdataset/SVM.model"};
//      args = new String[] {
//                           "-tr",
//                           "/media/Data/MachineLearningDataset/triazines_scale",
//                           "-m", "/home/maximzhao/SVMregression.model", "-s",
//                           "1"};
//      // for rcv1
//      args = new String[] {
//                           "-tr",
//                           "/media/Data/MachineLearningDataset/rcv1_test.binary",
//                           "-m", "/home/maximzhao/SVMrcv1.model", "-ts",
//                           "677399"};
//      args = new String[] {"-tr", "/media/Data/MachineLearningDataset/protein",
//                           "-m", "/home/maximzhao/sectormulti/SVMprotein.model",
//                           "-s", "2"};
//        args = new String[] {"-tr", "/media/Data/MachineLearningDataset/poker",
//                           "-m", "/home/maximzhao/sectormulti/SVMpoker.model",
//                           "-s", "3"};
//      args = new String[] {"-tr", "/user/maximzhao/dataset/train.dat", "-hdfs",
//                           "hdfs://localhost:12009", "-m",
//                           "../examples/src/test/resources/svmdataset/SVM.model"};
    }
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option trainFileOpt = obuilder
        .withLongName("trainFile")
        .withRequired(true)
        .withArgument(
          abuilder.withName("trainFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Training data set file").withShortName("tr").create();
    
    Option modelFileOpt = obuilder.withLongName("modelFile")
        .withRequired(false).withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of model file (default = noModelFile) ")
        .withShortName("m").create();
    
    Option svmTypeOpt = obuilder
        .withLongName("svmType")
        .withRequired(false)
        .withArgument(
          abuilder.withName("svmType").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "0 -> Binary Classfication, 1 -> Regression, "
              + "2 -> Multi-Classification (one-vs.-one), 3 -> Multi-Classification (one-vs.-others) ")
        .withShortName("s").create();
    
    Option epsilonOpt = obuilder.withLongName("epsilon").withRequired(false)
        .withArgument(
          abuilder.withName("epsilon").withMinimum(1).withMaximum(1).create())
        .withDescription("epsilon for regression (default = 0.1) ")
        .withShortName("e").create();
    
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
    
    Option validateExampleNumberOpt = obuilder.withLongName(
      "validateExampleNumber").withRequired(false).withArgument(
      abuilder.withName("validateExampleNumber").withMinimum(1).withMaximum(1)
          .create()).withDescription(
      "Number of validate Examples (default = Maximum iteration / 10) ")
        .withShortName("ven").create();
    
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
        .withShortName("ts").create();
    
    Option hdfsServerOpt = obuilder.withLongName("HDFSServer").withRequired(
      false).withArgument(
      abuilder.withName("HDFSServer").withMinimum(1).withMaximum(1).create())
        .withDescription("HDFS Server's Address (default = null) ")
        .withShortName("hdfs").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription(
      "Print out help").withShortName("h").create();
    
    Group group = gbuilder.withName("Options").withOption(trainFileOpt)
        .withOption(validateExampleNumberOpt).withOption(modelFileOpt)
        .withOption(svmTypeOpt).withOption(lambdaOpt).withOption(hdfsServerOpt)
        .withOption(iterOpt).withOption(epsilonOpt).withOption(kOpt)
        .withOption(sampleNumOpt).withOption(helpOpt).create();
    
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
      
      // svm classificationType
      if (cmdLine.hasOption(svmTypeOpt)) {
        para.setClassificationType(Integer.parseInt(cmdLine.getValue(svmTypeOpt)
            .toString()));
      }
      
      // epsilon
      if (cmdLine.hasOption(epsilonOpt)) {
        para.setEpsilon(Double.parseDouble(cmdLine.getValue(epsilonOpt)
            .toString()));
      }
      
      // lambda
      if (cmdLine.hasOption(lambdaOpt)) {
        para.setLambda(Double
            .parseDouble(cmdLine.getValue(lambdaOpt).toString()));
      }
      
      // iteration
      if (cmdLine.hasOption(iterOpt)) {
        para.setMaxIter(Integer.parseInt(cmdLine.getValue(iterOpt).toString()));
      }
      
      // iteration
      if (cmdLine.hasOption(validateExampleNumberOpt)) {
        para.setValidateExampleNumber(Integer.parseInt(cmdLine.getValue(
          validateExampleNumberOpt).toString()));
      } else {
        para.setValidateExampleNumber(para.getMaxIter() / 10);
      }
      
      // k
      if (cmdLine.hasOption(kOpt)) {
        para.setExamplesPerIter(Integer.parseInt(cmdLine.getValue(kOpt)
            .toString()));
      }
      
      if (cmdLine.hasOption(modelFileOpt)) {
        para.setModelFileName(cmdLine.getValue(modelFileOpt).toString());
      } else {
        para.setModelFileName("SVM.model");
      }
      
      // number of samples in training data set.
      if (cmdLine.hasOption(sampleNumOpt)) {
        para.setTrainSampleNumber(Integer.parseInt(cmdLine
            .getValue(sampleNumOpt).toString()));
      }
      
      // hdfs server address
      if (cmdLine.hasOption(hdfsServerOpt)) {
        para.setHdfsServerAddr(cmdLine.getValue(hdfsServerOpt).toString());
      } else {
        para.setHdfsServerAddr(null);
      }
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
    
    DataSetHandler train = new DataSetHandler(para.getTrainFile());
    
    // Get data set
    train.getData(para);
    
    Training classifier = TrainingFactory.getInstance(para.getClassificationType());
    classifier.training(train, para);
    para.report(para.getClassificationType());
    log.info("All Processes are Finished!!");
  }
}
