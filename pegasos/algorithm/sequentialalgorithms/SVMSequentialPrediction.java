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
import org.apache.mahout.classifier.svm.algorithm.metafunctions.Prediction;
import org.apache.mahout.classifier.svm.algorithm.metafunctions.PredictionFactory;
import org.apache.mahout.classifier.svm.datastore.DataSetHandler;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The main entry of Prediction
 */
public class SVMSequentialPrediction {
  
  private static final Logger log = LoggerFactory
      .getLogger(SVMSequentialPrediction.class);
  
  public SVMSequentialPrediction() {

  }
  
  public static void main(String[] args) throws IOException, OptionException {
    if (args.length < 1) {
      args = new String[] {
                           "-te",
                           "../examples/src/test/resources/svmdataset/test.dat",
                           "-m",
                           "../examples/src/test/resources/svmdataset/SVM.model"};
      // args = new String[] {
      // "-te",
      // "/media/Data/MachineLearningDataset/triazines_scale.t",
      // "-m", "/home/maximzhao/SVMregression.model", "-s",
      // "1"};
      // args = new String[] {
      // "-te",
      // "/media/Data/MachineLearningDataset/rcv1_train.binary",
      // "-m", "/home/maximzhao/SVMrcv1.model"};
      // args = new String[] {"-te",
      // "/media/Data/MachineLearningDataset/protein.t",
      // "-m", "/home/maximzhao/sectormulti/SVMprotein.model",
      // "-s", "2"};
      // args = new String[] {"-te",
      // "/media/Data/MachineLearningDataset/poker.t",
      // "-m", "/home/maximzhao/sectormulti/SVMpoker.model",
      // "-s", "3"};
      // args = new String[] {"-te", "/media/Data/MachineLearningDataset/poker",
      // "-m", "/user/maximzhao/pokerpro", "-s", "3",
      // "-hdfs", "hdfs://localhost:12009"};
    }
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option testFileOpt = obuilder.withLongName("testFile").withRequired(true)
        .withArgument(
          abuilder.withName("testFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of test data file (default = noTestFile)")
        .withShortName("te").create();
    
    Option svmTypeOpt = obuilder
        .withLongName("svmType")
        .withRequired(false)
        .withArgument(
          abuilder.withName("svmType").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "0 -> Binary Classfication, 1 -> Regression, "
              + "2 -> Multi-Classification (one-vs.-one), 3 -> Multi-Classification (one-vs.-others) ")
        .withShortName("s").create();
    
    Option modelFileOpt = obuilder
        .withLongName("modelFile")
        .withRequired(true)
        .withArgument(
          abuilder.withName("modelFile").withMinimum(1).withMaximum(1).create())
        .withDescription("Name of model file (default = noModelFile) ")
        .withShortName("m").create();
    
    Option hdfsServerOpt = obuilder.withLongName("HDFSServer").withRequired(
      false).withArgument(
      abuilder.withName("HDFSServer").withMinimum(1).withMaximum(1).create())
        .withDescription("HDFS Server's Address (default = null) ")
        .withShortName("hdfs").create();
    
    Option predictedFileOpt = obuilder.withLongName("predictedFile")
        .withRequired(false).withArgument(
          abuilder.withName("predictedFile").withMinimum(1).withMaximum(1)
              .create()).withDescription(
          "File to store predicted label(default = testFile.predict) ")
        .withShortName("p").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription(
      "Print out help").withShortName("h").create();
    
    Group group = gbuilder.withName("Options").withOption(modelFileOpt)
        .withOption(predictedFileOpt).withOption(testFileOpt).withOption(
          svmTypeOpt).withOption(helpOpt).withOption(hdfsServerOpt).create();
    
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
      para.setModelFileName(cmdLine.getValue(modelFileOpt).toString());
      
      // svm classificationType
      if (cmdLine.hasOption(svmTypeOpt)) {
        para.setClassificationType(Integer.parseInt(cmdLine.getValue(svmTypeOpt)
            .toString()));
      } else {
        para.setClassificationType(0); // default classfication
      }
      
      if (cmdLine.hasOption(predictedFileOpt)) {
        para.setOutFile(cmdLine.getValue(predictedFileOpt).toString());
      } else {
        para.setOutFile(para.getTestFile() + ".predict");
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
    
    // load test data set
    DataSetHandler test = new DataSetHandler(para.getTestFile());
    
    Prediction predictor = PredictionFactory.getInstance(para.getClassificationType());
    predictor.prediction(test, para);
    para.report(para.getClassificationType());
    log.info("Done!");
  }
}
