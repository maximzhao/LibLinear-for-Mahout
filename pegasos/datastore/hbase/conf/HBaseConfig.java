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
package org.apache.mahout.classifier.svm.datastore.hbase.conf;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.mahout.classifier.svm.parameters.SVMParameters;

/**
 * 
 */
public class HBaseConfig {
  
  private static final String HBASE_MASTER_VALUE = SVMParameters.DEFAULT_HBASE_SERVER;
  private static final String HBASE_MASTER_NAME = "hbase.master";
  private static final String HBASE_HOME_DIRECTORY = "/home/hbase-0.20.2/";
  private static final String HBASE_SITE = HBASE_HOME_DIRECTORY
                                           + "conf/hbase-site.xml";
  private static final String HBASE_DEFAULT = HBASE_HOME_DIRECTORY
                                              + "conf/hbase-default.xml";
  
  public HBaseConfig() {
    
  }
  
  /**
   * Get HBase Configuration
   * 
   * @return configuration instance
   */
  public static HBaseConfiguration getConf() {
    HBaseConfiguration conf = new HBaseConfiguration();
    conf.set(HBASE_MASTER_NAME, HBASE_MASTER_VALUE);
    return conf;
  }
  
  /**
   * Get HBase Configuration
   * 
   * @param hbaseHostPort
   *          address of hbase server, e.g. localhost:60000
   * @return configuration instance
   */
  public static HBaseConfiguration getConf(String hbaseHostPort) {
    HBaseConfiguration conf = new HBaseConfiguration();
    conf.set(HBASE_MASTER_NAME, hbaseHostPort);
    return conf;
  }
  
  /**
   * Set hbase-site.xml and hbase-default.xml. Users are required to call
   * getConf() first.
   * 
   * @param conf
   */
  public static void setLocalHbaseConfiguDirectory(HBaseConfiguration conf) {
    Path hbaseSitePath = new Path(HBASE_SITE);
    Path hbaseDefaultPath = new Path(HBASE_DEFAULT);
    conf.addResource(hbaseSitePath);
    conf.addResource(hbaseDefaultPath);
  }
  
  /**
   * Set hbase-site.xml and hbase-default.xml. Users are required to call
   * getConf() first.
   * 
   * @param conf
   * @param hbaseSite
   *          /path/to/hbase
   */
  public static void setLocalHbaseConfiguDirectory(HBaseConfiguration conf,
                                                   String hbaseSite) {
    Path hbaseSitePath = new Path(hbaseSite + "conf/hbase-site.xml");
    Path hbaseDefaultPath = new Path(hbaseSite + "conf/hbase-default.xml");
    conf.addResource(hbaseSitePath);
    conf.addResource(hbaseDefaultPath);
  }
}
