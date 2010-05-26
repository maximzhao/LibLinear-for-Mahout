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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;

/**
 * 
 */
public class HDFSConfig {
  
  public static void setSeverAddress(Configuration conf, String address) {
    conf.set("fs.default.name", address);
  }
  
  public static void setUSerPasswd(Configuration conf, String userPasswd) {
    conf.set("hadoop.job.ugi", userPasswd);
  }
  
  /**
   * Gets the local file system or the distributed system (HDFS).
   * 
   * @param conf
   * @param fsDefaultName
   * @return FileSystem
   * @throws java.io.IOException
   */
  public FileSystem getFileSystem(Configuration conf, String fsDefaultName) throws IOException {
    if (null == conf) {
      conf = new Configuration();
    }
    if (true) {
      conf.set("fs.default.name", fsDefaultName);
    }
    return FileSystem.get(conf);
  }
}
