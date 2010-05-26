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
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class HDFSReader {
  
  private static final Logger log = LoggerFactory.getLogger(HDFSReader.class);
  private Configuration conf;
  
  // public static void main(String[] args) throws IOException {
  //
  // HDFSReader rd = new HDFSReader();
  // String hostName = "hdfs://localhost:12009";
  // rd.setServerAddress(hostName);
  // // rd.setServerUserPasswd("maximzhao,zhaozd1@");
  // Path file = new Path("/user/maximzhao/pokerovo/part-r-00000");
  //
  // List<String> lines = rd.readAllLines(file);
  // Iterator<String> elt = lines.iterator();
  // while (elt.hasNext()) {
  // String it = elt.next();
  // }
  // }
  public HDFSReader() {
    this.conf = new Configuration();
  }
  
  /**
   * Set HDFS server address
   * 
   * @param serverAdress
   */
  public void setServerAddress(String serverAdress) {
    HDFSConfig.setSeverAddress(this.conf, serverAdress);
  }
  
  /**
   * Set User account and password
   * 
   * @param userpass
   */
  public void setServerUserPasswd(String userpass) {
    HDFSConfig.setUSerPasswd(this.conf, userpass);
  }
  
  public List<String> readADirectory(Path filePath) {
    List<String> lines = new ArrayList<String>();
    try {
      FileSystem fs = FileSystem.get(this.conf);
      if (!fs.isFile(filePath)) {
        FileStatus[] fsList = fs.listStatus(filePath);
        for (FileStatus file : fsList) {
          if (!file.isDir()) {
            lines.addAll(readAllLines(new Path(filePath.toString() + "/"
                                               + file.getPath().getName())));
          }
        }
      }
    } catch (Exception e) {
      log.error("Exception: " + e.getMessage());
    }
    return lines;
  }
  
  public boolean isDir(Path filePath) {
    try {
      FileSystem fs = FileSystem.get(this.conf);
      if (fs.isFile(filePath)) {
        return false;
      } else {
        return true;
      }
    } catch (Exception e) {
      log.error("Exception: " + e.getMessage());
      return false;
    }
  }
  
  /**
   * Simple implementation of reading certain lines from HDFS server.
   * 
   * @param filePath
   *          file path in sever
   * @param startLine
   *          starting line
   * @param endLine
   *          endLine
   * @return
   * @throws IOException
   */
  public List<String> readLines(Path filePath, int startLine, int endLine) throws IOException {
    FileSystem fs = FileSystem.get(this.conf);
    // set buff to 64MB
    // LineReader lr = new LineReader(fs.open(filePath), 64000000);
    LineReader lr = new LineReader(fs.open(filePath));
    List<String> lines = new ArrayList<String>();
    Text line = new Text();
    try {
      for (int i = 0; i <= endLine; i++) {
        lr.readLine(line); // read
        if (line.getLength() < 1) {
          log.info("No more line! " + i);
          break;
        }
        if (i >= startLine) {
          lines.add(line.toString());
        }
      }
    } finally {
      lr.close();
    }
    return lines;
  }
  
  /**
   * Read all files. Very Dangerous.
   * 
   * @param filePath
   * @return
   * @throws IOException
   */
  public List<String> readAllLines(Path filePath) throws IOException {
    FileSystem fs = FileSystem.get(this.conf);
    // set buff to 64MB
    // LineReader lr = new LineReader(fs.open(filePath), 64000000);
    LineReader lr = new LineReader(fs.open(filePath));
    List<String> lines = new ArrayList<String>();
    Text line = new Text();
    try {
      while (lr.readLine(line) > 0) {
        lines.add(line.toString());
      }
    } finally {
      lr.close();
    }
    return lines;
  }
  
  /**
   * Read all lines in the range of (0, maxLines).
   * 
   * @param filePath
   *          file name.
   * @param maxLines
   *          maximun number of lines
   * @return
   * @throws IOException
   */
  public List<String> readAllLines(Path filePath, long maxLines) throws IOException {
    FileSystem fs = FileSystem.get(this.conf);
    // set buff to 64MB
    // LineReader lr = new LineReader(fs.open(filePath), 64000000);
    LineReader lr = new LineReader(fs.open(filePath));
    List<String> lines = new ArrayList<String>();
    Text line = new Text();
    long i = 0;
    try {
      while (lr.readLine(line) > 1 && i < maxLines) {
        lines.add(line.toString());
      }
    } finally {
      lr.close();
    }
    return lines;
  }
}
