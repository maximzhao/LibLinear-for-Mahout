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
package org.apache.mahout.classifier.svm.datastore.hbase;

import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.mahout.classifier.svm.datastore.hbase.conf.HBaseConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Create HBase Table
 * 
 */
public class HBaseCreateATable {
  
  private static final Logger log = LoggerFactory
      .getLogger(HBaseCreateATable.class);
  
  public HBaseCreateATable() {
    
  }
  
  /**
   * Create a table according to the table name and table columns
   * 
   * @param tableName
   * @param tableColumn
   *          List of name of table columns
   * @return
   */
  public static boolean createHBaseTable(String tableName, String[] tableColumn) {
    boolean result = true;
    try {
      HBaseAdmin admin = new HBaseAdmin(HBaseConfig.getConf());
      if (!admin.tableExists(tableName)) {
        HColumnDescriptor[] columns = new HColumnDescriptor[tableColumn.length];
        for (int i = 0; i < tableColumn.length; i++) {
          columns[i] = new HColumnDescriptor(tableColumn[i]);
        }
        
        HTableDescriptor table = new HTableDescriptor(Bytes.toBytes(tableName));
        for (int i = 0; i < tableColumn.length; i++) {
          table.addFamily(columns[i]);
        }
        admin.createTable(table);
        admin.enableTable(tableName);
      } else {
        result = false;
      }
    } catch (Exception e) {
      log.error("Hbase Create Table Exception:" + e.getMessage());
    }
    return result;
  }
  
  /**
   * Delete one table
   * 
   * @param tableName
   * @return
   * @throws java.lang.Exception
   */
  public static boolean deleteHbaseTable(String tableName) throws java.lang.Exception {
    HBaseAdmin admin = new HBaseAdmin(HBaseConfig.getConf());
    if (admin.isTableEnabled(tableName)) {
      admin.disableTable(tableName);
    }
    admin.deleteTable(tableName);
    return true;
  }
}
