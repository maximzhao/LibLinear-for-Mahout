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

import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.MasterNotRunningException;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.util.Bytes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 */
public class HBaseDatastore {
  
  private HTable table;
  private static final Logger log = LoggerFactory.getLogger(HBaseDatastore.class);
  
  public static void main(String[] args) throws Exception {
    
    HBaseDatastore test = new HBaseDatastore();
    test.creatTable();
    test.addRow();
    test.getRow();
  }
  
  public void creatTable() throws Exception {
    HBaseConfiguration config = new HBaseConfiguration();
    // Path hbaseSitePath = new
    // Path("/usr/local/hbase-0.20.2/conf/hbase-site.xml");
    config.set("hbase.master", "localhost:60000");
    // config.addResource(hbaseSitePath);
    
    HBaseAdmin admin = null;
    
    try {
      admin = new HBaseAdmin(config);
    } catch (MasterNotRunningException e) {
      throw new Exception(
          "Could not setup HBaseAdmin as no master is running, did you start HBase?...");
    }
    
    if (!admin.tableExists("testTable")) {
      admin.createTable(new HTableDescriptor("testTable"));
      
      // disable so we can make changes to it
      admin.disableTable("testTable");
      
      // lets add 2 columns
      admin.addColumn("testTable", new HColumnDescriptor("firstName"));
      admin.addColumn("testTable", new HColumnDescriptor("lastName"));
      
      // enable the table for use
      admin.enableTable("testTable");
      
    }
    
    // get the table so we can use it in the next set of examples
    this.setTable(new HTable(config, "testTable"));
  }
  
  public void addRow() {
    // lets put a new object with a unique "row" identifier, this is the key
    // HBase stores everything in bytes so you need to convert string to
    // bytes
    Put row = new Put(Bytes.toBytes("myID"));
    
    /*
     * lets start adding data to this row. The first parameter is the
     * "familyName" which essentially is the column name, the second parameter
     * is the qualifier, think of it as a way to sub-qualify values within a
     * particular column. For now we won't so we just make the qualifier name
     * the same as the column name. The last parameter is the actual value to
     * store
     */
    row.add(Bytes.toBytes("firstName"), Bytes.toBytes("firstName"), Bytes
        .toBytes("joe"));
    row.add(Bytes.toBytes("lastName"), Bytes.toBytes("lastName"), Bytes
        .toBytes("maxim"));
    
    try {
      // add it!
      this.getTable().put(row);
    } catch (Exception e) {
      // handle me!
    }
    
  }
  
  public void getRow() {
    // a GET fetches a row by it's identifier key
    Get get = new Get(Bytes.toBytes("myID"));
    
    Result result = null;
    try {
      // exec the get
      result = getTable().get(get);
    } catch (Exception e) {
      log.error(e.getMessage());
    }
    
    // Again the result speaks in terms of a familyName(column)
    // and a qualifier, since ours our both the same, we pass the same
    // value for both
    byte[] firstName = Bytes.toBytes("firstName");
    byte[] lastName = Bytes.toBytes("lastName");
    byte[] fnameVal = result.getValue(firstName, firstName);
    byte[] lnameVal = result.getValue(lastName, lastName);
    
    log.info(new String(fnameVal) + " " + new String(lnameVal));
  }

  public void setTable(HTable table) {
    this.table = table;
  }

  public HTable getTable() {
    return table;
  }
}
