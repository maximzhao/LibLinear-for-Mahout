/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.apache.mahout.classifier.svm.mapreduce;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A key-value parameter parser.
 * 
 */
public class KeyValueParameterParser {
  private static final Logger log = LoggerFactory.getLogger(KeyValueParameterParser.class);
  /** The key indicator */
  protected String keyIndicator = "--";
  /** The separator between the key and the value */
  protected String keyValueSeparator = " ";
  /** A map used to store the key-value parameter pair */
  protected Map<String,String> keyValueMap = new HashMap();
  
//  public static void main(String[] args) {
//    args = new String[] {"--oxx", "1,2 ", "--p", "2", "2", "--x -x -t", "--y"};
//    KeyValueParameterParser mp = new KeyValueParameterParser();
//    mp.parse(args);
//  }
  
  /**
   * Constructs a parser with the default <code>keyIndicator</code> and
   * <code>keyValueSeparator</code>.
   * 
   */
  public KeyValueParameterParser() {
    new MapStringConverter(new HashMap(), "--", " ");
  }
  
  /**
   * Constructs a parser with the given <code>keyIndicator</code> and
   * <code>keyValueSeparator</code>.
   * 
   * @param keyIndicator
   *          the key
   * @param keyValueSeparator
   */
  public KeyValueParameterParser(String keyIndicator, String keyValueSeparator) {
    new MapStringConverter(new HashMap(), keyIndicator,
        keyValueSeparator);
  }
  
  /**
   * Print the key-value pairs.
   * 
   */
  public void listParameters() {
    for (Entry<String,String> e : this.keyValueMap.entrySet()) {
      log.info(e.getKey() + " <-> " + e.getValue());
    }
  }
  
  public void parse(String commandStr) {
    this.keyValueMap.clear();
    String[] c = commandStr.split(this.keyIndicator); // token the parameters
    String[] t = null;
    for (int i = 0; i < c.length; i++) {
      t = c[i].split(this.keyValueSeparator); // token the key and value
      if (t.length < 2) {
        continue; // must contain the key and the value
      }
      keyValueMap.put(t[0], (c[i].substring(c[i]
          .indexOf(this.keyValueSeparator))).trim());
    }
  }
  
  /**
   * Parse a string which is constructed by connecting the elements of the
   * string array with space. It is used to accept commands from terminal.
   * 
   * @param command
   * @see #parse(java.lang.String)
   */
  public void parse(String[] command) {
    String com = command[0];
    for (int i = 1; i < command.length; i++) {
      com += " " + command[i];
    }
    this.parse(com);
  }
  
  /**
   * Gets the value of a key.
   * 
   * @param key
   *          a key
   * @return the value of the given key, null if the key doesn't exists
   */
  public String getValue(String key) {
    return this.keyValueMap.get(key);
  }
  
  /**
   * Gets the key-value entry set.
   * 
   * @param key
   * @return
   */
  public Set<Entry<String,String>> getKeyValueEntrySet(String key) {
    return this.keyValueMap.entrySet();
  }
  
  /**
   * Gets the key-value map.
   * 
   * @return the key-value map.
   */
  public Map<String,String> getKeyValueMap() {
    return this.keyValueMap;
  }
}
