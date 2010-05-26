/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.apache.mahout.classifier.svm.mapreduce;

import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * A string converter for <code>Map<String,String></code>.
 * 
 */
public class MapStringConverter implements StringConverter {
  
  private static final Logger log = LoggerFactory.getLogger(MapStringConverter.class);
  
  /** The key indicator */
  private String keyIndicator; // "--";
  /** The separator between the key and the value */
  private String keyValueSeparator; // " ";
  /** A map used to store the key-value parameter pair */
  private Map<String,String> keyValueMap;
  
//  public static void main(String[] args) throws Exception {
//    args = new String[] {"--oxx", "1,2 ", "--p", "2", "2", "--x -x -t", "--y"};
//    Map<String,String> m = new HashMap<String,String>();
//    MapStringConverter mp = new MapStringConverter(m, "--", " ");
//    mp.fromStringRepresentation(args);
//    log.info(m.get("oxx"));
//    log.info(m.get("p"));
//    log.info(m.get("x"));
//  }
  
  /**
   * Constructs a converter with the given <code>keyIndicator</code> and
   * <code>keyValueSeparator</code>.
   * 
   * @param map
   *          the key-value map
   * @param keyIndicator
   *          the key indicator
   * @param keyValueSeparator
   *          the key-value separator
   */
  public MapStringConverter(Map<String,String> map,
                            String keyIndicator,
                            String keyValueSeparator) {
    this.keyValueMap = map;
    this.keyIndicator = keyIndicator;
    this.keyValueSeparator = keyValueSeparator;
  }
  
  /**
   * Gets the key indicator.
   * 
   * @return
   */
  public String getKeyIndicator() {
    return this.keyIndicator;
  }
  
  /**
   * Gets the key-value separator.
   * 
   * @return
   */
  public String getKeyValueSeparator() {
    return this.keyValueSeparator;
  }
  
  /**
   * Gets the key-value map.
   * 
   * @return the key-value map.
   */
  public Map<String,String> getKeyValueMap() {
    return this.keyValueMap;
  }
  
  /**
   * Parses a string of command into key-value pairs.
   * 
   * <p>
   * Each parameter contains four parts:<br>
   * <ol>
   * <li>Key indicator: Character '--' indicates starting of a key.</li>
   * <li>Key: It should not contain spaces.</li>
   * <li>Key-value separator: Using space to separate a key and its value.</li>
   * <li>value: It should not contain the key indicator. Note that the heading
   * and trailing spaces are trimmed.</li>
   * </ol>
   * 
   * <p>
   * For example, the following command:<br>
   * "-key1 value1_part1 value1_part2 -key2 value2" means<br>
   * key1=value1_part1 value1_part2 <br>
   * key2=value2
   * 
   * @param str
   */
  public void fromStringRepresentation(String str) throws Exception {
    this.keyValueMap.clear();
    String[] c = str.split(this.keyIndicator); // token the parameters
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
   * @throws Exception
   * @see #parse(java.lang.String)
   */
  public void fromStringRepresentation(String[] command) throws Exception {
    String com = command[0];
    for (int i = 1; i < command.length; i++) {
      com += " " + command[i];
    }
    this.fromStringRepresentation(com);
  }
  
  public String toStringRepresentation() throws Exception {
    throw new UnsupportedOperationException("Not supported yet.");
  }
}
