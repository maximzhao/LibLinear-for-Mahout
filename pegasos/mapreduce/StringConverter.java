/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.apache.mahout.classifier.svm.mapreduce;

/**
 * XML converter.
 * 
 */
public interface StringConverter {
  
  /**
   * Reconstructs the object from its string representation.
   * 
   * @param str
   *          str representation of the object
   * @throws java.lang.Exception
   */
  void fromStringRepresentation(String str) throws Exception;
  
  /**
   * Gets the string representation of the object.
   * 
   * @return string representation of the object
   * @throws Exception
   */
  String toStringRepresentation() throws Exception;
}
