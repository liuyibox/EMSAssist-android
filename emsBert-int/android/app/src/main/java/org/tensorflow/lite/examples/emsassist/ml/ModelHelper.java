/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.tensorflow.lite.examples.emsassist.ml;

import static com.google.common.base.Verify.verify;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

/** Helper to load TfLite model and dictionary. */
public class ModelHelper {

  private static final String TAG = "ModelHelper";
  public static final String MODEL_PATH = "model.tflite";
  public static final String EMS_MODEL_PATH = "FineTune_BertBase4_Fitted_Desc_Nopi.tflite";
  public static final String DIC_PATH = "vocab.txt";
  public static final String TEST_FILE_PATH = "test_0.txt";
  public static final String LABEL_FILE_PATH = "no-fitted_label_names.txt";

  public static final String FITTED_LABEL_FILE_PATH = "fitted_label_names.txt";

  public static String TFLITE_TEST_RESULT_FILE;

  private ModelHelper() {}

  /** Load test model from context. */
  public static List<String> loadTestData(Context context) {

    String[] file_strs = TEST_FILE_PATH.split("\\.");
    TFLITE_TEST_RESULT_FILE = file_strs[0] + "_result.txt";
    Log.v(TAG, String.format(" test results will be written to %s", TFLITE_TEST_RESULT_FILE));

    List<String> testData = new ArrayList<>();
    BufferedReader reader = null;
    int total_line_num = 0;
    try {
      reader = new BufferedReader(
              new InputStreamReader(context.getAssets().open(TEST_FILE_PATH)));
//      reader.readLine();
      String mLine;
      while ((mLine = reader.readLine()) != null) {
        testData.add(mLine);
        if(total_line_num < 2){
          Log.v(TAG, String.format("reading test data at line %d is %s", total_line_num, mLine));
        }
        total_line_num += 1;
      }
    } catch (IOException e) {
      Log.e(TAG, e.getMessage());
    }
    Log.i(TAG,String.format("total test data number %d\n", total_line_num));
    return testData;
  }

  /** Load label names from context. */
  public static HashMap<String, Integer> loadLabels(Context context) {
    HashMap<String, Integer> label_map = new HashMap<>();
    List<String> labels = new ArrayList<>();
    BufferedReader reader = null;
    int total_line_num = 0;
    try {
      reader = new BufferedReader(
              new InputStreamReader(context.getAssets().open(LABEL_FILE_PATH)));
      String mLine;
      while ((mLine = reader.readLine()) != null) {
        labels.add(mLine);
        if(total_line_num < 1){
          Log.v(TAG, String.format("reading label at line %d is %s", total_line_num, mLine));
        }
        total_line_num += 1;
      }
    } catch (IOException e) {
      Log.e(TAG, e.getMessage());
    }
    Log.i(TAG,String.format("total labels: %d\n", total_line_num));
    for(int i = 0; i < labels.size(); i++){
      label_map.put(labels.get(i), i);
    }
    return label_map;
  }

  /** Load label names from context. */
  public static HashMap<String, Integer> loadFittedLabels(Context context) {
    HashMap<String, Integer> label_map = new HashMap<>();
    List<String> labels = new ArrayList<>();
    BufferedReader reader = null;
    int total_line_num = 0;
    try {
      reader = new BufferedReader(
              new InputStreamReader(context.getAssets().open(FITTED_LABEL_FILE_PATH)));
      String mLine;
      while ((mLine = reader.readLine()) != null) {
        labels.add(mLine);
        if(total_line_num < 1){
          Log.v(TAG, String.format("reading label at line %d is %s", total_line_num, mLine));
        }
        total_line_num += 1;
      }
    } catch (IOException e) {
      Log.e(TAG, e.getMessage());
    }
    Log.i(TAG,String.format("total labels: %d\n", total_line_num));
    for(int i = 0; i < labels.size(); i++){
      label_map.put(labels.get(i), i);
    }
    return label_map;
  }

  /** Load tflite model from context. */
  public static MappedByteBuffer loadModelFile(Context context) throws IOException {
    return loadModelFile(context.getAssets());
  }

  /** Load tflite ems_bert_model from context. */
  public static MappedByteBuffer loadEMSModelFile(Context context) throws IOException {
    return loadEMSModelFile(context.getAssets());
  }

  /** Load tflite model from assets. */
  public static MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Load tflite ems model from assets. */
  public static MappedByteBuffer loadEMSModelFile(AssetManager assetManager) throws IOException {
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(EMS_MODEL_PATH);
         FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Extract dictionary from metadata. */
  public static Map<String, Integer> extractDictionary(MetadataExtractor metadataExtractor) {
    Map<String, Integer> dic = null;
    try {
      verify(metadataExtractor != null, "metadataExtractor can't be null.");
      dic = loadDictionaryFile(metadataExtractor.getAssociatedFile(DIC_PATH));
      Log.v(TAG, "Dictionary loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
    return dic;
  }

  /** Load dictionary from assets. */
  public static Map<String, Integer> loadDictionaryFile(InputStream inputStream)
      throws IOException {
    Map<String, Integer> dic = new HashMap<>();
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
      int index = 0;
      while (reader.ready()) {
        String key = reader.readLine();
        dic.put(key, index++);
      }
    }
    return dic;
  }
}
