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

import android.annotation.SuppressLint;
import android.os.Environment;

import static com.google.common.base.Verify.verify;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import androidx.annotation.WorkerThread;
import com.google.common.base.Joiner;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.Tensor;

/** Interface to load TfLite model and provide predictions. */
public class QaClient implements AutoCloseable {
  private static final String TAG = "QaClient";

  private static final int MAX_ANS_LEN = 32;
  private static final int MAX_QUERY_LEN = 128;
//  private static final int MAX_SEQ_LEN = 384;
  private static final int MAX_SEQ_LEN = 128;
  private static final int NUM_PRED_NO_FITTED_CLASSES = 102;
  private static final int NUM_PRED_CLASSES = 46;
  private static final int BATCH_SIZE = 32;

  private static final boolean DO_LOWER_CASE = true;
  private static final int PREDICT_ANS_NUM = 5;
  private static final int NUM_LITE_THREADS = 4;

  private static final String IDS_TENSOR_NAME = "input_ids_1:0";
  private static final String MASK_TENSOR_NAME = "input_mask_1:0";
  private static final String SEGMENT_IDS_TENSOR_NAME = "segment_ids_1:0";
  private static final String END_LOGITS_TENSOR_NAME = "end_logits";
  private static final String START_LOGITS_TENSOR_NAME = "start_logits";

  // Need to shift 1 for outputs ([CLS]).
  private static final int OUTPUT_OFFSET = 1;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final FeatureConverter featureConverter;
  private Interpreter tflite;
  private MetadataExtractor metadataExtractor = null;

  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  public QaClient(Context context) {
    this.context = context;
    this.featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
  }

  private List<String> testData;
  private HashMap<String, Integer> label_map;
  private HashMap<String, Integer> fitted_label_map;
  private static final int K = 5;
  private String PROB_TEXT_PATH = Environment.getExternalStorageDirectory().getPath() +
          "/distressnet/MStorm/EMSFiles/";

  @WorkerThread
  public synchronized void loadModel() {
    try {
//      ByteBuffer buffer = ModelHelper.loadModelFile(context);
      testData = ModelHelper.loadTestData(context);
      label_map = ModelHelper.loadLabels(context);
      fitted_label_map = ModelHelper.loadFittedLabels(context);

      ByteBuffer buffer = ModelHelper.loadEMSModelFile(context);
      metadataExtractor = new MetadataExtractor(buffer);
      Map<String, Integer> loadedDic = ModelHelper.extractDictionary(metadataExtractor);
      verify(loadedDic != null, "dic can't be null.");
      dic.putAll(loadedDic);

      Interpreter.Options opt = new Interpreter.Options();
      opt.setNumThreads(NUM_LITE_THREADS);
      tflite = new Interpreter(buffer, opt);

      Log.v(TAG, "TFLite model loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized void unload() {
    close();
  }

  @Override
  public void close() {
    if (tflite != null) {
      tflite.close();
      tflite = null;
    }
    dic.clear();
  }

  @SuppressLint("DefaultLocale")
  public synchronized void run_pp_test_for_no_fitted_batch(){

    float[][] emsPredLogits = new float[BATCH_SIZE][NUM_PRED_NO_FITTED_CLASSES];
    int[][] inputIds = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][] inputMask = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][] segmentIds = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][][] inputs = new int[3][BATCH_SIZE][MAX_SEQ_LEN];

    for(int f_idx = 0; f_idx < 10; f_idx++){
      String test_data_file = "no-fitted_desc_nosi_test_" + f_idx + ".txt";
      String test_result_file = "no-fitted_desc_nosi_test_" + f_idx + "_result.txt";
      String test_latency_file = "no-fitted_desc_nosi_test_" + f_idx + "_latency.txt";

      List<List<String>> testDataInBatch = new ArrayList<>();
      BufferedReader reader = null;
      int total_line_num = 0;
      try {
        reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(test_data_file)));
//      reader.readLine();
        String mLine;
        List<String> current_batch = new ArrayList<>();
        while ((mLine = reader.readLine()) != null) {
          if(current_batch.size() == BATCH_SIZE){
            testDataInBatch.add(current_batch);
            if(current_batch.size() < BATCH_SIZE){
              Log.v(TAG, String.format("current batch size %d when appending", current_batch.size()));
            }
            current_batch = new ArrayList<>();
          }
          current_batch.add(mLine);
          if (total_line_num < 2) {
            Log.v(TAG, String.format("reading test data at line %d is %s", total_line_num, mLine));
          }
          total_line_num += 1;
        }
      } catch (IOException e) {
        Log.e(TAG, e.getMessage());
      }

      Log.i(TAG,String.format("total test data number %d in test file %s\n", total_line_num, test_data_file));
      Log.i(TAG,String.format("total batches %d\n", testDataInBatch.size()));
      String output_str_to_save = "";

      Map<Integer, Object> output = new HashMap<>();
      long total_latency = 0;

      // batch index
      int test_start_idx = 0;
      int test_end_idx = testDataInBatch.size();
//      int test_end_idx = 2;
      for(int test_idx = test_start_idx; test_idx < test_end_idx; test_idx++){    // check the first test data

        if(test_idx % 100 == 0){
          Log.v(TAG, String.format("we have processed %d test data prediction", test_idx * BATCH_SIZE));
        }

        List<String> data_str_batch = testDataInBatch.get(test_idx);

        for(int data_idx = 0; data_idx < BATCH_SIZE; data_idx++){

          String data_str = data_str_batch.get(data_idx);
          String[] sample_label = data_str.split("\t");
          String query = sample_label[0];
          String label_name = sample_label[1];
          Integer label_id = label_map.get(label_name);

          EMSBertFeature emsBertFeature = featureConverter.convert(query);
          inputIds[data_idx] = emsBertFeature.inputIds;
          inputMask[data_idx] = emsBertFeature.inputMask;
          segmentIds[data_idx] = emsBertFeature.segmentIds;

        }

        inputs[0] = inputIds;
//      inputs[1] = inputMask;
//      inputs[2] = segmentIds;
        inputs[2] = inputMask;
        inputs[1] = segmentIds;

        output.put(0, emsPredLogits);

        long infer_start = System.currentTimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, output);
        long infer_latency = System.currentTimeMillis() - infer_start;
        total_latency += infer_latency;

        if(test_idx - test_start_idx < 1) {
          StringBuilder outputStr = new StringBuilder();
          for(int out_idx = 0; out_idx < emsPredLogits[0].length; out_idx++){
            outputStr.append(String.format("%.7f", emsPredLogits[0][out_idx])).append(", ");
            if((out_idx + 1) % 5 == 0) { outputStr.append("\n"); }
          }
          Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));
        }

        for(int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
          StringBuilder cur_out = new StringBuilder(String.format("%.7f", emsPredLogits[data_idx][0]));
          cur_out.append(" ");
          for (int i = 1; i < emsPredLogits[data_idx].length; i++) {
            cur_out.append(String.format("%.7f", emsPredLogits[data_idx][i])).append(" ");
          }
          cur_out.append("\n");
          output_str_to_save += cur_out;
        }
      }

      double averaged_latency = (total_latency * 1.0 / (test_end_idx - test_start_idx) / BATCH_SIZE);
//    double averaged_latency = total_latency;
      Log.v(TAG, String.format("Averaged time latency: %s", averaged_latency));

//      String tflite_test_result_file = ModelHelper.TFLITE_TEST_RESULT_FILE;
      try {

        String model_name = ModelHelper.EMS_MODEL_PATH.split("\\.")[0];
        File prob_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_result_file);
        Log.v(TAG, String.format("we are trying to create a file for result %s", prob_txt.toString()));
        prob_txt.createNewFile();
        FileWriter fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_result_file);
        fw.write(output_str_to_save);
        fw.flush();
        fw.close();

        //write out the inference latency
        File latency_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("we are trying to create a file for latency %s", latency_txt.toString()));
        latency_txt.createNewFile();
        fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("inference average latency %s", averaged_latency));
        fw.write(String.valueOf(averaged_latency));
        fw.flush();
        fw.close();

      } catch (IOException e) {
        e.printStackTrace();
      }
      Log.v(TAG, String.format("finish writing result to result file %s and latency file %s",
              test_result_file, test_latency_file));
    }

  }

  @SuppressLint("DefaultLocale")
  public synchronized void run_pp_test_for_no_fitted(){

    // TO_DO needs modification to 0
    for(int f_idx = 0; f_idx < 19; f_idx++){
      String test_data_file = "no-fitted_desc_test_" + f_idx + ".txt";
      String test_result_file = "no-fitted_desc_test_" + f_idx + "_result.txt";
      String test_latency_file = "no-fitted_desc_test_" + f_idx + "_latency.txt";

      List<String> testData = new ArrayList<>();
      BufferedReader reader = null;
      int total_line_num = 0;
      try {
        reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(test_data_file)));
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
      Log.i(TAG,String.format("total test data number %d in test file %s\n", total_line_num, test_data_file));
//      return testData;

//      float top1 = 0.0f, top3 = 0.0f, top5 = 0.0f;

//    float[][] output_to_save = new float[testData.size()][NUM_PRED_CLASSES];
      String output_str_to_save = "";
//    for(; test_idx < testData.size(); test_idx++){
      float[][] emsPredLogits = new float[1][NUM_PRED_NO_FITTED_CLASSES];
      int[] inputIds = new int[MAX_SEQ_LEN];
      int[] inputMask = new int[MAX_SEQ_LEN];
      int[] segmentIds = new int[MAX_SEQ_LEN];
      int[][] inputs = new int[3][MAX_SEQ_LEN];

      Map<Integer, Object> output = new HashMap<>();
      long total_latency = 0;

//    for(int test_idx = 0; test_idx < testData.size(); test_idx++){
      int test_start_idx = 0;
      int test_end_idx = testData.size();
//    int test_end_idx = 10;
      for(int test_idx = test_start_idx; test_idx < test_end_idx; test_idx++){    // check the first test data

        if(test_idx % 1000 == 0){
          Log.v(TAG, String.format("we have processed %d test data prediction", test_idx));
        }

        String data_str = testData.get(test_idx);
        String[] sample_label = data_str.split("\t");
        String query = sample_label[0];
        String label_name = sample_label[1];
        EMSBertFeature emsBertFeature = featureConverter.convert(query);
//        Integer label_id = fitted_label_map.get(label_name);
        Integer label_id = label_map.get(label_name);

        if(test_idx - test_start_idx < 1){
          Log.i(TAG, String.format("We have converted sample %d, string: %s, label_name: %s, label_idx: %d",
                  test_idx, query, label_name, label_id));
          Log.v(TAG, "TFLite model: " + ModelHelper.EMS_MODEL_PATH + " running...");
          Log.v(TAG, "Convert Feature...");
          Log.v(TAG, "Set inputs...");
        }

        for (int j = 0; j < MAX_SEQ_LEN; j++) {
          inputIds[j] = emsBertFeature.inputIds[j];
          inputMask[j] = emsBertFeature.inputMask[j];
          segmentIds[j] = emsBertFeature.segmentIds[j];
        }

        if(test_idx - test_start_idx < 1){

          String inputIdsStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            inputIdsStr += inputIds[inp_idx] + ", ";
            if((inp_idx + 1) % 9 == 0) { inputIdsStr += "\n"; }
          }
          Log.v(TAG, String.format("inputIds: \n%s", inputIdsStr));

          String inputMaskStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            inputMaskStr += inputMask[inp_idx] + ", ";
            if((inp_idx + 1) % 23 == 0) { inputMaskStr += "\n"; }
          }
          Log.v(TAG, String.format("inputMask: \n%s", inputMaskStr));

          String segIdStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            segIdStr += segmentIds[inp_idx] + ", ";
            if((inp_idx + 1) % 23 == 0) { segIdStr += "\n"; }
          }
          Log.v(TAG, String.format("segmentIds: \n%s", segIdStr));
        }


        inputs[0] = inputIds;
//      inputs[1] = inputMask;
//      inputs[2] = segmentIds;
        inputs[2] = inputMask;
        inputs[1] = segmentIds;


        output.put(0, emsPredLogits);

        if(test_idx - test_start_idx < 1) {

          int input_count = tflite.getInputTensorCount();
          int output_count = tflite.getOutputTensorCount();
          // model make 2-class text classification bert

          int input_idx = tflite.getInputIndex("serving_default_input_word_ids:0");
          int mask_idx = tflite.getInputIndex("serving_default_input_mask:0");
          int segment_idx = tflite.getInputIndex("serving_default_input_type_ids:0");

          Tensor input_word_id_tensor = tflite.getInputTensor(0);
//          tflite.resizeInput(0, new int[]{1, input_word_id_tensor.shape()[1]});
          Log.v(TAG, String.format("input_word_id_tensor shape num_dim: %d\n", input_word_id_tensor.numDimensions()));
          Log.v(TAG, String.format("input_word_id_tensor shape[0]: %d\n", input_word_id_tensor.shape()[0]));
          Log.v(TAG, String.format("input_word_id_tensor shape[1]: %d\n", input_word_id_tensor.shape()[1]));

          Tensor segment_ids_tensor = tflite.getInputTensor(1);
//          tflite.resizeInput(1, new int[]{1, input_word_id_tensor.shape()[1]});
          Log.v(TAG, String.format("segment_ids_tensor shape num_dim: %d\n", segment_ids_tensor.numDimensions()));
          Log.v(TAG, String.format("segment_ids_tensor shape[0]: %d\n", segment_ids_tensor.shape()[0]));
          Log.v(TAG, String.format("segment_ids_tensor shape[1]: %d\n", segment_ids_tensor.shape()[1]));

          Tensor input_mask_tensor = tflite.getInputTensor(2);
//          tflite.resizeInput(2, new int[]{1, input_word_id_tensor.shape()[1]});
          Log.v(TAG, String.format("input_mask_tensor shape num_dim: %d\n", input_mask_tensor.numDimensions()));
          Log.v(TAG, String.format("input_mask_tensor shape[0]: %d\n", input_mask_tensor.shape()[0]));
          Log.v(TAG, String.format("input_mask_tensor shape[1]: %d\n", input_mask_tensor.shape()[1]));

          int prob_idx = tflite.getOutputIndex("StatefulPartitionedCall:0");
          Tensor out_tensor = tflite.getOutputTensor(0);
          Log.v(TAG, String.format("out_tensor shape num_dim: %d\n", out_tensor.numDimensions()));
          Log.v(TAG, String.format("out_tensor shape[0]: %d\n", out_tensor.shape()[0]));
          Log.v(TAG, String.format("out_tensor shape[1]: %d\n", out_tensor.shape()[1]));

          Log.v(TAG, String.format("input tensor count: %d, output tensor count: %d\n",
                  input_count, output_count));
          Log.v(TAG, String.format("Inputs: input_ids idx: %d, input_mask idx: %d, segment_ids idx: %d\n",
                  input_idx, mask_idx, segment_idx));
          Log.v(TAG, String.format("Outputs: probabilities idx: %d\n", prob_idx));

          Log.v(TAG, "Run inference...");
        }

        long infer_start = System.currentTimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, output);
        long infer_latency = System.currentTimeMillis() - infer_start;
        total_latency += infer_latency;

        if(test_idx - test_start_idx < 1) {
          StringBuilder outputStr = new StringBuilder();
          for(int out_idx = 0; out_idx < emsPredLogits[0].length; out_idx++){
            outputStr.append(String.format("%.7f", emsPredLogits[0][out_idx])).append(", ");
            if((out_idx + 1) % 5 == 0) { outputStr.append("\n"); }
          }
          Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));
        }

        StringBuilder cur_out = new StringBuilder(String.format("%.7f",emsPredLogits[0][0]));
        cur_out.append(" ");
        for(int i = 1; i < emsPredLogits[0].length; i++){
          cur_out.append(String.format("%.7f", emsPredLogits[0][i])).append(" ");
        }
        cur_out.append("\n");
        output_str_to_save += cur_out;

      }

      double averaged_latency = (total_latency * 1.0 / (test_end_idx - test_start_idx));
//    double averaged_latency = total_latency;
      Log.v(TAG, String.format("Averaged time latency: %s", averaged_latency));

//      String tflite_test_result_file = ModelHelper.TFLITE_TEST_RESULT_FILE;
      try {

        String model_name = ModelHelper.EMS_MODEL_PATH.split("\\.")[0];
        File prob_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_result_file);
        Log.v(TAG, String.format("we are trying to create a file %s", prob_txt.toString()));
        prob_txt.createNewFile();
        FileWriter fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_result_file);
        fw.write(output_str_to_save);
        fw.flush();
        fw.close();

        //write out the inference latency
        File latency_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("we are trying to create a file %s", latency_txt.toString()));
        latency_txt.createNewFile();
        fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("inference average latency %s", averaged_latency));
        fw.write(String.valueOf(averaged_latency));
        fw.flush();
        fw.close();

      } catch (IOException e) {
        e.printStackTrace();
      }
      Log.v(TAG, String.format("finish writing result to file %s", test_result_file));
    }

  }

  @SuppressLint("DefaultLocale")
  public synchronized void run_pp_test_for_fitted_batch(){

    float[][] emsPredLogits = new float[BATCH_SIZE][NUM_PRED_CLASSES];
    int[][] inputIds = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][] inputMask = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][] segmentIds = new int[BATCH_SIZE][MAX_SEQ_LEN];
    int[][][] inputs = new int[3][BATCH_SIZE][MAX_SEQ_LEN];

    for(int f_idx = 0; f_idx < 3; f_idx++) {

      // TO_DO needs modification to 0
      String test_data_file = "fitted_desc_nopi_test_" + f_idx + ".txt";
      String test_result_file = "fitted_desc_nopi_test" + f_idx + "_result.txt";
      String test_latency_file = "fitted_desc_nopi_test" + f_idx + "_latency.txt";

      List<List<String>> testData = new ArrayList<>();
      BufferedReader reader = null;
      int total_line_num = 0;
      try {
        reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(test_data_file)));
        reader.readLine();
        String mLine;
        List<String> current_batch = new ArrayList<>();
        while ((mLine = reader.readLine()) != null) {
          if (current_batch.size() == BATCH_SIZE) {
            testData.add(current_batch);
            current_batch = new ArrayList<>();
          }
          current_batch.add(mLine);
          if (total_line_num < 2) {
            Log.v(TAG, String.format("reading test data at line %d is %s", total_line_num, mLine));
          }
          total_line_num += 1;
        }
      } catch (IOException e) {
        Log.e(TAG, e.getMessage());
      }
      Log.i(TAG, String.format("total test data number %d in test file %s\n", total_line_num, test_data_file));
      Log.i(TAG, String.format("total batches %d\n", testData.size()));

      String output_str_to_save = "";
      Map<Integer, Object> output = new HashMap<>();
      long total_latency = 0;

      //    for(int test_idx = 0; test_idx < testData.size(); test_idx++){
      int test_start_idx = 0;
      int test_end_idx = testData.size();
      //      int test_end_idx = 10;
      for (int test_idx = test_start_idx; test_idx < test_end_idx; test_idx++) {    // check the first test data

        if (test_idx % 100 == 0) {
          Log.v(TAG, String.format("we have processed %d test data prediction", test_idx));
        }
        List<String> data_str_batch = testData.get(test_idx);

        for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {

          String data_str = data_str_batch.get(data_idx);
          String[] sample_label = data_str.split("\t");
          String query = sample_label[0];
          String label_name = sample_label[1];
          Integer label_id = label_map.get(label_name);

          EMSBertFeature emsBertFeature = featureConverter.convert(query);
          inputIds[data_idx] = emsBertFeature.inputIds;
          inputMask[data_idx] = emsBertFeature.inputMask;
          segmentIds[data_idx] = emsBertFeature.segmentIds;

        }

        inputs[0] = inputIds;
        //      inputs[1] = inputMask;
        //      inputs[2] = segmentIds;
        inputs[2] = inputMask;
        inputs[1] = segmentIds;


        output.put(0, emsPredLogits);

        long infer_start = System.currentTimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, output);
        long infer_latency = System.currentTimeMillis() - infer_start;
        total_latency += infer_latency;

        if (test_idx - test_start_idx < 1) {
          StringBuilder outputStr = new StringBuilder();
          for (int out_idx = 0; out_idx < emsPredLogits[0].length; out_idx++) {
            outputStr.append(String.format("%.7f", emsPredLogits[0][out_idx])).append(", ");
            if ((out_idx + 1) % 5 == 0) {
              outputStr.append("\n");
            }
          }
          Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));
        }

        for (int data_idx = 0; data_idx < BATCH_SIZE; data_idx++) {
          StringBuilder cur_out = new StringBuilder(String.format("%.7f", emsPredLogits[data_idx][0]));
          cur_out.append(" ");
          for (int i = 1; i < emsPredLogits[data_idx].length; i++) {
            cur_out.append(String.format("%.7f", emsPredLogits[data_idx][i])).append(" ");
          }
          cur_out.append("\n");
          output_str_to_save += cur_out;
        }
      }

      double averaged_latency = (total_latency * 1.0 / (test_end_idx - test_start_idx) / BATCH_SIZE);
      //    double averaged_latency = total_latency;
      Log.v(TAG, String.format("Averaged time latency: %s", averaged_latency));

      //      String tflite_test_result_file = ModelHelper.TFLITE_TEST_RESULT_FILE;
      try {
        // write out the inference result
        String model_name = ModelHelper.EMS_MODEL_PATH.split("\\.")[0];
        File prob_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_result_file);
        Log.v(TAG, String.format("we are trying to create a file for result %s", prob_txt.toString()));
        prob_txt.createNewFile();
        FileWriter fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_result_file);
        fw.write(output_str_to_save);
        fw.flush();
        fw.close();

        //write out the inference latency
        File latency_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("we are trying to create a file for latency %s", latency_txt.toString()));
        latency_txt.createNewFile();
        fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_latency_file);
        Log.v(TAG, String.format("inference average latency %s", averaged_latency));
        fw.write(String.valueOf(averaged_latency));
        fw.flush();
        fw.close();


      } catch (IOException e) {
        e.printStackTrace();
      }
      Log.v(TAG, String.format("finish writing result to result file %s and latency file %s",
              test_result_file, test_latency_file));
    }
  }

  @SuppressLint("DefaultLocale")
  public synchronized void run_pp_test_for_fitted_batch_am(){

    float[] emsPredLogits = new float[NUM_PRED_CLASSES];
    int[] inputIds = new int[MAX_SEQ_LEN];
    int[] inputMask = new int[MAX_SEQ_LEN];
    int[] segmentIds = new int[MAX_SEQ_LEN];
    int[][] inputs = new int[3][MAX_SEQ_LEN];

    String output_str_to_save = "";
    Map<Integer, Object> output = new HashMap<>();
    long total_latency = 0;

    String query = "";

    EMSBertFeature emsBertFeature = featureConverter.convert(query);
    inputIds = emsBertFeature.inputIds;
    inputMask = emsBertFeature.inputMask;
    segmentIds = emsBertFeature.segmentIds;

    inputs[0] = inputIds;
    inputs[2] = inputMask;
    inputs[1] = segmentIds;


    output.put(0, emsPredLogits);

    long infer_start = System.currentTimeMillis();
    tflite.runForMultipleInputsOutputs(inputs, output);
    long infer_latency = System.currentTimeMillis() - infer_start;
    total_latency += infer_latency;


    StringBuilder outputStr = new StringBuilder();
    for (int out_idx = 0; out_idx < emsPredLogits.length; out_idx++) {
      outputStr.append(String.format("%.7f", emsPredLogits[out_idx])).append(", ");
      if ((out_idx + 1) % 5 == 0) {
        outputStr.append("\n");
      }
    }
    Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));


    StringBuilder cur_out = new StringBuilder(String.format("%.7f", emsPredLogits[0]));
    cur_out.append(" ");
    for (int i = 1; i < emsPredLogits.length; i++) {
      cur_out.append(String.format("%.7f", emsPredLogits[i])).append(" ");
    }
    cur_out.append("\n");
    output_str_to_save += cur_out;



    double averaged_latency = 3;
    //    double averaged_latency = total_latency;
    Log.v(TAG, String.format("Averaged time latency: %s", averaged_latency));

    //      String tflite_test_result_file = ModelHelper.TFLITE_TEST_RESULT_FILE;
    try {
      // write out the inference result
      String model_name = ModelHelper.EMS_MODEL_PATH.split("\\.")[0];
      File prob_txt = new File(PROB_TEXT_PATH, model_name + "_" + "myOutPut");
      Log.v(TAG, String.format("we are trying to create a file for result %s", prob_txt.toString()));
      prob_txt.createNewFile();
      FileWriter fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + "myOutPut");
      fw.write(output_str_to_save);
      fw.flush();
      fw.close();

      //write out the inference latency
      File latency_txt = new File(PROB_TEXT_PATH, model_name + "_" + "myOutPutLatency");
      Log.v(TAG, String.format("we are trying to create a file for latency %s", latency_txt.toString()));
      latency_txt.createNewFile();
      fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + "myOutPutLatency");
      Log.v(TAG, String.format("inference average latency %s", averaged_latency));
      fw.write(String.valueOf(averaged_latency));
      fw.flush();
      fw.close();


    } catch (IOException e) {
      e.printStackTrace();
    }
    Log.v(TAG, String.format("finish writing result to result file %s and latency file %s",
            "myOutPut", "myOutPutLatency"));
  }


  @SuppressLint("DefaultLocale")
  public synchronized void run_pp_test_for_fitted(){

    // TO_DO needs modification to 0
      String test_data_file = "fitted_desc_mmlc_test.tsv";
      String test_result_file = "fitted_desc_mmlc_test_result.txt";

      List<String> testData = new ArrayList<>();
      BufferedReader reader = null;
      int total_line_num = 0;
      try {
        reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(test_data_file)));
        reader.readLine();
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
      Log.i(TAG,String.format("total test data number %d in test file %s\n", total_line_num, test_data_file));



      String output_str_to_save = "";
//    for(; test_idx < testData.size(); test_idx++){
      float[][] emsPredLogits = new float[1][NUM_PRED_CLASSES];
      int[] inputIds = new int[MAX_SEQ_LEN];
      int[] inputMask = new int[MAX_SEQ_LEN];
      int[] segmentIds = new int[MAX_SEQ_LEN];
      int[][] inputs = new int[3][MAX_SEQ_LEN];

      Map<Integer, Object> output = new HashMap<>();
      long total_latency = 0;

//    for(int test_idx = 0; test_idx < testData.size(); test_idx++){
      int test_start_idx = 0;
      int test_end_idx = testData.size();
//      int test_end_idx = 10;
      for(int test_idx = test_start_idx; test_idx < test_end_idx; test_idx++){    // check the first test data

        if(test_idx % 1000 == 0){
          Log.v(TAG, String.format("we have processed %d test data prediction", test_idx));
        }

        String data_str = testData.get(test_idx);
        String[] sample_label = data_str.split("\t");
        String query = sample_label[0];
        String label_name = sample_label[1];
        EMSBertFeature emsBertFeature = featureConverter.convert(query);
        Integer label_id = fitted_label_map.get(label_name);
//        Integer label_id = label_map.get(label_name);

        if(test_idx - test_start_idx < 1){
          Log.i(TAG, String.format("We have converted sample %d, string: %s, label_name: %s, label_idx: %d",
                  test_idx, query, label_name, label_id));
          Log.v(TAG, "TFLite model: " + ModelHelper.EMS_MODEL_PATH + " running...");
          Log.v(TAG, "Convert Feature...");
          Log.v(TAG, "Set inputs...");
        }



        for (int j = 0; j < MAX_SEQ_LEN; j++) {
          inputIds[j] = emsBertFeature.inputIds[j];
          inputMask[j] = emsBertFeature.inputMask[j];
          segmentIds[j] = emsBertFeature.segmentIds[j];
        }

        if(test_idx - test_start_idx < 1){

          String inputIdsStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            inputIdsStr += inputIds[inp_idx] + ", ";
            if((inp_idx + 1) % 9 == 0) { inputIdsStr += "\n"; }
          }
          Log.v(TAG, String.format("inputIds: \n%s", inputIdsStr));

          String inputMaskStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            inputMaskStr += inputMask[inp_idx] + ", ";
            if((inp_idx + 1) % 23 == 0) { inputMaskStr += "\n"; }
          }
          Log.v(TAG, String.format("inputMask: \n%s", inputMaskStr));

          String segIdStr = "";
          for(int inp_idx = 0; inp_idx < MAX_SEQ_LEN; inp_idx++){
            segIdStr += segmentIds[inp_idx] + ", ";
            if((inp_idx + 1) % 23 == 0) { segIdStr += "\n"; }
          }
          Log.v(TAG, String.format("segmentIds: \n%s", segIdStr));
        }


        inputs[0] = inputIds;
//      inputs[1] = inputMask;
//      inputs[2] = segmentIds;
        inputs[2] = inputMask;
        inputs[1] = segmentIds;


        output.put(0, emsPredLogits);

        if(test_idx - test_start_idx < 1) {

          int input_count = tflite.getInputTensorCount();
          int output_count = tflite.getOutputTensorCount();
          // model make 2-class text classification bert
          int input_idx = tflite.getInputIndex("serving_default_input_word_ids:0");
          int mask_idx = tflite.getInputIndex("serving_default_input_mask:0");
          int segment_idx = tflite.getInputIndex("serving_default_input_type_ids:0");
          int prob_idx = tflite.getOutputIndex("StatefulPartitionedCall:0");
          Log.v(TAG, String.format("input tensor count: %d, output tensor count: %d\n",
                  input_count, output_count));
          Log.v(TAG, String.format("Inputs: input_ids idx: %d, input_mask idx: %d, segment_ids idx: %d\n",
                  input_idx, mask_idx, segment_idx));
          Log.v(TAG, String.format("Outputs: probabilities idx: %d\n", prob_idx));

          Log.v(TAG, "Run inference...");
        }

        long infer_start = System.currentTimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, output);
        long infer_latency = System.currentTimeMillis() - infer_start;
        total_latency += infer_latency;

        if(test_idx - test_start_idx < 1) {
          StringBuilder outputStr = new StringBuilder();
          for(int out_idx = 0; out_idx < emsPredLogits[0].length; out_idx++){
            outputStr.append(String.format("%.7f", emsPredLogits[0][out_idx])).append(", ");
            if((out_idx + 1) % 5 == 0) { outputStr.append("\n"); }
          }
          Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));
        }

        StringBuilder cur_out = new StringBuilder(String.format("%.7f",emsPredLogits[0][0]));
        cur_out.append(" ");
        for(int i = 1; i < emsPredLogits[0].length; i++){
          cur_out.append(String.format("%.7f", emsPredLogits[0][i])).append(" ");
        }
        cur_out.append("\n");
        output_str_to_save += cur_out;

      }

      double averaged_latency = (total_latency * 1.0 / (test_end_idx - test_start_idx));
//    double averaged_latency = total_latency;
      Log.v(TAG, String.format("Averaged time latency: %s", averaged_latency));

//      String tflite_test_result_file = ModelHelper.TFLITE_TEST_RESULT_FILE;
      try {
        // write out the inference result
        String model_name = ModelHelper.EMS_MODEL_PATH.split("\\.")[0];
        File prob_txt = new File(PROB_TEXT_PATH, model_name + "_" + test_result_file);
        Log.v(TAG, String.format("we are trying to create a file %s", prob_txt.toString()));
        prob_txt.createNewFile();
        FileWriter fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" + test_result_file);
        fw.write(output_str_to_save);
        fw.flush();
        fw.close();

        //write out the inference latency
        File latency_txt = new File(PROB_TEXT_PATH, model_name + "_" +"latency.txt");
        Log.v(TAG, String.format("we are trying to create a file %s", latency_txt.toString()));
        latency_txt.createNewFile();
        fw = new FileWriter(PROB_TEXT_PATH + model_name + "_" +"latency.txt");
        Log.v(TAG, String.format("inference average latency %s", averaged_latency));
        fw.write(String.valueOf(averaged_latency));
        fw.flush();
        fw.close();


      } catch (IOException e) {
        e.printStackTrace();
      }
      Log.v(TAG, String.format("finish writing result to file %s", test_result_file));

  }

  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   * logits.
   */
  @WorkerThread
  public synchronized List<QaAnswer> predict(String query, String content) {

//    run_pp_test_for_no_fitted();

//    run_pp_test_for_fitted();

//    run_pp_test_for_no_fitted_batch();

    run_pp_test_for_fitted_batch_am();


//    Log.v(TAG, "TFLite model: " + ModelHelper.MODEL_PATH + " running...");
//    Log.v(TAG, "Convert Feature...");
//    Feature feature = featureConverter.convert(query, content);
//
//    Log.v(TAG, "Set inputs...");
////    int[][] inputIds = new int[1][MAX_SEQ_LEN];
////    int[][] inputMask = new int[1][MAX_SEQ_LEN];
////    int[][] segmentIds = new int[1][MAX_SEQ_LEN];
//    float[][] emsPredLogits = new float[1][NUM_PRED_CLASSES];
//    int[] inputIds = new int[MAX_SEQ_LEN];
//    int[] inputMask = new int[MAX_SEQ_LEN];
//    int[] segmentIds = new int[MAX_SEQ_LEN];
////    float[] emsPredLogits = new float[NUM_PRED_CLASSES];
//
//    float[][] startLogits = new float[1][MAX_SEQ_LEN];
//    float[][] endLogits = new float[1][MAX_SEQ_LEN];
//
//    for (int j = 0; j < MAX_SEQ_LEN; j++) {
////      inputIds[0][j] = feature.inputIds[j];
////      inputMask[0][j] = feature.inputMask[j];
////      segmentIds[0][j] = feature.segmentIds[j];
//      inputIds[j] = feature.inputIds[j];
//      inputMask[j] = feature.inputMask[j];
//      segmentIds[j] = feature.segmentIds[j];
//    }
//
////    Object[] inputs = new Object[3];
//    int[][] inputs = new int[3][MAX_SEQ_LEN];
//    boolean useInputMetadata = false;
//    if (metadataExtractor != null && metadataExtractor.getInputTensorCount() == 3) {
//      // If metadata exists and the size of input tensors in metadata is 3, use metadata to treat
//      // the tensor order. Since the order of input tensors can be different for different models,
//      // set the inputs according to input tensor names.
//      useInputMetadata = true;
//      for (int i = 0; i < 3; i++) {
//        TensorMetadata inputMetadata = metadataExtractor.getInputTensorMetadata(i);
//
//        switch (inputMetadata.name()) {
//          case IDS_TENSOR_NAME:
//            inputs[i] = inputIds;
//            break;
//          case MASK_TENSOR_NAME:
//            inputs[i] = inputMask;
//            break;
//          case SEGMENT_IDS_TENSOR_NAME:
//            inputs[i] = segmentIds;
//            break;
//          default:
//            Log.e(TAG, "Input name in metadata doesn't match the default input tensor names.");
//            useInputMetadata = false;
//        }
//      }
//    }
//    if (!useInputMetadata) {
//      // If metadata doesn't exists or doesn't contain the info, fail back to a hard-coded order.
//      Log.v(TAG, "Use hard-coded order of input tensors.");
//      inputs[0] = inputIds;
//      inputs[1] = inputMask;
//      inputs[2] = segmentIds;
//    }
//
//    Map<Integer, Object> output = new HashMap<>();
//    // Hard-coded idx for output, maybe changed according to metadata below.
//    int endLogitsIdx = 0;
//    int startLogitsIdx = 1;
//    boolean useOutputMetadata = false;
//    if (metadataExtractor != null && metadataExtractor.getOutputTensorCount() == 2) {
//      // If metadata exists and the size of output tensors in metadata is 2, use metadata to treat
//      // the tensor order. Since the order of output tensors can be different for different models,
//      // set the indexs of the outputs according to output tensor names.
//      useOutputMetadata = true;
//      for (int i = 0; i < 2; i++) {
//        TensorMetadata outputMetadata = metadataExtractor.getOutputTensorMetadata(i);
//        switch (outputMetadata.name()) {
//          case END_LOGITS_TENSOR_NAME:
//            endLogitsIdx = i;
//            break;
//          case START_LOGITS_TENSOR_NAME:
//            startLogitsIdx = i;
//            break;
//          default:
//            Log.e(TAG, "Output name in metadata doesn't match the default output tensor names.");
//            useOutputMetadata = false;
//        }
//      }
//    }
//    if (!useOutputMetadata) {
//      Log.v(TAG, "Use hard-coded order of output tensors.");
//      endLogitsIdx = 0;
//      startLogitsIdx = 1;
//    }
////    output.put(endLogitsIdx, endLogits);
////    output.put(startLogitsIdx, startLogits);
//    output.put(0, emsPredLogits);
//
//    Log.v(TAG, "Run inference...");
////    tflite.runForMultipleInputsOutputs(inputs, output);
//    tflite.runForMultipleInputsOutputs(inputs, output);
////    Object emsOutput = emsPredLogits;
////    tflite.run(inputs, emsPredLogits);
//    int input_count = tflite.getInputTensorCount();
//    int output_count = tflite.getOutputTensorCount();
////    int input_idx = tflite.getInputIndex("input_ids_1:0");
////    int mask_idx = tflite.getInputIndex("input_mask_1:0");
////    int segment_idx = tflite.getInputIndex("segment_ids_1:0");
////    int prob_idx = tflite.getOutputIndex("loss/Sigmoid:0");
//
//    // model make 2-class text classification bert
//    int input_idx = tflite.getInputIndex("serving_default_input_word_ids:0");
//    int mask_idx = tflite.getInputIndex("serving_default_input_mask:0");
//    int segment_idx = tflite.getInputIndex("serving_default_input_type_ids:0");
//    int prob_idx = tflite.getOutputIndex("StatefulPartitionedCall:0");
//
//    Log.v(TAG, String.format("input tensor count: %d, output tensor count: %d\n",
//            input_count, output_count));
//    Log.v(TAG, String.format("Inputs: input_ids idx: %d, input_mask idx: %d, segment_ids idx: %d\n",
//            input_idx, mask_idx, segment_idx));
//    Log.v(TAG, String.format("Outputs: probabilities idx: %d\n", prob_idx));
////    Log.v(TAG, "Convert answers...");
    List<QaAnswer> answers = new ArrayList<>();
//    List<QaAnswer> answers = getBestAnswers(startLogits[0], endLogits[0], feature);

//    int maxIndex = argMax(emsPredLogits[0]);
//    Log.v(TAG, String.format("The output max index is: %d", maxIndex));
//    for(int idx = 0; idx < NUM_PRED_CLASSES; idx++){
//      Log.v(TAG, String.format("The probability for label %d is: %f", idx, emsPredLogits[0][idx]));
//    }

    Log.v(TAG, "Finish.");
    return answers;
  }

  private synchronized int argMax(float[] arr){
    int maxIdx = 0;
    for(int i = 0; i < arr.length; i++){
      if(arr[maxIdx] < arr[i]){
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  private List<Pair<Float, Integer>> getTopK(float[] arr, int k){

    PriorityQueue<Pair<Float, Integer>> pq =
            new PriorityQueue<Pair<Float, Integer>>(k, (p1, p2) -> (int) (p2.first - p1.first));
    for(int i = 0; i < arr.length; i++){
      Pair<Float, Integer> p = new Pair<>(arr[i], i);
      pq.add(p);
      if(pq.size() > k){
        pq.poll();
      }
    }
    List<Pair<Float, Integer>> topKList = new ArrayList<>(pq);
    Collections.reverse(topKList);
    return topKList;
  }


  /** Find the Best N answers & logits from the logits array and input feature. */
  private synchronized List<QaAnswer> getBestAnswers(
      float[] startLogits, float[] endLogits, Feature feature) {
    // Model uses the closed interval [start, end] for indices.
    int[] startIndexes = getBestIndex(startLogits);
    int[] endIndexes = getBestIndex(endLogits);

    List<QaAnswer.Pos> origResults = new ArrayList<>();
    for (int start : startIndexes) {
      for (int end : endIndexes) {
        if (!feature.tokenToOrigMap.containsKey(start + OUTPUT_OFFSET)) {
          continue;
        }
        if (!feature.tokenToOrigMap.containsKey(end + OUTPUT_OFFSET)) {
          continue;
        }
        if (end < start) {
          continue;
        }
        int length = end - start + 1;
        if (length > MAX_ANS_LEN) {
          continue;
        }
        origResults.add(new QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]));
      }
    }

    Collections.sort(origResults);

    List<QaAnswer> answers = new ArrayList<>();
    for (int i = 0; i < origResults.size(); i++) {
      if (i >= PREDICT_ANS_NUM) {
        break;
      }

      String convertedText;
      if (origResults.get(i).start > 0) {
        convertedText = convertBack(feature, origResults.get(i).start, origResults.get(i).end);
      } else {
        convertedText = "";
      }
      QaAnswer ans = new QaAnswer(convertedText, origResults.get(i));
      answers.add(ans);
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  @WorkerThread
  private synchronized int[] getBestIndex(float[] logits) {
    List<QaAnswer.Pos> tmpList = new ArrayList<>();
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
      tmpList.add(new QaAnswer.Pos(i, i, logits[i]));
    }
    Collections.sort(tmpList);

    int[] indexes = new int[PREDICT_ANS_NUM];
    for (int i = 0; i < PREDICT_ANS_NUM; i++) {
      indexes[i] = tmpList.get(i).start;
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  @WorkerThread
  private static String convertBack(Feature feature, int start, int end) {
     // Shifted index is: index of logits + offset.
    int shiftedStart = start + OUTPUT_OFFSET;
    int shiftedEnd = end + OUTPUT_OFFSET;
    int startIndex = feature.tokenToOrigMap.get(shiftedStart);
    int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
    // end + 1 for the closed interval.
    String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
    return ans;
  }
}
