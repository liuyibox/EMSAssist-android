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


import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.metadata.MetadataExtractor;

/** Interface to load TfLite model and provide predictions. */
public class QaClient<myMap> implements AutoCloseable {
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
  private HashMap<Integer, String> fitted_label_map_rev;
  private static final int K = 5;
  private String PROB_TEXT_PATH = Environment.getExternalStorageDirectory().getPath() +
          "/distressnet/MStorm/EMSFiles/";

  @WorkerThread
  public synchronized void loadModel() {
    try {
//      ByteBuffer buffer = ModelHelper.loadModelFile(context);
      testData = ModelHelper.loadTestData(context);
      label_map = ModelHelper.loadLabels(context);
      //fitted_label_map = ModelHelper.loadFittedLabels(context);
      fitted_label_map_rev = ModelHelper.loadFittedLabelsRev(context);

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
  public synchronized String run_pp_test_for_fitted_am(String que_ry){
//    que_ry = "mental status changes mental status changes septicemia pulmonary edema septicemia";
    Log.i(TAG, "Called the run_pp_test with query : " + que_ry);

    float[][] emsPredLogits = new float[1][NUM_PRED_CLASSES];
//    int[] inputIds = new int[MAX_SEQ_LEN];
//    int[] inputMask = new int[MAX_SEQ_LEN];
//    int[] segmentIds = new int[MAX_SEQ_LEN];
    int[][] inputs = new int[3][MAX_SEQ_LEN];

    String output_str_to_save = "";
    Map<Integer, Object> output = new HashMap<>();
    long total_latency = 0;

    String query = que_ry;

//    tflite.run

    int input_idx = tflite.getInputIndex("serving_default_input_word_ids:0");
    int mask_idx = tflite.getInputIndex("serving_default_input_mask:0");
    int segment_idx = tflite.getInputIndex("serving_default_input_type_ids:0");

    Log.i(TAG, "Called the EMSBertFeature");
    EMSBertFeature emsBertFeature = featureConverter.convert(query);
    int[] inputIds = emsBertFeature.inputIds;
    int[] inputMask = emsBertFeature.inputMask;
    int[] segmentIds = emsBertFeature.segmentIds;

    inputs[0] = inputIds;
    inputs[2] = inputMask;
    inputs[1] = segmentIds;

    Log.i(TAG, "inputIds " + input_idx + " " + inputIds.length +  ":" + inputIds);
    Log.i(TAG, "inputMask " + mask_idx + " " + inputMask.length +  ":" +  inputMask);
    Log.i(TAG, "segmentIds " + segment_idx + " " + segmentIds.length +  ":" +  segmentIds);

    output.put(0, emsPredLogits);

//    long infer_start = System.currentTimeMillis();
    Log.i(TAG, "Called the tflite model for inference...");
    tflite.runForMultipleInputsOutputs(inputs, output);
//    tflite.run(inputs, emsPredLogits);
//    long infer_latency = System.currentTimeMillis() - infer_start;
//    total_latency += infer_latency;
    Log.i(TAG, "After inference");

    StringBuilder outputStr = new StringBuilder();
    float [] predFloat = new float[emsPredLogits[0].length];
    for (int out_idx = 0; out_idx < emsPredLogits[0].length; out_idx++) {

      outputStr.append(String.format("%.7f", emsPredLogits[0][out_idx])).append(", ");
      predFloat[out_idx] = emsPredLogits[0][out_idx];
      if ((out_idx + 1) % 5 == 0) {
        outputStr.append("\n");
      }
    }

    Log.v(TAG, String.format("Output Predication: \n%s", outputStr.toString()));
    Log.i(TAG, "Fitted label map : \n" + fitted_label_map_rev.toString());

    /** Create an unsorted Map with Index **/
    Map unsortedMap = getMapFromArrayRev(emsPredLogits[0]);
    Log.i(TAG, "Unsorted Map" + unsortedMap);


    /** Sort the prediction array **/
    Arrays.sort(emsPredLogits[0]);

    /** Reverse the sorted prediction array **/
    for(int i = 0; i < emsPredLogits[0].length / 2; i++)
    {
      float temp = emsPredLogits[0][i];
      emsPredLogits[0][i] = emsPredLogits[0][emsPredLogits[0].length - i - 1];
      emsPredLogits[0][emsPredLogits[0].length - i - 1] = temp;
    }

    /** Create a sorted Map with Index **/
    Map sortedMap = getMapFromArray(emsPredLogits[0]);
    Log.i(TAG,"Sorted Map: " + sortedMap);

    /** Create a top five prediction table **/

    //Map returnMap = new HashMap<Float,String>();

    int i_map = 0;
    StringBuilder outputStr2 = new StringBuilder();
    for(int i = 0; i<5; i++) {
      i_map =  (int) unsortedMap.get(sortedMap.get(i));
      Log.i(TAG, "i_map value : " + i_map);
      //returnMap.put(sortedMap.get(i), fitted_label_map_rev.get(i_map));
      outputStr2.append(fitted_label_map_rev.get(i_map));
      outputStr2.append("\t\t");
      outputStr2.append(sortedMap.get(i));
      outputStr2.append("\n");
    }
    //Log.i(TAG, "return map: " + returnMap);
    Log.i(TAG, outputStr2.toString());
    //return predFloat;
    return outputStr2.toString();
  }


  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   * logits.
   * @return
   */
  @WorkerThread
  public synchronized String predict(String query, String Content) {
    Log.i(TAG, "Just called the predict function");

    return run_pp_test_for_fitted_am(query);
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

  private Map<Integer,Float> getMapFromArray(float[] arr){
    HashMap<Integer,Float> map = new HashMap<>();
    for (int i = 0; i < arr.length; i++) {
      map.put(i, arr[i]);
    }
    return map;
  }

  private Map<Float,Integer> getMapFromArrayRev(float[] arr){
    HashMap<Float,Integer> map = new HashMap<>();
    for (int i = 0; i < arr.length; i++) {
      map.put(arr[i],i);
    }
    return map;
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
