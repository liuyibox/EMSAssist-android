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
package org.tensorflow.lite.examples.emsassist.ui;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.textfield.TextInputEditText;
import java.util.List;
import java.util.Locale;
import org.tensorflow.lite.examples.emsassist.R;
import org.tensorflow.lite.examples.emsassist.ml.QaAnswer;
import org.tensorflow.lite.examples.emsassist.ml.QaClient;

import androidx.annotation.RequiresApi;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;

import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;


public class AsrActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;

    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;

    // emsBERT declarations
    private TextInputEditText questionEditText;
    private TextView contentTextView;
    private TextToSpeech textToSpeech;

    private String content;
    private Handler handler;
    private QaClient qaClient;
    //

    private String textToFeed;

    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();

    private final static String TAG = "TfLiteASR";
    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;
    private final static String[] WAV_FILENAMES = {"audio_clip_1.wav", "audio_clip_2.wav", "audio_clip_3.wav", "audio_clip_4.wav"};
    private final static String TFLITE_FILE = "CONFORMER.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.v(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        JLibrosa jLibrosa = new JLibrosa();

        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        ArrayAdapter<String>adapter = new ArrayAdapter<String>(AsrActivity.this,
                android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);

        playAudioButton = findViewById(R.id.play);
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try (AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                    mediaPlayer.reset();
                    mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                    mediaPlayer.prepare();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                mediaPlayer.start();
            }
        });

        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                try {
                    float audioFeatureValues[] = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);

                    Object[] inputArray = {audioFeatureValues};
                    IntBuffer outputBuffer = IntBuffer.allocate(2000);

                    Map<Integer, Object> outputMap = new HashMap<>();
                    outputMap.put(0, outputBuffer);

                    tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
                    Interpreter.Options tfLiteOptions = new Interpreter.Options();
                    Log.i(TAG, "Before instance");
                    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                    Log.i(TAG, "After instance");

                    tfLite.resizeInput(0, new int[] {audioFeatureValues.length});


                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

                    int outputSize = tfLite.getOutputTensor(0).shape()[0];
                    int[] outputArray = new int[outputSize];
                    outputBuffer.rewind();
                    outputBuffer.get(outputArray);
                    StringBuilder finalResult = new StringBuilder();
                    for (int i=0; i < outputSize; i++) {
                        char c = (char) outputArray[i];
                        if (outputArray[i] != 0) {
                            finalResult.append((char) outputArray[i]);
                        }
                    }
                    textToFeed = finalResult.toString();
                    Log.i(TAG, "asr result: " + textToFeed);
                    String myResult = qaClient.predict(textToFeed);
                    Log.i(TAG, "Got result from predict function on myResult");
                    resultTextview.setText(myResult);
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }
        });

        // Setup QA client to and background thread to run inference.
        HandlerThread handlerThread = new HandlerThread("QAClient");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        qaClient = new QaClient(this);

    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String copyWavFileToCache(String wavFilename) {
        File destinationFile = new File(getCacheDir() + wavFilename);
        if (!destinationFile.exists()) {
            try {
                InputStream inputStream = getAssets().open(wavFilename);
                int inputStreamSize = inputStream.available();
                byte[] buffer = new byte[inputStreamSize];
                inputStream.read(buffer);
                inputStream.close();

                FileOutputStream fileOutputStream = new FileOutputStream(destinationFile);
                fileOutputStream.write(buffer);
                fileOutputStream.close();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());
            }
        }

        return getCacheDir() + wavFilename;
    }

    @Override
    protected void onStart() {
        Log.v(TAG, "onStart");
        super.onStart();
        handler.post(
                () -> {
                    qaClient.loadModel();
                });

        textToSpeech =
                new TextToSpeech(
                        this,
                        status -> {
                            if (status == TextToSpeech.SUCCESS) {
                                textToSpeech.setLanguage(Locale.US);
                            } else {
                                textToSpeech = null;
                            }
                        });
    }

    @Override
    protected void onStop() {
        Log.v(TAG, "onStop");
        super.onStop();
        handler.post(() -> qaClient.unload());

        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
    }

//    private void answerQuestion(String question) {
//        question = question.trim();
//        if (question.isEmpty()) {
//            questionEditText.setText(question);
//            return;
//        }
//
//        // Append question mark '?' if not ended with '?'.
//        // This aligns with question format that trains the model.
//        if (!question.endsWith("?")) {
//            question += '?';
//        }
//        final String questionToAsk = question;
//        questionEditText.setText(questionToAsk);
//
//        // Delete all pending tasks.
//        handler.removeCallbacksAndMessages(null);
//
//        // Hide keyboard and dismiss focus on text edit.
//        InputMethodManager imm =
//                (InputMethodManager) getSystemService(AppCompatActivity.INPUT_METHOD_SERVICE);
//        imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
//        View focusView = getCurrentFocus();
//        if (focusView != null) {
//            focusView.clearFocus();
//        }
//
//        // Reset content text view
//        contentTextView.setText(content);
//
//        questionAnswered = false;
//
//        Snackbar runningSnackbar =
//                Snackbar.make(contentTextView, "Looking up answer...", Integer.MAX_VALUE);
//        runningSnackbar.show();
//
//        // Run TF Lite model to get the answer.
//        handler.post(
//                () -> {
//                    long beforeTime = System.currentTimeMillis();
//                    final List<QaAnswer> answers = qaClient.predict(questionToAsk, content);
//                    long afterTime = System.currentTimeMillis();
//                    double totalSeconds = (afterTime - beforeTime) / 1000.0;
//
//                    if (!answers.isEmpty()) {
//                        // Get the top answer
//                        QaAnswer topAnswer = answers.get(0);
//                        // Show the answer.
//                        runOnUiThread(
//                                () -> {
//                                    runningSnackbar.dismiss();
//                                    presentAnswer(topAnswer);
//
//                                    String displayMessage = "Top answer was successfully highlighted.";
//                                    if (DISPLAY_RUNNING_TIME) {
//                                        displayMessage = String.format("%s %.3fs.", displayMessage, totalSeconds);
//                                    }
//                                    Snackbar.make(contentTextView, displayMessage, Snackbar.LENGTH_LONG).show();
//                                    questionAnswered = true;
//                                });
//                    }else {
//                        Log.v(TAG, "QA inference returns an empty list!");
//                    }
//                });
//    }
//
//    private void presentAnswer(QaAnswer answer) {
//        // Highlight answer.
//        Spannable spanText = new SpannableString(content);
//        int offset = content.indexOf(answer.text, 0);
//        if (offset >= 0) {
//            spanText.setSpan(
//                    new BackgroundColorSpan(getColor(R.color.tfe_qa_color_highlight)),
//                    offset,
//                    offset + answer.text.length(),
//                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
//        }
//        contentTextView.setText(spanText);
//
//        // Use TTS to speak out the answer.
//        if (textToSpeech != null) {
//            textToSpeech.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text);
//        }
//    }

}

/** Activity for doing Q&A on a specific dataset */
//public class AsrActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener{
//
//
//  private static final String DATASET_POSITION_KEY = "DATASET_POSITION";
//  private static final String TAG = "QaActivity";
//  private static final boolean DISPLAY_RUNNING_TIME = false;
//
//  private TextInputEditText questionEditText;
//  private TextView contentTextView;
//  private TextToSpeech textToSpeech;
//
//  private boolean questionAnswered = false;
//  private String content;
//  private Handler handler;
//  private QaClient qaClient;
//
//  static private final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1;
//  static private final int PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 1;
//
//  public static Intent newInstance(Context context, int datasetPosition) {
//    Intent intent = new Intent(context, AsrActivity.class);
//    intent.putExtra(DATASET_POSITION_KEY, datasetPosition);
//    return intent;
//  }
//
//  @Override
//  protected void onCreate(Bundle savedInstanceState) {
//    Log.v(TAG, "onCreate");
//    super.onCreate(savedInstanceState);
//    setContentView(R.layout.tfe_qa_activity_qa);
//
//
//      int writeStoragePermissionCheck = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
//      if(writeStoragePermissionCheck != PackageManager.PERMISSION_GRANTED){
//          ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
//          return;
//      }
//
//      int readStoragePermissionCheck = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE);
//      if(readStoragePermissionCheck != PackageManager.PERMISSION_GRANTED){
//          ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);
//          return;
//      }
//
//    // Get content of the selected dataset.
//    int datasetPosition = getIntent().getIntExtra(DATASET_POSITION_KEY, -1);
//    LoadDatasetClient datasetClient = new LoadDatasetClient(this);
//
//    // Show the dataset title.
//    TextView titleText = findViewById(R.id.title_text);
//    titleText.setText(datasetClient.getTitles()[datasetPosition]);
//
//    // Show the text content of the selected dataset.
//    content = datasetClient.getContent(datasetPosition);
//    contentTextView = findViewById(R.id.content_text);
//    contentTextView.setText(content);
//    contentTextView.setMovementMethod(new ScrollingMovementMethod());
//
//
//    // Setup ask button.
//    ImageButton askButton = findViewById(R.id.ask_button);
//    askButton.setOnClickListener(
//        view -> answerQuestion(questionEditText.getText().toString()));
//
//    // Setup text edit where users can input their question.
//    questionEditText = findViewById(R.id.question_edit_text);
//    questionEditText.setOnFocusChangeListener(
//        (view, hasFocus) -> {
//          // If we already answer current question, clear the question so that user can input a new
//          // one.
//          if (hasFocus && questionAnswered) {
//            questionEditText.setText(null);
//          }
//        });
//    questionEditText.addTextChangedListener(
//        new TextWatcher() {
//          @Override
//          public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
//
//          @Override
//          public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {
//            // Only allow clicking Ask button if there is a question.
//            boolean shouldAskButtonActive = !charSequence.toString().isEmpty();
//            askButton.setClickable(shouldAskButtonActive);
//            askButton.setImageResource(
//                shouldAskButtonActive ? R.drawable.ic_ask_active : R.drawable.ic_ask_inactive);
//          }
//
//          @Override
//          public void afterTextChanged(Editable editable) {}
//        });
//    questionEditText.setOnKeyListener(
//        (v, keyCode, event) -> {
//          if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_ENTER) {
//            answerQuestion(questionEditText.getText().toString());
//          }
//          return false;
//        });
//
//    // Setup QA client to and background thread to run inference.
//    HandlerThread handlerThread = new HandlerThread("QAClient");
//    handlerThread.start();
//    handler = new Handler(handlerThread.getLooper());
//    qaClient = new QaClient(this);
//  }
//
//  @Override
//  protected void onStart() {
//    Log.v(TAG, "onStart");
//    super.onStart();
//    handler.post(
//        () -> {
//          qaClient.loadModel();
//        });
//
//    textToSpeech =
//        new TextToSpeech(
//            this,
//            status -> {
//              if (status == TextToSpeech.SUCCESS) {
//                textToSpeech.setLanguage(Locale.US);
//              } else {
//                textToSpeech = null;
//              }
//            });
//  }
//
//  @Override
//  protected void onStop() {
//    Log.v(TAG, "onStop");
//    super.onStop();
//    handler.post(() -> qaClient.unload());
//
//    if (textToSpeech != null) {
//      textToSpeech.stop();
//      textToSpeech.shutdown();
//    }
//  }
//
//  private void answerQuestion(String question) {
//    question = question.trim();
//    if (question.isEmpty()) {
//      questionEditText.setText(question);
//      return;
//    }
//
//    // Append question mark '?' if not ended with '?'.
//    // This aligns with question format that trains the model.
//    if (!question.endsWith("?")) {
//      question += '?';
//    }
//    final String questionToAsk = question;
//    questionEditText.setText(questionToAsk);
//
//    // Delete all pending tasks.
//    handler.removeCallbacksAndMessages(null);
//
//    // Hide keyboard and dismiss focus on text edit.
//    InputMethodManager imm =
//        (InputMethodManager) getSystemService(AppCompatActivity.INPUT_METHOD_SERVICE);
//    imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
//    View focusView = getCurrentFocus();
//    if (focusView != null) {
//      focusView.clearFocus();
//    }
//
//    // Reset content text view
//    contentTextView.setText(content);
//
//    questionAnswered = false;
//
//    Snackbar runningSnackbar =
//        Snackbar.make(contentTextView, "Looking up answer...", Integer.MAX_VALUE);
//    runningSnackbar.show();
//
//    // Run TF Lite model to get the answer.
//    handler.post(
//        () -> {
//          long beforeTime = System.currentTimeMillis();
//          final List<QaAnswer> answers = qaClient.predict(questionToAsk, content);
//          long afterTime = System.currentTimeMillis();
//          double totalSeconds = (afterTime - beforeTime) / 1000.0;
//
//          if (!answers.isEmpty()) {
//            // Get the top answer
//            QaAnswer topAnswer = answers.get(0);
//            // Show the answer.
//            runOnUiThread(
//                () -> {
//                  runningSnackbar.dismiss();
//                  presentAnswer(topAnswer);
//
//                  String displayMessage = "Top answer was successfully highlighted.";
//                  if (DISPLAY_RUNNING_TIME) {
//                    displayMessage = String.format("%s %.3fs.", displayMessage, totalSeconds);
//                  }
//                  Snackbar.make(contentTextView, displayMessage, Snackbar.LENGTH_LONG).show();
//                  questionAnswered = true;
//                });
//          }else {
//              Log.v(TAG, "QA inference returns an empty list!");
//          }
//        });
//  }
//
//  private void presentAnswer(QaAnswer answer) {
//    // Highlight answer.
//    Spannable spanText = new SpannableString(content);
//    int offset = content.indexOf(answer.text, 0);
//    if (offset >= 0) {
//      spanText.setSpan(
//          new BackgroundColorSpan(getColor(R.color.tfe_qa_color_highlight)),
//          offset,
//          offset + answer.text.length(),
//          Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
//    }
//    contentTextView.setText(spanText);
//
//    // Use TTS to speak out the answer.
//    if (textToSpeech != null) {
//      textToSpeech.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text);
//    }
//  }
//}
