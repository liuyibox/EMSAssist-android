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

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.textfield.TextInputEditText;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Locale;
import org.tensorflow.lite.examples.emsassist.R;
import org.tensorflow.lite.examples.emsassist.ml.QaAnswer;
import org.tensorflow.lite.examples.emsassist.ml.QaClient;

import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

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
    private Interpreter tfLiteASR;

    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;
    private TextView predictionView;

    // emsBERT declarations
    private TextInputEditText questionEditText;
    private TextView contentTextView;
    private TextToSpeech textToSpeech;

    private String content;
    private Handler handler;
    private QaClient qaClient;

    static private final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1;
    static private final int PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 1;
    //

    private String textToFeed;

    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();

    private final static String TAG = "TfLiteASR";
    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;
    private final static String[] WAV_FILENAMES = {"audio_clip_1.wav", "audio_clip_2.wav", "audio_clip_3.wav", "audio_clip_4.wav"};
    private final static String TFLITE_FILE = "CONFORMER.tflite";
    private final static String predictionFileName = "fitted_label_names.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.v(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        int writeStoragePermissionCheck = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if(writeStoragePermissionCheck != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
            return;
        }

        int readStoragePermissionCheck = ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.READ_EXTERNAL_STORAGE);
        if(readStoragePermissionCheck != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);
            return;
        }

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
        predictionView = findViewById(R.id.pred_view);
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
                    tfLiteASR = new Interpreter(tfLiteModel, tfLiteOptions);
                    Log.i(TAG, "After instance");

                    tfLiteASR.resizeInput(0, new int[] {audioFeatureValues.length});
                    tfLiteASR.runForMultipleInputsOutputs(inputArray, outputMap);

                    int outputSize = tfLiteASR.getOutputTensor(0).shape()[0];
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
                    textToFeed = "Transcribed Text: \n" + finalResult + "\n";
//                    tfLiteASR.setCancelled(true);
                    tfLiteASR.close();
                    Log.i(TAG, "asr result: " + textToFeed);

                    final String answers = qaClient.predict(textToFeed, content);
                    String display = "Predicted top 5 protocols :\n" + answers;
                    Log.i(TAG, "Got result from predict function on myResult");
                    resultTextview.setText(textToFeed);
                    predictionView.setText(display);
//                    predictionView.setText(top5Prediction);
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

    private String buildString(float[] answers) {
        StringBuilder newStr = new StringBuilder();
        for (int i = 0; i < answers.length; i++) {
            newStr.append(String.format("%.7f", answers[i])).append(", ");
            if ((i + 1) % 5 == 0) {
                newStr.append("\n");
            }
        }
        return newStr.toString();
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

    public static String convertStreamToString(InputStream is) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder sb = new StringBuilder();
        String line = null;
        while ((line = reader.readLine()) != null) {
            sb.append(line).append("\n");
        }
        reader.close();
        return sb.toString();
    }

    public static String getStringFromFile (String filePath) throws Exception {
        File fl = new File(filePath);
        FileInputStream fin = new FileInputStream(fl);
        String ret = convertStreamToString(fin);
        //Make sure you close all streams.
        fin.close();
        return ret;
    }

    public static String matchPrediction (String filePath) throws Exception {
        File fl = new File(filePath);
        FileInputStream fin = new FileInputStream(fl);
        String ret = convertStreamToString(fin);
        //Make sure you close all streams.
        fin.close();
        return ret;
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
}