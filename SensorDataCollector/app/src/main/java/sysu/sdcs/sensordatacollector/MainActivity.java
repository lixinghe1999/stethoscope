package sysu.sdcs.sensordatacollector;

import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.controls.Flash;
import sysu.sdcs.sensordatacollector.constant.GlobalConstants;
import sysu.sdcs.sensordatacollector.domain.image.PpgFrameProcessor;
import sysu.sdcs.sensordatacollector.service.MeasurementPhase;
import sysu.sdcs.sensordatacollector.service.MotionMonitoringService;

import android.content.Intent;
import android.graphics.ImageFormat;
import android.media.MediaRecorder;
import android.os.Environment;
import android.util.Log;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorManager;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
//import android.support.v7.app.AppCompatActivity;

import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {
    private CameraView camera;
    private TextView heartRate;
    private PpgFrameProcessor frameProcessor;
    private MediaRecorder recorder = null;
    private static String directory = null;
    private static final String LOG_TAG = "AudioRecordTest";
    boolean mStartRecording = false;

    private static final int REQ_CODE_PERMISSION_EXTERNAL_STORAGE = 0x1111;
    private static final int REQ_CODE_PERMISSION_SENSOR = 0x2222;

    private SensorManager sensorManager;
    private SensorListener sensorListener;
    private Sensor accelerometerSensor;
    private Sensor gyroscopeSensor;
    private Sensor magneticSensor;
    private Sensor orientationSensor;
    private Sensor stepCounterSensor;
    private Sensor stepDetectSensor;

    private Button btn_control;
    private EditText edt_path;
    private TextView tv_state;
    private TextView tv_record;

    private ScheduledFuture future;
    private String file_name = "";
    private String cap_records = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init();
        btn_control.setOnClickListener(btn_listener);
        //initCamera();
        //initButton();
    }

    public void init(){
        btn_control = findViewById(R.id.btn_control);
        edt_path = findViewById(R.id.edt_pathID);
        tv_state = findViewById(R.id.state);
        tv_record = findViewById(R.id.record);

        sensorListener = new SensorListener();
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magneticSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        orientationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION);
        stepCounterSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_COUNTER);
        stepDetectSensor = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR);
        gyroscopeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        directory = Environment.getExternalStorageDirectory().getAbsolutePath()+ "/SensorData/";
        permissionCheck();

    }
    private void onRecord(boolean start) {
        if (start) {
            startRecording();
        } else {
            stopRecording();
        }
    }
    private void startRecording() {
        recorder = new MediaRecorder();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setOutputFile(directory + "MIC_" + file_name + ".wav");
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setAudioSamplingRate(44100);
        recorder.setAudioEncodingBitRate(16*44100);
        try {
            recorder.prepare();
        } catch (IOException e) {
            Log.e(LOG_TAG, "prepare() failed");
            throw new RuntimeException(e);
        }

        recorder.start();
    }

    private void stopRecording() {
        recorder.stop();
        recorder.release();
        recorder = null;
    }
    public void permissionCheck(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED){
            //申请WRITE_EXTERNAL_STORAGE权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQ_CODE_PERMISSION_EXTERNAL_STORAGE);
        }
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS)
                != PackageManager.PERMISSION_GRANTED){
            //申请BODY_SENSOR权限
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.BODY_SENSORS},
                    REQ_CODE_PERMISSION_SENSOR);
        }
    }
    public String getCurrentTime(){
        return new SimpleDateFormat("yyyyMMdd_HHmmss_SSS").format(new Date());
    }
//    private void initCamera() {
//        camera = findViewById(R.id.view_camera);
//        camera.setVisibility(View.INVISIBLE);
//        camera.setLifecycleOwner(this);
//        camera.setFrameProcessingFormat(ImageFormat.YUV_420_888);
//        heartRate = findViewById(R.id.text_heart_rate);
//    }

    private void startMeasurement() {
        camera.setFlash(Flash.TORCH);
        PpgFrameProcessor frameprocessor = new PpgFrameProcessor(new WeakReference<>(heartRate));
        camera.addFrameProcessor(frameprocessor);
        motionMonitoring(MeasurementPhase.START);
    }
    private void stopMeasurement() {
        camera.setFlash(Flash.OFF);
        camera.clearFrameProcessors();
        heartRate.setText(R.string.label_empty);
        motionMonitoring(MeasurementPhase.STOP);
    }

    private void motionMonitoring(MeasurementPhase phase) {
        Intent intent = new Intent(GlobalConstants.MEASUREMENT_PHASE_CHANGE);
        intent.putExtra(GlobalConstants.MEASUREMENT_PHASE_CHANGE, phase.name());
        sendBroadcast(intent);
    }
//    private void initButton() {
//        findViewById(R.id.btn_start_measurement).setOnTouchListener((v, event) -> {
//            if (event.getAction() == MotionEvent.ACTION_DOWN) {
//                startMeasurement();
//                Log.d("action_down", "11");
//            } else if (event.getAction() == MotionEvent.ACTION_UP) {
//                stopMeasurement();
//                Log.d("action_up", "22");
//            }
//            return false;
//        });
//    }
        private View.OnClickListener btn_listener = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
//            if(edt_path.getText().toString().equals("") ||
//                    edt_path.getText().toString() == null) {
//                Toast.makeText(MainActivity.this, "path ID 不能为空", Toast.LENGTH_SHORT).show();
//            }
            if(btn_control.getText().toString().equals("start")){
                // startMeasurement();

                // file_name = edt_path.getText().toString() + "-" + getCurrentTime();
                file_name = getCurrentTime();
                onRecord(true);

                if(!sensorManager.registerListener(sensorListener, accelerometerSensor, SensorManager.SENSOR_DELAY_FASTEST ))
                    Toast.makeText(MainActivity.this, "加速度传感器不可用", Toast.LENGTH_SHORT).show();

//                if(!sensorManager.registerListener(sensorListener, magneticSensor, SensorManager.SENSOR_DELAY_FASTEST))
//                    Toast.makeText(MainActivity.this, "磁场传感器不可用", Toast.LENGTH_SHORT).show();
//
//                if(!sensorManager.registerListener(sensorListener, orientationSensor, SensorManager.SENSOR_DELAY_FASTEST))
//                    Toast.makeText(MainActivity.this, "方向传感器不可用", Toast.LENGTH_SHORT).show();
//
//                if(!sensorManager.registerListener(sensorListener, stepCounterSensor, SensorManager.SENSOR_DELAY_FASTEST))
//                    Toast.makeText(MainActivity.this, "记步传感器不可用", Toast.LENGTH_SHORT).show();
//
//                if(!sensorManager.registerListener(sensorListener, stepDetectSensor, SensorManager.SENSOR_DELAY_FASTEST))
//                    Toast.makeText(MainActivity.this, "记步传感器不可用", Toast.LENGTH_SHORT).show();
//
//                if(!sensorManager.registerListener(sensorListener, gyroscopeSensor, SensorManager.SENSOR_DELAY_FASTEST))
//                    Toast.makeText(MainActivity.this, "陀螺仪不可用", Toast.LENGTH_SHORT).show();


                InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                if (imm != null) {
                    imm.hideSoftInputFromWindow(getWindow().getDecorView().getWindowToken(), 0);
                }

                tv_state.setText("传感器数据正在采集中\n" + "当前采集路径为：" + edt_path.getText().toString());
                btn_control.setText("stop");
                FileUtil.saveSensorData("IMU_" + file_name + ".csv", SensorData.getFileHead());
                ScheduledExecutorService service = Executors.newScheduledThreadPool(5);
                future = service.scheduleAtFixedRate(new DataSaveTask(file_name), 5, 5, TimeUnit.SECONDS);
            }
            else{
                // stopMeasurement();

                onRecord(false);

                future.cancel(true);
                sensorManager.unregisterListener(sensorListener);
                if(FileUtil.saveSensorData("IMU_" + file_name + ".csv", SensorData.getAccData())){
                    cap_records += file_name + "\n";
                    tv_record.setText(cap_records);
                    tv_state.setText("");
                    Toast.makeText(MainActivity.this, "传感器数据保存成功", Toast.LENGTH_SHORT).show();
                }
                else
                    Toast.makeText(MainActivity.this, "传感器数据保存失败", Toast.LENGTH_SHORT).show();
                SensorData.clear();
                btn_control.setText("start");
                tv_state.setText("点击按钮开始采集\n");
            }

        }
    };
    @Override
    public void onStop() {
        super.onStop();
        if (recorder != null) {
            recorder.release();
            recorder = null;
        }
        camera.destroy();
    }
    //权限申请
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQ_CODE_PERMISSION_EXTERNAL_STORAGE: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // User agree the permission
                } else {
                    // User disagree the permission
                    Toast.makeText(MainActivity.this, "请打开存储权限", Toast.LENGTH_LONG).show();
                }
            }
            case REQ_CODE_PERMISSION_SENSOR: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // User agree the permission
                }
                else {
                    // User disagree the permission
                    Toast.makeText(this, "请打开传感器权限", Toast.LENGTH_LONG).show();
                }
            }
            break;
        }
    }

}