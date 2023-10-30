package sysu.sdcs.sensordatacollector.domain.image;

import android.util.Log;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;
import com.otaliastudios.cameraview.size.Size;

import sysu.sdcs.sensordatacollector.FileUtil;
import sysu.sdcs.sensordatacollector.SensorData;
import sysu.sdcs.sensordatacollector.domain.adapter.HeartRate;
import sysu.sdcs.sensordatacollector.domain.adapter.HeartRateAdapter;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.pipeline.Pipeline;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.pipeline.PpgProcessingPipeline;

import java.lang.ref.WeakReference;
import java.lang.System;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static sysu.sdcs.sensordatacollector.constant.GlobalConstants.BATCH_SIZE;
import static sysu.sdcs.sensordatacollector.domain.image.PixelProcessor.yuvToRedSum;
import static sysu.sdcs.sensordatacollector.root.PpgApplication.executor;

public class PpgFrameProcessor implements FrameProcessor {

    /**
     * The number of frames collected for a single batch
     */
    private int frameCounter;
    private String fname = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS").format(new Date());;

    /**
     * Y axis: the amount of color Red
     */
    public int[] signal;

    /**
     * X axis: current time in milliseconds when a frame is captured
     */
    public long[] time;

    /**
     * Reference to the UI element displaying heart rate
     */
    private WeakReference<TextView> viewWeakReference;

    public PpgFrameProcessor(WeakReference<TextView> viewWeakReference) {
        this.viewWeakReference = viewWeakReference;
        resetParameters();
    }
    @Override
    @WorkerThread
    public void process(@NonNull Frame frame){
        Size size = frame.getSize();
        int redSum = yuvToRedSum(frame.getData(), size.getWidth(), size.getHeight());
        signal[frameCounter] = redSum;
        time[frameCounter] = frame.getTime();
        if (++frameCounter == BATCH_SIZE) {
            data2string();
            calculateHeartRate();
            resetParameters();
        }
    }
    private String data2string(){
        String data = "";
        SimpleDateFormat dateformat = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS");
        for(int i = 0 ; i < BATCH_SIZE ; i++){
            String ppg = "" + signal[i];
            //String timestamp = "" + time[i];
            String timestamp = dateformat.format(new Date(time[i]));
            String one_detail = "" + ppg + "," + timestamp + "\n" ;
            data = data + one_detail;
        }
//        clear();
//        Log.d("data length", "" + data.length());
//        Log.d("fname", fname);
        FileUtil.saveSensorData("PPG_" + fname + ".csv", data);
        return data;
    }
    private void calculateHeartRate() {
        long startTime = time[0];
        int[] y = IntStream.of(signal).toArray();
        long[] x = LongStream.of(time).map(t -> t - startTime).toArray();

        CompletableFuture.supplyAsync(() -> processSignal(y), executor())
                .thenApply(signal -> toHeartRate(signal, x))
                .thenAccept(this::updateUI);
    }


    private int[] processSignal(int[] unprocessedSignal) {
        Pipeline pipeline = PpgProcessingPipeline.pipeline();
        return pipeline.execute(unprocessedSignal);
    }

    private String toHeartRate(int[] processedSignal, long[] timestamps) {
        HeartRate adapter = new HeartRateAdapter(processedSignal, timestamps);
        return adapter.convertToHeartRate();
    }

    private void updateUI(String heartRate) {
        TextView textView = viewWeakReference.get();
        Log.d("heart", ""+heartRate);
        textView.post(() -> textView.setText(heartRate));
    }

    private void resetParameters() {
        frameCounter = 0;
        signal = new int[BATCH_SIZE];
        time = new long[BATCH_SIZE];
    }
}
