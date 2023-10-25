package com.uni.ppg.domain.image;

import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;
import com.otaliastudios.cameraview.size.Size;
import com.uni.ppg.domain.adapter.HeartRate;
import com.uni.ppg.domain.adapter.HeartRateAdapter;
import com.uni.ppg.domain.signalprocessing.pipeline.Pipeline;
import com.uni.ppg.domain.signalprocessing.pipeline.PpgProcessingPipeline;

import java.lang.ref.WeakReference;
import java.util.concurrent.CompletableFuture;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static com.uni.ppg.constant.GlobalConstants.BATCH_SIZE;
import static com.uni.ppg.domain.image.PixelProcessor.yuvToRedSum;
import static com.uni.ppg.root.PpgApplication.executor;

public class PpgFrameProcessor implements FrameProcessor {

    /**
     * The number of frames collected for a single batch
     */
    private int frameCounter;

    /**
     * Y axis: the amount of color Red
     */
    private int[] signal;

    /**
     * X axis: current time in milliseconds when a frame is captured
     */
    private long[] time;

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
    public void process(@NonNull Frame frame) {
        Size size = frame.getSize();
        int redSum = yuvToRedSum(frame.getData(), size.getWidth(), size.getHeight());
        signal[frameCounter] = redSum;
        time[frameCounter] = frame.getTime();

        if (++frameCounter == BATCH_SIZE) {
            calculateHeartRate();
            resetParameters();
        }
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
        textView.post(() -> textView.setText(heartRate));
    }

    private void resetParameters() {
        frameCounter = 0;
        signal = new int[BATCH_SIZE];
        time = new long[BATCH_SIZE];
    }
}
