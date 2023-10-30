package sysu.sdcs.sensordatacollector.domain.adapter;

/**
 * Convert the time series of pixel values and timestamps to heart rate.
 */
public interface HeartRate {
    String convertToHeartRate();
}
