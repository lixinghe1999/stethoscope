package sysu.sdcs.sensordatacollector.domain.signalprocessing.pipeline;

import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.Derivation;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.MaximaCalculator;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.Preprocessor;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.ResultValidator;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.RollingAverage;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.filter.GaussianBlur;
import sysu.sdcs.sensordatacollector.domain.signalprocessing.steps.filter.LowPassFilter;

public class PpgProcessingPipeline {

    public static Pipeline pipeline() {
        return new Pipeline(new Preprocessor())
                .pipe(new RollingAverage())
                .pipe(new LowPassFilter(30))
                .pipe(new GaussianBlur())
                .pipe(new Derivation())
                .pipe(new MaximaCalculator())
                .pipe(new ResultValidator());
    }
}
