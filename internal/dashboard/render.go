package dashboard

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

func Render(w io.Writer, snap model.Snapshot) {
	clear(w)
	fmt.Fprintf(w, "TrainPulse  %s  samples=%d\n", snap.Timestamp.Format(time.RFC3339), snap.SampleCount)
	fmt.Fprintf(w, "Health: %.0f/100  Status: %s\n\n", snap.Health, strings.ToUpper(string(snap.Status)))
	for _, gpu := range snap.Telemetry.GPUs {
		fmt.Fprintf(w, "GPU %d %-18s util=%5.1f%% mem=%5d/%5dMB temp=%4.1fC power=%5.1fW clock=%4.0fMHz\n",
			gpu.Index, truncate(gpu.Name, 18), gpu.Utilization, gpu.MemoryUsed, gpu.MemoryTotal, gpu.Temperature, gpu.PowerWatts, gpu.SMClockMHz)
	}
	if snap.Telemetry.Training != nil {
		tr := snap.Telemetry.Training
		fmt.Fprintf(w, "\nTraining step=%d step_time=%.1fms throughput=%.1f data_wait=%.1fms sync_wait=%.1fms batch=%d\n",
			tr.GlobalStep, tr.StepTimeMS, tr.Throughput, tr.DataWaitMS, tr.SyncWaitMS, tr.BatchSize)
	}
	if len(snap.Signals) == 0 {
		fmt.Fprintln(w, "\nSignals: none")
	} else {
		fmt.Fprintln(w, "\nSignals:")
		for _, signal := range snap.Signals {
			fmt.Fprintf(w, "  [%s] %s: %s\n", signal.Severity, signal.Name, signal.Description)
			for _, evidence := range signal.Evidence {
				fmt.Fprintf(w, "       %s\n", evidence)
			}
		}
	}
	if len(snap.Diagnoses) > 0 {
		fmt.Fprintln(w, "\nLikely causes:")
		for _, diagnosis := range snap.Diagnoses {
			fmt.Fprintf(w, "  %s confidence=%.0f%%\n", diagnosis.RootCause, diagnosis.Confidence*100)
			fmt.Fprintf(w, "    %s\n", diagnosis.Explanation)
		}
	}
}

func clear(w io.Writer) {
	fmt.Fprint(w, "\033[H\033[2J")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
