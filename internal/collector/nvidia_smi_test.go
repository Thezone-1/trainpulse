package collector

import (
	"testing"
	"time"
)

func TestParseNvidiaSMI(t *testing.T) {
	out := "0, NVIDIA H100, GPU-abc, 93, 20100, 81559, 72, 421.5, 1410\n"
	gpus := parseNvidiaSMI(out, time.Unix(1, 0))
	if len(gpus) != 1 {
		t.Fatalf("expected 1 gpu, got %d", len(gpus))
	}
	if gpus[0].Index != 0 || gpus[0].Utilization != 93 || gpus[0].MemoryUsed != 20100 {
		t.Fatalf("unexpected gpu sample: %+v", gpus[0])
	}
}
