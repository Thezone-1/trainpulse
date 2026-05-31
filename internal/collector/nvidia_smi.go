package collector

import (
	"bytes"
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type NvidiaSMICollector struct {
	path string
}

func NewNvidiaSMICollector() *NvidiaSMICollector {
	return &NvidiaSMICollector{path: "nvidia-smi"}
}

func (n *NvidiaSMICollector) Name() string { return "nvidia-smi" }

func (n *NvidiaSMICollector) Collect(ctx context.Context) (model.TelemetryFrame, error) {
	args := []string{
		"--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
		"--format=csv,noheader,nounits",
	}
	cmd := exec.CommandContext(ctx, n.path, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return model.TelemetryFrame{}, ErrUnavailable
	}
	now := time.Now()
	gpus := parseNvidiaSMI(stdout.String(), now)
	return model.TelemetryFrame{Timestamp: now, GPUs: gpus}, nil
}

func parseNvidiaSMI(out string, ts time.Time) []model.GPUSample {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	gpus := make([]model.GPUSample, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.Split(line, ",")
		for i := range parts {
			parts[i] = strings.TrimSpace(parts[i])
		}
		if len(parts) < 9 {
			continue
		}
		index, _ := strconv.Atoi(parts[0])
		util, _ := strconv.ParseFloat(parts[3], 64)
		memUsed, _ := strconv.ParseUint(parts[4], 10, 64)
		memTotal, _ := strconv.ParseUint(parts[5], 10, 64)
		temp, _ := strconv.ParseFloat(parts[6], 64)
		power, _ := strconv.ParseFloat(parts[7], 64)
		clock, _ := strconv.ParseFloat(parts[8], 64)
		gpus = append(gpus, model.GPUSample{
			Index:       index,
			Name:        parts[1],
			UUID:        parts[2],
			Utilization: util,
			MemoryUsed:  memUsed,
			MemoryTotal: memTotal,
			Temperature: temp,
			PowerWatts:  power,
			SMClockMHz:  clock,
			Timestamp:   ts,
		})
	}
	return gpus
}
