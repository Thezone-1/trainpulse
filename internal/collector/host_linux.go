//go:build linux

package collector

import (
	"bufio"
	"context"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type HostCollector struct{}

func NewHostCollector() *HostCollector { return &HostCollector{} }

func (h *HostCollector) Name() string { return "linux-host" }

func (h *HostCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	total, available := readMemInfo()
	load := readLoad1()
	used := uint64(0)
	if total > available {
		used = total - available
	}
	now := time.Now()
	return model.TelemetryFrame{
		Timestamp: now,
		Host: model.HostSample{
			MemoryUsedMB:  used / 1024,
			MemoryTotalMB: total / 1024,
			Load1:         load,
			Timestamp:     now,
		},
	}, nil
}

func readMemInfo() (totalKB, availableKB uint64) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		value, _ := strconv.ParseUint(fields[1], 10, 64)
		switch strings.TrimSuffix(fields[0], ":") {
		case "MemTotal":
			totalKB = value
		case "MemAvailable":
			availableKB = value
		}
	}
	return totalKB, availableKB
}

func readLoad1() float64 {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(b))
	if len(fields) == 0 {
		return 0
	}
	v, _ := strconv.ParseFloat(fields[0], 64)
	return v
}
