//go:build !linux

package collector

import (
	"context"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type HostCollector struct{}

func NewHostCollector() *HostCollector { return &HostCollector{} }

func (h *HostCollector) Name() string { return "host-stub" }

func (h *HostCollector) Collect(context.Context) (model.TelemetryFrame, error) {
	now := time.Now()
	return model.TelemetryFrame{
		Timestamp: now,
		Host: model.HostSample{
			CPUUtilization: 0,
			MemoryUsedMB:   0,
			MemoryTotalMB:  0,
			Timestamp:      now,
		},
	}, nil
}
