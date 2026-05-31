package collector

import (
	"context"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type Composite struct {
	gpu  Collector
	host Collector
}

func NewComposite(gpu Collector, host Collector) *Composite {
	return &Composite{gpu: gpu, host: host}
}

func (c *Composite) Name() string {
	return "composite"
}

func (c *Composite) Collect(ctx context.Context) (model.TelemetryFrame, error) {
	frame := model.TelemetryFrame{Timestamp: time.Now()}
	if c.host != nil {
		host, err := c.host.Collect(ctx)
		if err == nil {
			frame.Host = host.Host
		}
	}
	if c.gpu != nil {
		gpu, err := c.gpu.Collect(ctx)
		if err != nil {
			return frame, err
		}
		frame.GPUs = gpu.GPUs
	}
	return frame, nil
}
