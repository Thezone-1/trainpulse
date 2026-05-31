package collector

import (
	"context"
	"errors"

	"github.com/somoprovo/trainpulse/internal/model"
)

var ErrUnavailable = errors.New("collector unavailable")

type Collector interface {
	Collect(context.Context) (model.TelemetryFrame, error)
	Name() string
}
