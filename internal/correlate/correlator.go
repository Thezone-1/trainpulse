package correlate

import "github.com/somoprovo/trainpulse/internal/model"

type Correlator struct{}

func New() *Correlator { return &Correlator{} }

func (c *Correlator) Correlate(signals []model.Signal) []model.Signal {
	return signals
}
