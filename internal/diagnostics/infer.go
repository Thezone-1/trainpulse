package diagnostics

import (
	"github.com/somoprovo/trainpulse/internal/knowledge"
	"github.com/somoprovo/trainpulse/internal/model"
)

// Inferer turns detected signals into root-cause diagnoses using a diagnostic
// knowledge base. The knowledge base is data-driven (see internal/knowledge),
// so the signal-to-diagnosis mapping is no longer hard-coded here.
type Inferer struct {
	kb *knowledge.Base
}

// New returns an Inferer backed by the built-in knowledge base.
func New() *Inferer {
	return &Inferer{kb: knowledge.Default()}
}

// NewWithBase returns an Inferer backed by the given knowledge base, allowing
// callers to supply config-driven overrides via knowledge.New.
func NewWithBase(kb *knowledge.Base) *Inferer {
	if kb == nil {
		kb = knowledge.Default()
	}
	return &Inferer{kb: kb}
}

func (i *Inferer) Infer(signals []model.Signal) []model.Diagnosis {
	return i.kb.Infer(signals)
}
