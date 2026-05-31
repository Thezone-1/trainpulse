package health

import "github.com/somoprovo/trainpulse/internal/model"

type Scorer struct{}

func New() *Scorer { return &Scorer{} }

func (s *Scorer) Score(signals []model.Signal) (float64, model.Severity) {
	score := 100.0
	for _, signal := range signals {
		score -= signal.ScoreImpact
	}
	if score < 0 {
		score = 0
	}
	status := model.SeverityInfo
	switch {
	case score < 55:
		status = model.SeverityCritical
	case score < 80:
		status = model.SeverityWarning
	}
	return score, status
}
