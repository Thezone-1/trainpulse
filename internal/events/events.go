package events

import (
	"encoding/json"
	"io"
	"time"

	"github.com/somoprovo/trainpulse/internal/model"
)

type Event struct {
	Time        time.Time         `json:"time"`
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Severity    string            `json:"severity,omitempty"`
	Message     string            `json:"message"`
	Tags        map[string]string `json:"tags,omitempty"`
	Evidence    []string          `json:"evidence,omitempty"`
	Confidence  float64           `json:"confidence,omitempty"`
	Actions     []string          `json:"actions,omitempty"`
	HealthScore float64           `json:"health_score"`
}

func FromSnapshot(snap model.Snapshot) []Event {
	events := make([]Event, 0, len(snap.Signals)+len(snap.Diagnoses))
	tags := trainingTags(snap.Telemetry.Training)
	for _, signal := range snap.Signals {
		events = append(events, Event{
			Time:        signal.Timestamp,
			Type:        "signal",
			Name:        signal.Name,
			Severity:    string(signal.Severity),
			Message:     signal.Description,
			Tags:        tags,
			Evidence:    signal.Evidence,
			HealthScore: snap.Health,
		})
	}
	for _, diagnosis := range snap.Diagnoses {
		events = append(events, Event{
			Time:        snap.Timestamp,
			Type:        "diagnosis",
			Name:        diagnosis.RootCause,
			Message:     diagnosis.Explanation,
			Tags:        tags,
			Confidence:  diagnosis.Confidence,
			Actions:     diagnosis.Actions,
			HealthScore: snap.Health,
		})
	}
	return events
}

func WriteJSON(w io.Writer, snap model.Snapshot) error {
	return json.NewEncoder(w).Encode(map[string]any{
		"events": FromSnapshot(snap),
	})
}

func WriteNDJSON(w io.Writer, snap model.Snapshot) error {
	enc := json.NewEncoder(w)
	for _, event := range FromSnapshot(snap) {
		if err := enc.Encode(event); err != nil {
			return err
		}
	}
	return nil
}

func trainingTags(tr *model.TrainingSample) map[string]string {
	if tr == nil {
		return nil
	}
	return map[string]string{
		"workload_kind": tr.WorkloadKind,
		"model_family":  tr.ModelFamily,
		"model_name":    tr.ModelName,
		"framework":     tr.Framework,
		"precision":     tr.Precision,
	}
}
