// Package knowledge holds TrainPulse's diagnostic knowledge base: the mapping
// from detected signals to root-cause diagnoses, confidence, explanations, and
// remediation actions.
//
// The knowledge base ships as data (default.json, embedded into the binary) so
// the default behavior is identical out of the box, while teams can extend or
// override any diagnosis through the config file without recompiling. This is
// the data-driven replacement for what used to be a hard-coded map in the
// diagnostics package.
package knowledge

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

//go:embed default.json
var defaultBytes []byte

type file struct {
	Diagnoses []config.DiagnosisRule `json:"diagnoses"`
}

// Base is an ordered, evaluated diagnostic knowledge base.
type Base struct {
	rules []config.DiagnosisRule
}

// Default returns the built-in knowledge base embedded in the binary.
func Default() *Base {
	rules, err := parse(defaultBytes)
	if err != nil {
		// The embedded default is authored in-repo and covered by tests, so a
		// parse failure here is a build-time bug, not a runtime condition.
		panic(fmt.Sprintf("knowledge: invalid embedded default.json: %v", err))
	}
	return &Base{rules: rules}
}

// New returns the built-in knowledge base merged with the given overrides.
// An override whose RootCause matches a built-in entry replaces it in place
// (preserving order); an override with a new RootCause is appended.
func New(overrides []config.DiagnosisRule) *Base {
	base := Default()
	return base.With(overrides)
}

// With returns a copy of the base with overrides merged in.
func (b *Base) With(overrides []config.DiagnosisRule) *Base {
	merged := make([]config.DiagnosisRule, len(b.rules))
	copy(merged, b.rules)
	index := map[string]int{}
	for i, r := range merged {
		index[r.RootCause] = i
	}
	for _, o := range overrides {
		if o.RootCause == "" {
			continue
		}
		if i, ok := index[o.RootCause]; ok {
			merged[i] = o
			continue
		}
		index[o.RootCause] = len(merged)
		merged = append(merged, o)
	}
	return &Base{rules: merged}
}

// Rules returns the evaluated diagnostic rules in order.
func (b *Base) Rules() []config.DiagnosisRule {
	out := make([]config.DiagnosisRule, len(b.rules))
	copy(out, b.rules)
	return out
}

// Infer walks the knowledge base in order and emits a diagnosis for every rule
// whose signal condition is satisfied by the active signals.
func (b *Base) Infer(signals []model.Signal) []model.Diagnosis {
	present := make(map[string]bool, len(signals))
	for _, s := range signals {
		present[s.Name] = true
	}
	var out []model.Diagnosis
	for _, r := range b.rules {
		if !ruleMatches(r, present) {
			continue
		}
		out = append(out, model.Diagnosis{
			RootCause:   r.RootCause,
			Confidence:  r.Confidence,
			Explanation: r.Explanation,
			Actions:     r.Actions,
		})
	}
	return out
}

func ruleMatches(r config.DiagnosisRule, present map[string]bool) bool {
	if len(r.WhenSignals) == 0 {
		return false
	}
	if strings.EqualFold(r.Match, "all") {
		for _, s := range r.WhenSignals {
			if !present[s] {
				return false
			}
		}
		return true
	}
	for _, s := range r.WhenSignals {
		if present[s] {
			return true
		}
	}
	return false
}

func parse(b []byte) ([]config.DiagnosisRule, error) {
	var f file
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, err
	}
	return f.Diagnoses, nil
}
