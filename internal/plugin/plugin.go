package plugin

import (
	"github.com/somoprovo/trainpulse/internal/anomaly"
	"github.com/somoprovo/trainpulse/internal/collector"
	"github.com/somoprovo/trainpulse/internal/framework"
)

type Plugin interface {
	Name() string
	Collectors() []collector.Collector
	Detectors() []anomaly.Detector
	FrameworkAdapters() []framework.Adapter
}

type Static struct {
	NameValue     string
	CollectorList []collector.Collector
	DetectorList  []anomaly.Detector
	AdapterList   []framework.Adapter
}

func (s Static) Name() string { return s.NameValue }

func (s Static) Collectors() []collector.Collector { return s.CollectorList }

func (s Static) Detectors() []anomaly.Detector { return s.DetectorList }

func (s Static) FrameworkAdapters() []framework.Adapter { return s.AdapterList }

type Registry struct {
	plugins []Plugin
}

func NewRegistry() *Registry {
	return &Registry{}
}

func (r *Registry) Register(plugin Plugin) {
	r.plugins = append(r.plugins, plugin)
}

func (r *Registry) Detectors() []anomaly.Detector {
	var out []anomaly.Detector
	for _, plugin := range r.plugins {
		out = append(out, plugin.Detectors()...)
	}
	return out
}
