package api

import (
	"encoding/json"
	"net/http"

	"github.com/somoprovo/trainpulse/internal/agent"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/framework"
	"github.com/somoprovo/trainpulse/internal/metrics"
	"github.com/somoprovo/trainpulse/internal/model"
)

type Server struct {
	agent      *agent.Agent
	namespace  string
	frameworks *framework.Registry
}

func New(a *agent.Agent, cfg ...config.Config) *Server {
	namespace := "trainpulse"
	if len(cfg) > 0 && cfg[0].MetricsNamespace != "" {
		namespace = cfg[0].MetricsNamespace
	}
	return &Server{agent: a, namespace: namespace, frameworks: framework.DefaultRegistry()}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.healthz)
	mux.HandleFunc("/metrics", s.prometheus)
	mux.HandleFunc("/v1/metrics", s.metricsJSON)
	mux.HandleFunc("/v1/snapshot", s.snapshot)
	mux.HandleFunc("/v1/training", s.training)
	mux.HandleFunc("/v1/framework", s.framework)
	return mux
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

func (s *Server) snapshot(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.agent.Snapshot())
}

func (s *Server) prometheus(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_ = metrics.WritePrometheus(w, s.agent.Snapshot(), s.namespace)
}

func (s *Server) metricsJSON(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = metrics.EncodeJSON(w, s.agent.Snapshot(), s.namespace)
}

func (s *Server) training(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	var sample model.TrainingSample
	if err := json.NewDecoder(r.Body).Decode(&sample); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.agent.UpdateTraining(sample)
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write([]byte("accepted\n"))
}

func (s *Server) framework(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	var payload map[string]any
	dec := json.NewDecoder(r.Body)
	dec.UseNumber()
	if err := dec.Decode(&payload); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	name := r.URL.Query().Get("name")
	if name == "" {
		if raw, ok := payload["framework"].(string); ok {
			name = raw
		}
	}
	sample, err := s.frameworks.Normalize(name, payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.agent.UpdateTraining(sample)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(sample)
}
