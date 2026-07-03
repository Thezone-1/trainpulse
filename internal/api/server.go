package api

import (
	"crypto/subtle"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/somoprovo/trainpulse/internal/agent"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/events"
	"github.com/somoprovo/trainpulse/internal/framework"
	"github.com/somoprovo/trainpulse/internal/integrations"
	"github.com/somoprovo/trainpulse/internal/metrics"
	"github.com/somoprovo/trainpulse/internal/model"
	"github.com/somoprovo/trainpulse/internal/version"
)

// maxBodyBytes bounds POST payloads so a misbehaving client cannot exhaust
// daemon memory. Training samples are a few KB; 1 MiB leaves ample headroom.
const maxBodyBytes = 1 << 20

type Server struct {
	agent      *agent.Agent
	namespace  string
	authToken  string
	frameworks *framework.Registry
}

func New(a *agent.Agent, cfg ...config.Config) *Server {
	namespace := "trainpulse"
	token := ""
	if len(cfg) > 0 {
		if cfg[0].MetricsNamespace != "" {
			namespace = cfg[0].MetricsNamespace
		}
		token = cfg[0].AuthToken
	}
	return &Server{agent: a, namespace: namespace, authToken: token, frameworks: framework.DefaultRegistry()}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.healthz)
	mux.HandleFunc("/metrics", s.auth(s.prometheus))
	mux.HandleFunc("/v1/metrics", s.auth(s.metricsJSON))
	mux.HandleFunc("/v1/events", s.auth(s.events))
	mux.HandleFunc("/v1/integrations", s.auth(s.integrations))
	mux.HandleFunc("/v1/snapshot", s.auth(s.snapshot))
	mux.HandleFunc("/v1/version", s.version)
	mux.HandleFunc("/v1/training", s.auth(s.training))
	mux.HandleFunc("/v1/framework", s.auth(s.framework))
	return mux
}

// auth enforces the optional bearer token on everything except liveness and
// version probes. With no token configured it is a no-op, preserving the
// zero-config localhost experience.
func (s *Server) auth(next http.HandlerFunc) http.HandlerFunc {
	if s.authToken == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if subtle.ConstantTimeCompare([]byte(got), []byte(s.authToken)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="trainpulse"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

func (s *Server) version(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"name":    "trainpulse",
		"version": version.Version,
		"commit":  version.Commit,
	})
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

func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	if r.URL.Query().Get("format") == "ndjson" {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_ = events.WriteNDJSON(w, s.agent.Snapshot())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = events.WriteJSON(w, s.agent.Snapshot())
}

func (s *Server) integrations(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"integrations": integrations.Catalog(),
	})
}

func (s *Server) training(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
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
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
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
