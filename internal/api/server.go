package api

import (
	"encoding/json"
	"net/http"

	"github.com/somoprovo/trainpulse/internal/agent"
)

type Server struct {
	agent *agent.Agent
}

func New(a *agent.Agent) *Server {
	return &Server{agent: a}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.healthz)
	mux.HandleFunc("/v1/snapshot", s.snapshot)
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
