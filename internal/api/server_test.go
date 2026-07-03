package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/somoprovo/trainpulse/internal/agent"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/model"
)

func TestTrainingIngestion(t *testing.T) {
	a := agent.New(config.Config{Interval: time.Second, HistorySize: 4}, noTrainingCollector{})
	server := New(a)
	body, err := json.Marshal(model.TrainingSample{
		WorkloadKind:    "llm_pretraining",
		ModelFamily:     "llama",
		ModelName:       "llama-test",
		StepTimeMS:      200,
		TokensPerSec:    30000,
		MFU:             0.20,
		TokenizerWaitMS: 75,
		AllReduceWaitMS: 80,
		AvgSeqLen:       600,
		MaxSeqLen:       2048,
		Ranks: []model.RankSample{
			{Rank: 0, StepTimeMS: 150},
			{Rank: 1, StepTimeMS: 230},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/training", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected accepted, got %d", rec.Code)
	}
	if err := a.Tick(req.Context()); err != nil {
		t.Fatal(err)
	}
	snap := a.Snapshot()
	if snap.Telemetry.Training == nil || snap.Telemetry.Training.ModelName != "llama-test" {
		t.Fatalf("expected posted training sample, got %+v", snap.Telemetry.Training)
	}
}

func TestMetricsEndpoints(t *testing.T) {
	a := agent.New(config.Config{Interval: time.Second, HistorySize: 4}, noTrainingCollector{})
	a.UpdateTraining(model.TrainingSample{TokensPerSec: 1000, MFU: 0.10})
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	server := New(a, config.Config{MetricsNamespace: "trainpulse_test"})
	for _, path := range []string{"/metrics", "/v1/metrics", "/v1/events", "/v1/events?format=ndjson", "/v1/integrations"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		server.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s expected ok, got %d", path, rec.Code)
		}
		if rec.Body.Len() == 0 {
			t.Fatalf("%s returned empty body", path)
		}
	}
}

func TestFrameworkIngestion(t *testing.T) {
	a := agent.New(config.Config{Interval: time.Second, HistorySize: 4}, noTrainingCollector{})
	server := New(a)
	body := []byte(`{"framework":"deepspeed","model":"llama-ds","train_tokens_per_second":12345,"model_flops_utilization":0.31,"global_batch_size":64}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/framework?name=deepspeed", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	server.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected accepted, got %d body=%s", rec.Code, rec.Body.String())
	}
	if err := a.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	tr := a.Snapshot().Telemetry.Training
	if tr == nil || tr.ModelName != "llama-ds" || tr.TokensPerSec != 12345 {
		t.Fatalf("expected normalized framework sample, got %+v", tr)
	}
}

func TestAuthTokenEnforced(t *testing.T) {
	a := agent.New(config.Config{Interval: time.Second, HistorySize: 4}, noTrainingCollector{})
	server := New(a, config.Config{AuthToken: "secret-token"})
	handler := server.Handler()

	// /healthz stays open for liveness probes.
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("healthz should not require auth, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/snapshot", nil)
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without token, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/snapshot", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 with token, got %d", rec.Code)
	}
}

func TestVersionEndpoint(t *testing.T) {
	a := agent.New(config.Config{Interval: time.Second, HistorySize: 4}, noTrainingCollector{})
	server := New(a)
	req := httptest.NewRequest(http.MethodGet, "/v1/version", nil)
	rec := httptest.NewRecorder()
	server.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["version"] == "" {
		t.Fatal("version endpoint returned empty version")
	}
}

type noTrainingCollector struct{}

func (noTrainingCollector) Name() string { return "no-training" }

func (noTrainingCollector) Collect(_ context.Context) (model.TelemetryFrame, error) {
	now := time.Now()
	return model.TelemetryFrame{
		Timestamp: now,
		GPUs: []model.GPUSample{
			{Index: 0, Name: "test-gpu", Utilization: 80, MemoryUsed: 1000, MemoryTotal: 2000, Timestamp: now},
		},
	}, nil
}
