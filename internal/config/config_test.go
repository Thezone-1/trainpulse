package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trainpulse.json")
	if err := os.WriteFile(path, []byte(`{"addr":"0.0.0.0:9999","interval":"250ms","mode":"sim","history_size":12,"log_level":"debug","metrics_namespace":"tp","rules":[{"name":"low_tokens","field":"training.tokens_per_sec","operator":"lt","value":1000}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Addr != "0.0.0.0:9999" || cfg.Interval != 250*time.Millisecond || cfg.Mode != "sim" || cfg.HistorySize != 12 {
		t.Fatalf("unexpected config: %+v", cfg)
	}
	if len(cfg.Rules) != 1 || cfg.Rules[0].Name != "low_tokens" {
		t.Fatalf("expected rule from config, got %+v", cfg.Rules)
	}
}

func TestFromFlagsUsesConfigFileDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trainpulse.json")
	if err := os.WriteFile(path, []byte(`{"addr":"0.0.0.0:9999","interval":"250ms","mode":"sim"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, command, err := FromFlags([]string{"daemon", "-config", path})
	if err != nil {
		t.Fatal(err)
	}
	if command != "daemon" {
		t.Fatalf("unexpected command %q", command)
	}
	if cfg.Addr != "0.0.0.0:9999" || cfg.Interval != 250*time.Millisecond || cfg.Mode != "sim" {
		t.Fatalf("config file defaults were not preserved: %+v", cfg)
	}
}
