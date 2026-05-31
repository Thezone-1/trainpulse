package config

import (
	"encoding/json"
	"flag"
	"os"
	"time"
)

type Config struct {
	Addr             string        `json:"addr"`
	Interval         time.Duration `json:"-"`
	IntervalText     string        `json:"interval,omitempty"`
	Mode             string        `json:"mode"`
	TrainingSocket   string        `json:"training_socket,omitempty"`
	HistorySize      int           `json:"history_size"`
	ConfigPath       string        `json:"-"`
	LogLevel         string        `json:"log_level"`
	LogFormat        string        `json:"log_format"`
	MetricsNamespace string        `json:"metrics_namespace"`
}

func FromFlags(args []string) (Config, string, error) {
	command := "daemon"
	if len(args) > 0 && len(args[0]) > 0 && args[0][0] != '-' {
		command = args[0]
		args = args[1:]
	}
	configPath := findConfigPath(args)

	cfg := Default()
	if configPath != "" {
		loaded, err := LoadFile(configPath)
		if err != nil {
			return Config{}, "", err
		}
		cfg = loaded
		cfg.ConfigPath = configPath
	}

	fs := flag.NewFlagSet("trainpulse", flag.ContinueOnError)
	fs.StringVar(&cfg.Addr, "addr", cfg.Addr, "HTTP listen address for daemon stats")
	fs.DurationVar(&cfg.Interval, "interval", cfg.Interval, "telemetry collection interval")
	fs.StringVar(&cfg.Mode, "mode", cfg.Mode, "collector mode: auto, nvidia-smi, sim")
	fs.StringVar(&cfg.TrainingSocket, "training-socket", cfg.TrainingSocket, "optional unix datagram socket for training runtime metrics")
	fs.IntVar(&cfg.HistorySize, "history", cfg.HistorySize, "rolling sample history size")
	fs.StringVar(&cfg.ConfigPath, "config", cfg.ConfigPath, "path to JSON config file")
	fs.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel, "log level: debug, info, warn, error")
	fs.StringVar(&cfg.LogFormat, "log-format", cfg.LogFormat, "log format: json or text")
	fs.StringVar(&cfg.MetricsNamespace, "metrics-namespace", cfg.MetricsNamespace, "metrics name prefix")
	if err := fs.Parse(args); err != nil {
		return Config{}, "", err
	}
	if fs.NArg() > 0 {
		command = fs.Arg(0)
	}
	cfg.IntervalText = cfg.Interval.String()
	return cfg, command, nil
}

func findConfigPath(args []string) string {
	for i, arg := range args {
		if arg == "-config" && i+1 < len(args) {
			return args[i+1]
		}
		if len(arg) > len("-config=") && arg[:len("-config=")] == "-config=" {
			return arg[len("-config="):]
		}
	}
	return ""
}

func Default() Config {
	return Config{
		Addr:             "127.0.0.1:9876",
		Interval:         time.Second,
		IntervalText:     time.Second.String(),
		Mode:             "auto",
		HistorySize:      120,
		LogLevel:         "info",
		LogFormat:        "json",
		MetricsNamespace: "trainpulse",
	}
}

func LoadFile(path string) (Config, error) {
	cfg := Default()
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, err
	}
	if cfg.IntervalText != "" {
		d, err := time.ParseDuration(cfg.IntervalText)
		if err != nil {
			return Config{}, err
		}
		cfg.Interval = d
	}
	if cfg.HistorySize == 0 {
		cfg.HistorySize = Default().HistorySize
	}
	if cfg.MetricsNamespace == "" {
		cfg.MetricsNamespace = Default().MetricsNamespace
	}
	return cfg, nil
}
