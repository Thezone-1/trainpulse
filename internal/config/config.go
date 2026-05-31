package config

import (
	"flag"
	"time"
)

type Config struct {
	Addr           string
	Interval       time.Duration
	Mode           string
	TrainingSocket string
	HistorySize    int
}

func FromFlags(args []string) (Config, string, error) {
	command := "daemon"
	if len(args) > 0 && len(args[0]) > 0 && args[0][0] != '-' {
		command = args[0]
		args = args[1:]
	}
	fs := flag.NewFlagSet("trainpulse", flag.ContinueOnError)
	cfg := Config{}
	fs.StringVar(&cfg.Addr, "addr", "127.0.0.1:9876", "HTTP listen address for daemon stats")
	fs.DurationVar(&cfg.Interval, "interval", time.Second, "telemetry collection interval")
	fs.StringVar(&cfg.Mode, "mode", "auto", "collector mode: auto, nvidia-smi, sim")
	fs.StringVar(&cfg.TrainingSocket, "training-socket", "", "optional unix datagram socket for training runtime metrics")
	fs.IntVar(&cfg.HistorySize, "history", 120, "rolling sample history size")
	if err := fs.Parse(args); err != nil {
		return Config{}, "", err
	}
	if fs.NArg() > 0 {
		command = fs.Arg(0)
	}
	return cfg, command, nil
}
