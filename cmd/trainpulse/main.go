package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/somoprovo/trainpulse/internal/agent"
	"github.com/somoprovo/trainpulse/internal/api"
	"github.com/somoprovo/trainpulse/internal/collector"
	"github.com/somoprovo/trainpulse/internal/config"
	"github.com/somoprovo/trainpulse/internal/dashboard"
	"github.com/somoprovo/trainpulse/internal/logging"
	"github.com/somoprovo/trainpulse/internal/version"
)

func main() {
	if len(os.Args) > 1 && (os.Args[1] == "version" || os.Args[1] == "-version" || os.Args[1] == "--version") {
		fmt.Printf("trainpulse %s (commit %s)\n", version.Version, version.Commit)
		return
	}
	cfg, command, err := config.FromFlags(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := run(command, cfg); err != nil && !errors.Is(err, context.Canceled) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(command string, cfg config.Config) error {
	switch command {
	case "daemon":
		return runDaemon(cfg)
	case "top":
		return runTop(cfg)
	case "snapshot":
		return runSnapshot(cfg)
	default:
		return fmt.Errorf("unknown command %q; use daemon, top, snapshot, or version", command)
	}
}

func runDaemon(cfg config.Config) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	logger := logging.New(cfg, os.Stdout)
	col := chooseCollector(cfg)
	a := agent.New(cfg, col)
	a.SetLogger(logger)
	server := &http.Server{Addr: cfg.Addr, Handler: api.New(a, cfg).Handler(), ReadHeaderTimeout: 2 * time.Second}
	errc := make(chan error, 2)
	go func() { errc <- a.Run(ctx) }()
	go func() { errc <- server.ListenAndServe() }()
	logger.Info("daemon_started", "version", version.Version, "addr", cfg.Addr, "collector", col.Name(), "mode", cfg.Mode)
	err := <-errc
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = server.Shutdown(shutdownCtx)
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	logger.Error("daemon_stopped", "error", err)
	return err
}

func runTop(cfg config.Config) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	col := chooseCollector(cfg)
	logging.New(cfg, os.Stderr).Debug("top_started", "collector", col.Name())
	a := agent.New(cfg, col)
	errc := make(chan error, 1)
	go func() { errc <- a.Run(ctx) }()
	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errc:
			return err
		case <-ticker.C:
			dashboard.Render(os.Stdout, a.Snapshot())
		}
	}
}

func runSnapshot(cfg config.Config) error {
	col := chooseCollector(cfg)
	a := agent.New(cfg, col)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := a.Tick(ctx); err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(a.Snapshot())
}

func chooseCollector(cfg config.Config) collector.Collector {
	host := collector.NewHostCollector()
	if cfg.Mode == "sim" {
		return collector.NewSimCollector()
	}
	nvidia := collector.NewNvidiaSMICollector()
	if cfg.Mode == "nvidia-smi" {
		return collector.NewComposite(nvidia, host)
	}
	if cfg.Mode == "auto" {
		return collector.NewFallback(collector.NewComposite(nvidia, host), collector.NewSimCollector())
	}
	return collector.NewSimCollector()
}
