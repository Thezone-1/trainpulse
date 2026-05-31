package logging

import (
	"io"
	"log/slog"
	"strings"

	"github.com/somoprovo/trainpulse/internal/config"
)

func New(cfg config.Config, w io.Writer) *slog.Logger {
	level := new(slog.LevelVar)
	switch strings.ToLower(cfg.LogLevel) {
	case "debug":
		level.Set(slog.LevelDebug)
	case "warn", "warning":
		level.Set(slog.LevelWarn)
	case "error":
		level.Set(slog.LevelError)
	default:
		level.Set(slog.LevelInfo)
	}
	opts := &slog.HandlerOptions{Level: level}
	if strings.EqualFold(cfg.LogFormat, "text") {
		return slog.New(slog.NewTextHandler(w, opts))
	}
	return slog.New(slog.NewJSONHandler(w, opts))
}
