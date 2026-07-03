// Package version carries the build identity stamped in at release time:
//
//	go build -ldflags "-X github.com/somoprovo/trainpulse/internal/version.Version=v0.2.0 \
//	                   -X github.com/somoprovo/trainpulse/internal/version.Commit=$(git rev-parse --short HEAD)" ./cmd/trainpulse
package version

var (
	Version = "dev"
	Commit  = "unknown"
)
