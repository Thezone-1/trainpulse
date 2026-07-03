VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
LDFLAGS  = -s -w \
	-X github.com/somoprovo/trainpulse/internal/version.Version=$(VERSION) \
	-X github.com/somoprovo/trainpulse/internal/version.Commit=$(COMMIT)

.PHONY: build test vet release clean

build:
	go build -ldflags "$(LDFLAGS)" -o trainpulse ./cmd/trainpulse

test:
	go test ./...

vet:
	go vet ./...

# Cross-compiled release binaries for the platforms GPU hosts actually run.
release: test vet
	GOOS=linux  GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/trainpulse-linux-amd64 ./cmd/trainpulse
	GOOS=linux  GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/trainpulse-linux-arm64 ./cmd/trainpulse
	GOOS=darwin GOARCH=arm64 go build -ldflags "$(LDFLAGS)" -o dist/trainpulse-darwin-arm64 ./cmd/trainpulse
	GOOS=windows GOARCH=amd64 go build -ldflags "$(LDFLAGS)" -o dist/trainpulse-windows-amd64.exe ./cmd/trainpulse

clean:
	rm -rf dist trainpulse trainpulse.exe
