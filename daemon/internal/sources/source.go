// Package sources defines the Source interface and IngestEvent types
// for the fieldnotes daemon. Every source adapter implements this interface.
package sources

import "context"

// Source is the interface every adapter must implement.
// It is responsible for detecting changes in an external system
// and emitting IngestEvents to the dispatcher.
type Source interface {
	// Name returns the stable identifier for this source type.
	// Used as the "source_type" field in IngestEvent and in config.toml section names.
	Name() string

	// Configure initialises the source from its config section.
	// Called once at startup before Start().
	Configure(cfg map[string]any) error

	// Start begins watching/polling and emits events to the provided channel.
	// Must be safe to call in a goroutine. Blocks until ctx is cancelled.
	Start(ctx context.Context, events chan<- IngestEvent) error

	// Healthcheck returns a non-nil error if the source is unhealthy
	// (e.g. lost OAuth token, missing directory). Called by the status API.
	Healthcheck() error
}
