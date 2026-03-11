// Package sources defines the Source interface and IngestEvent types
// for the fieldnotes daemon. Every source adapter implements this interface.
package sources

import (
	"context"
	"time"
)

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

// IngestEvent is the single envelope passed from every source to the dispatcher,
// and from the dispatcher to the Python worker.
type IngestEvent struct {
	// Identity
	ID         string    `json:"id"`          // UUIDv4, set by dispatcher
	SourceType string    `json:"source_type"` // matches Source.Name()
	SourceID   string    `json:"source_id"`   // stable external identifier (path, message_id, etc.)
	Operation  Operation `json:"operation"`   // Created | Modified | Deleted

	// Content
	Text     string `json:"text,omitempty"`
	RawBytes []byte `json:"raw_bytes,omitempty"` // base64 in JSON
	MimeType string `json:"mime_type,omitempty"`

	// Source-specific structured metadata.
	Meta map[string]any `json:"meta,omitempty"`

	// Timestamps
	SourceModifiedAt time.Time `json:"source_modified_at"`
	EnqueuedAt       time.Time `json:"enqueued_at"`
}

// Operation represents the type of change detected by a source.
type Operation string

const (
	OperationCreated  Operation = "created"
	OperationModified Operation = "modified"
	OperationDeleted  Operation = "deleted"
)
