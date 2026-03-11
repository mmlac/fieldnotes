package sources

import "time"

// IngestEvent is the single envelope passed from every source to the dispatcher,
// and from the dispatcher to the Python worker. Its shape is fixed; adapters fill
// the fields that are relevant to them and leave the rest empty.
type IngestEvent struct {
	// Identity
	ID         string    `json:"id"`          // UUIDv4, set by dispatcher
	SourceType string    `json:"source_type"` // matches Source.Name()
	SourceID   string    `json:"source_id"`   // stable external identifier (path, message_id, etc.)
	Operation  Operation `json:"operation"`   // Created | Modified | Deleted

	// Content
	// For text-based sources: Text is populated, RawBytes is nil.
	// For binary sources (images, PDFs): RawBytes is populated, Text is empty.
	// The Python parser decides how to handle each content type.
	Text     string `json:"text,omitempty"`
	RawBytes []byte `json:"raw_bytes,omitempty"` // base64 in JSON
	MimeType string `json:"mime_type,omitempty"`

	// Source-specific structured metadata.
	// Adapters may include anything here — the Python parser for this source type
	// knows how to interpret it. The core pipeline ignores unknown fields.
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
