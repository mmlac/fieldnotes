package sources

import "fmt"

// SourceFactory is a constructor function for a Source implementation.
type SourceFactory func() Source

// registry holds source factories. It is written only during init() and read
// after that, so no mutex is needed. Do not write to it after program startup.
var registry = map[string]SourceFactory{}

// Register adds a source factory to the registry.
// It must only be called from init() functions before any concurrent access.
func Register(name string, factory SourceFactory) {
	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("source already registered: %s", name))
	}
	registry[name] = factory
}

// Build constructs all sources enabled in the config.
func Build(sourceCfgs map[string]map[string]any) ([]Source, error) {
	var sources []Source
	for name, sectionCfg := range sourceCfgs {
		factory, ok := registry[name]
		if !ok {
			return nil, fmt.Errorf("unknown source type %q in config — is the adapter compiled in?", name)
		}
		s := factory()
		if err := s.Configure(sectionCfg); err != nil {
			return nil, fmt.Errorf("configuring source %q: %w", name, err)
		}
		sources = append(sources, s)
	}
	return sources, nil
}
