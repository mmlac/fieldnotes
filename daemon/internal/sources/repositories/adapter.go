// Package repositories implements the Source interface for scanning local
// git repositories.  It discovers repos under configured root directories,
// enumerates tracked files matching include/exclude glob patterns, extracts
// recent commit history, and emits IngestEvents that match the schema
// produced by the Python RepositorySource.
package repositories

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/bmatcuk/doublestar/v4"
	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"

	"github.com/mmlac/fieldnotes/daemon/internal/sources"
)

func init() {
	sources.Register("repositories", func() sources.Source { return &RepoSource{} })
}

// Default configuration values matching the Python source.
const (
	defaultPollInterval = 300 * time.Second
	defaultMaxFileSize  = 100 * 1024 * 1024 // 100 MiB
	defaultMaxCommits   = 200
)

var defaultIncludePatterns = []string{
	"README*",
	"CHANGELOG*",
	"CONTRIBUTING*",
	"docs/**/*.md",
	"*.toml",
	"ADR/**/*.md",
	"*.csproj",
	"*.fsproj",
	"*.vbproj",
	"Directory.Packages.props",
	"packages.config",
}

var defaultExcludePatterns = []string{
	"node_modules/",
	".git/",
	"vendor/",
	"target/",
	"__pycache__/",
}

// RepoSource scans local git repositories for documentation files and commits.
type RepoSource struct {
	repoRoots       []string
	pollInterval    time.Duration
	includePatterns []string
	excludePatterns []string
	maxFileSize     int64
	maxCommits      int
	cursors         map[string]string // repo_path → HEAD sha
}

func (s *RepoSource) Name() string { return "repositories" }

func (s *RepoSource) Configure(cfg map[string]any) error {
	roots, ok := cfg["repo_roots"]
	if !ok {
		return fmt.Errorf("repositories source requires 'repo_roots' in config")
	}
	rootSlice, ok := roots.([]any)
	if !ok {
		return fmt.Errorf("repo_roots must be a list of strings")
	}
	s.repoRoots = make([]string, 0, len(rootSlice))
	for _, r := range rootSlice {
		rs, ok := r.(string)
		if !ok {
			return fmt.Errorf("repo_roots entries must be strings")
		}
		expanded := expandHome(rs)
		abs, err := filepath.Abs(expanded)
		if err != nil {
			return fmt.Errorf("resolving repo root %q: %w", rs, err)
		}
		s.repoRoots = append(s.repoRoots, abs)
	}

	s.pollInterval = defaultPollInterval
	if v, ok := cfg["poll_interval_seconds"].(float64); ok {
		s.pollInterval = time.Duration(v) * time.Second
	}

	s.includePatterns = defaultIncludePatterns
	if v, ok := cfg["include_patterns"].([]any); ok {
		s.includePatterns = toStringSlice(v)
	}

	s.excludePatterns = defaultExcludePatterns
	if v, ok := cfg["exclude_patterns"].([]any); ok {
		s.excludePatterns = toStringSlice(v)
	}

	s.maxFileSize = defaultMaxFileSize
	if v, ok := cfg["max_file_size"].(float64); ok {
		s.maxFileSize = int64(v)
	}

	s.maxCommits = defaultMaxCommits
	if v, ok := cfg["max_commits"].(float64); ok {
		s.maxCommits = int(v)
	}

	s.cursors = make(map[string]string)
	return nil
}

func (s *RepoSource) Start(ctx context.Context, events chan<- sources.IngestEvent) error {
	s.poll(ctx, events)
	ticker := time.NewTicker(s.pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.poll(ctx, events)
		case <-ctx.Done():
			return nil
		}
	}
}

func (s *RepoSource) Healthcheck() error {
	for _, root := range s.repoRoots {
		info, err := os.Stat(root)
		if err != nil {
			return fmt.Errorf("repo root %q: %w", root, err)
		}
		if !info.IsDir() {
			return fmt.Errorf("repo root %q is not a directory", root)
		}
	}
	return nil
}

// poll performs a single scan cycle across all configured repo roots.
func (s *RepoSource) poll(ctx context.Context, events chan<- sources.IngestEvent) {
	for _, root := range s.repoRoots {
		repos := discoverRepos(root)
		for _, repoPath := range repos {
			if ctx.Err() != nil {
				return
			}
			s.scanRepo(ctx, repoPath, events)
		}
	}
}

// discoverRepos finds git repositories under root (non-bare, one level deep).
func discoverRepos(root string) []string {
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		slog.Warn("repo_root is not a directory, skipping", "root", root)
		return nil
	}

	var repos []string

	// Check if root itself is a repo.
	if isGitRepo(root) {
		repos = append(repos, root)
	}

	// Walk one level of subdirectories.
	entries, err := os.ReadDir(root)
	if err != nil {
		slog.Warn("failed to list repo_root", "root", root, "error", err)
		return repos
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		child := filepath.Join(root, entry.Name())
		if isGitRepo(child) {
			repos = append(repos, child)
		}
	}
	return repos
}

func isGitRepo(dir string) bool {
	info, err := os.Stat(filepath.Join(dir, ".git"))
	return err == nil && info.IsDir()
}

// scanRepo scans a single repository, emitting file and commit events.
func (s *RepoSource) scanRepo(ctx context.Context, repoPath string, events chan<- sources.IngestEvent) {
	repo, err := git.PlainOpen(repoPath)
	if err != nil {
		slog.Warn("failed to open git repo", "path", repoPath, "error", err)
		return
	}

	head, err := repo.Head()
	if err != nil {
		slog.Debug("skipping repo with no HEAD", "path", repoPath)
		return
	}
	headSHA := head.Hash().String()

	prevSHA, seen := s.cursors[repoPath]
	if seen && prevSHA == headSHA {
		return // no changes
	}

	repoName := filepath.Base(repoPath)
	remoteURL := getRemoteURL(repo)

	// Emit file events.
	s.scanFiles(ctx, repo, repoPath, repoName, remoteURL, events)

	// Emit commit events.
	s.scanCommits(ctx, repo, repoPath, repoName, remoteURL, events)

	s.cursors[repoPath] = headSHA
}

// scanFiles walks tracked files and emits events for those matching include patterns.
func (s *RepoSource) scanFiles(ctx context.Context, repo *git.Repository, repoPath, repoName, remoteURL string, events chan<- sources.IngestEvent) {
	head, err := repo.Head()
	if err != nil {
		return
	}
	commit, err := repo.CommitObject(head.Hash())
	if err != nil {
		return
	}
	tree, err := commit.Tree()
	if err != nil {
		return
	}

	count := 0
	tree.Files().ForEach(func(f *object.File) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		relPath := f.Name
		if !s.matchesInclude(relPath) || s.matchesExclude(relPath) {
			return nil
		}

		absPath := filepath.Join(repoPath, relPath)
		info, err := os.Stat(absPath)
		if err != nil {
			return nil
		}
		if info.Size() > s.maxFileSize {
			slog.Warn("skipping file exceeding max size", "path", absPath, "size", info.Size())
			return nil
		}

		text := ""
		mimeType := guessMIME(relPath)
		if isTextMIME(mimeType) {
			data, err := os.ReadFile(absPath)
			if err == nil {
				text = string(data)
			}
		}

		events <- sources.IngestEvent{
			SourceType: "repositories",
			SourceID:   fmt.Sprintf("repo:%s:%s", repoPath, relPath),
			Operation:  sources.OperationCreated,
			Text:       text,
			MimeType:   mimeType,
			Meta: map[string]any{
				"repo_name":     repoName,
				"repo_path":     repoPath,
				"remote_url":    remoteURL,
				"relative_path": relPath,
			},
			SourceModifiedAt: info.ModTime().UTC(),
			EnqueuedAt:       time.Now().UTC(),
		}
		count++
		return nil
	})

	if count > 0 {
		slog.Info("scanned files from repo", "repo", repoName, "count", count)
	}
}

// scanCommits emits events for recent commits, up to maxCommits.
func (s *RepoSource) scanCommits(ctx context.Context, repo *git.Repository, repoPath, repoName, remoteURL string, events chan<- sources.IngestEvent) {
	head, err := repo.Head()
	if err != nil {
		return
	}

	iter, err := repo.Log(&git.LogOptions{
		From:  head.Hash(),
		Order: git.LogOrderCommitterTime,
	})
	if err != nil {
		slog.Warn("failed to iterate commits", "repo", repoName, "error", err)
		return
	}

	count := 0
	iter.ForEach(func(c *object.Commit) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if count >= s.maxCommits {
			return fmt.Errorf("stop") // break iteration
		}

		changedFiles := commitChangedFiles(c)
		authorDate := c.Author.When.UTC()

		events <- sources.IngestEvent{
			SourceType: "repositories",
			SourceID:   fmt.Sprintf("commit:%s:%s", repoPath, c.Hash.String()),
			Operation:  sources.OperationCreated,
			Text:       c.Message,
			MimeType:   "text/plain",
			Meta: map[string]any{
				"sha":           c.Hash.String(),
				"author_name":   c.Author.Name,
				"author_email":  c.Author.Email,
				"date":          authorDate.Format(time.RFC3339),
				"repo_name":     repoName,
				"repo_path":     repoPath,
				"remote_url":    remoteURL,
				"changed_files": changedFiles,
			},
			SourceModifiedAt: authorDate,
			EnqueuedAt:       time.Now().UTC(),
		}
		count++
		return nil
	})

	if count > 0 {
		slog.Info("indexed commits from repo", "repo", repoName, "count", count)
	}
}

// commitChangedFiles returns the list of file paths changed in a commit.
func commitChangedFiles(c *object.Commit) []string {
	stats, err := c.Stats()
	if err != nil {
		return nil
	}
	paths := make([]string, 0, len(stats))
	for _, s := range stats {
		paths = append(paths, s.Name)
	}
	return paths
}

// getRemoteURL returns the origin remote URL or empty string.
func getRemoteURL(repo *git.Repository) string {
	remote, err := repo.Remote("origin")
	if err != nil {
		return ""
	}
	urls := remote.Config().URLs
	if len(urls) == 0 {
		return ""
	}
	return urls[0]
}

// matchesInclude checks if relPath matches any include glob pattern.
func (s *RepoSource) matchesInclude(relPath string) bool {
	name := filepath.Base(relPath)
	for _, pat := range s.includePatterns {
		if matched, _ := doublestar.Match(pat, relPath); matched {
			return true
		}
		if matched, _ := doublestar.Match(pat, name); matched {
			return true
		}
	}
	return false
}

// matchesExclude checks if relPath matches any exclude glob pattern.
// Patterns ending in "/" are treated as directory prefixes.
func (s *RepoSource) matchesExclude(relPath string) bool {
	for _, pat := range s.excludePatterns {
		// Directory prefix pattern: "vendor/" matches "vendor/lib.go".
		if len(pat) > 0 && pat[len(pat)-1] == '/' {
			prefix := pat[:len(pat)-1]
			if relPath == prefix || len(relPath) > len(prefix) && relPath[:len(prefix)+1] == prefix+"/" {
				return true
			}
			continue
		}
		if matched, _ := doublestar.Match(pat, relPath); matched {
			return true
		}
	}
	return false
}

// guessMIME returns a MIME type based on file extension.
func guessMIME(path string) string {
	ext := filepath.Ext(path)
	switch ext {
	case ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".xml", ".csv":
		return "text/plain"
	case ".go", ".py", ".js", ".ts", ".rs", ".rb", ".java", ".c", ".h", ".cpp":
		return "text/plain"
	case ".csproj", ".fsproj", ".vbproj", ".props", ".config":
		return "text/xml"
	default:
		return "application/octet-stream"
	}
}

func isTextMIME(mime string) bool {
	return len(mime) >= 5 && mime[:5] == "text/"
}

// expandHome replaces a leading ~ with the user's home directory.
func expandHome(path string) string {
	if len(path) == 0 || path[0] != '~' {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[1:])
}

// toStringSlice converts []any to []string, skipping non-string entries.
func toStringSlice(in []any) []string {
	out := make([]string, 0, len(in))
	for _, v := range in {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
