package repositories

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/mmlac/fieldnotes/daemon/internal/sources"
)

// initTestRepo creates a temporary git repository with one committed file
// and returns its path along with a cleanup function.
func initTestRepo(t *testing.T, fileName, content string) string {
	t.Helper()
	dir := t.TempDir()

	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=Test",
			"GIT_AUTHOR_EMAIL=test@test.com",
			"GIT_COMMITTER_NAME=Test",
			"GIT_COMMITTER_EMAIL=test@test.com",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v\n%s", args, err, out)
		}
	}

	run("init")
	run("config", "user.email", "test@test.com")
	run("config", "user.name", "Test")

	fpath := filepath.Join(dir, fileName)
	if err := os.MkdirAll(filepath.Dir(fpath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fpath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", fileName)
	run("commit", "-m", "initial commit")

	return dir
}

func TestName(t *testing.T) {
	s := &RepoSource{}
	if s.Name() != "repositories" {
		t.Fatalf("expected Name() = %q, got %q", "repositories", s.Name())
	}
}

func TestConfigure_MissingRepoRoots(t *testing.T) {
	s := &RepoSource{}
	err := s.Configure(map[string]any{})
	if err == nil {
		t.Fatal("expected error when repo_roots is missing")
	}
}

func TestConfigure_ValidConfig(t *testing.T) {
	s := &RepoSource{}
	err := s.Configure(map[string]any{
		"repo_roots":           []any{"/tmp/test"},
		"poll_interval_seconds": float64(60),
		"max_commits":          float64(10),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if s.pollInterval != 60*time.Second {
		t.Fatalf("expected poll interval 60s, got %v", s.pollInterval)
	}
	if s.maxCommits != 10 {
		t.Fatalf("expected max_commits 10, got %d", s.maxCommits)
	}
}

func TestHealthcheck_ValidRoot(t *testing.T) {
	dir := t.TempDir()
	s := &RepoSource{repoRoots: []string{dir}}
	if err := s.Healthcheck(); err != nil {
		t.Fatalf("unexpected healthcheck error: %v", err)
	}
}

func TestHealthcheck_MissingRoot(t *testing.T) {
	s := &RepoSource{repoRoots: []string{"/nonexistent/path/xyzzy"}}
	if err := s.Healthcheck(); err == nil {
		t.Fatal("expected healthcheck error for missing root")
	}
}

func TestDiscoverRepos(t *testing.T) {
	parent := t.TempDir()
	// Create two child repos.
	for _, name := range []string{"repo-a", "repo-b"} {
		child := filepath.Join(parent, name)
		os.MkdirAll(filepath.Join(child, ".git"), 0o755)
	}
	// Create a non-repo directory.
	os.MkdirAll(filepath.Join(parent, "not-a-repo"), 0o755)

	repos := discoverRepos(parent)
	if len(repos) != 2 {
		t.Fatalf("expected 2 repos, got %d: %v", len(repos), repos)
	}
}

func TestDiscoverRepos_IgnoresSymlinkedDotGit(t *testing.T) {
	parent := t.TempDir()
	// Create a real repo for the symlink target.
	realRepo := initTestRepo(t, "README.md", "# Real\n")

	// Create a directory whose .git is a symlink to the real repo's .git.
	fakeRepo := filepath.Join(parent, "fake-repo")
	if err := os.MkdirAll(fakeRepo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(realRepo, ".git"), filepath.Join(fakeRepo, ".git")); err != nil {
		t.Skip("symlink not supported:", err)
	}

	repos := discoverRepos(parent)
	for _, r := range repos {
		if r == fakeRepo {
			t.Errorf("discoverRepos should not include directory with symlinked .git: %s", fakeRepo)
		}
	}
}

func TestMatchesIncludeExclude(t *testing.T) {
	s := &RepoSource{
		includePatterns: []string{"README*", "docs/**/*.md", "*.toml"},
		excludePatterns: []string{"vendor/"},
	}

	tests := []struct {
		path    string
		include bool
		exclude bool
	}{
		{"README.md", true, false},
		{"docs/guide/intro.md", true, false},
		{"pyproject.toml", true, false},
		{"main.go", false, false},
		{"vendor/lib.go", false, true},
	}

	for _, tt := range tests {
		if got := s.matchesInclude(tt.path); got != tt.include {
			t.Errorf("matchesInclude(%q) = %v, want %v", tt.path, got, tt.include)
		}
		if got := s.matchesExclude(tt.path); got != tt.exclude {
			t.Errorf("matchesExclude(%q) = %v, want %v", tt.path, got, tt.exclude)
		}
	}
}

func TestGuessMIME(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{"README.md", "text/plain"},
		{"config.toml", "text/plain"},
		{"main.go", "text/plain"},
		{"image.png", "application/octet-stream"},
		{"project.csproj", "text/xml"},
	}
	for _, tt := range tests {
		if got := guessMIME(tt.path); got != tt.want {
			t.Errorf("guessMIME(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}

func TestExpandHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("cannot determine home dir")
	}
	got := expandHome("~/projects")
	want := filepath.Join(home, "projects")
	if got != want {
		t.Errorf("expandHome(~/projects) = %q, want %q", got, want)
	}

	// Non-home path should be unchanged.
	if got := expandHome("/tmp/foo"); got != "/tmp/foo" {
		t.Errorf("expandHome(/tmp/foo) = %q, want /tmp/foo", got)
	}

	// ~user syntax is not supported and should be returned unchanged.
	if got := expandHome("~alice/docs"); got != "~alice/docs" {
		t.Errorf("expandHome(~alice/docs) = %q, want ~alice/docs", got)
	}
}

func TestRegistration(t *testing.T) {
	// Verify that the init() function registered the source.
	srcs, err := sources.Build(map[string]map[string]any{
		"repositories": {
			"repo_roots": []any{"/tmp"},
		},
	})
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}
	if len(srcs) != 1 {
		t.Fatalf("expected 1 source, got %d", len(srcs))
	}
	if srcs[0].Name() != "repositories" {
		t.Fatalf("expected source name %q, got %q", "repositories", srcs[0].Name())
	}
}

func TestScanRepo_EmitsFileAndCommitEvents(t *testing.T) {
	repoDir := initTestRepo(t, "README.md", "# Hello World\n")

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(repoDir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"README*"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	s.scanRepo(ctx, repoDir, events)

	cancel() // stop any further work
	close(events)

	var fileEvents, commitEvents []sources.IngestEvent
	for ev := range events {
		if ev.SourceID[:5] == "repo:" {
			fileEvents = append(fileEvents, ev)
		} else if ev.SourceID[:7] == "commit:" {
			commitEvents = append(commitEvents, ev)
		}
	}

	if len(fileEvents) == 0 {
		t.Error("expected at least one file event")
	}
	if len(commitEvents) == 0 {
		t.Error("expected at least one commit event")
	}

	// Verify file event fields.
	if len(fileEvents) > 0 {
		ev := fileEvents[0]
		if ev.SourceType != "repositories" {
			t.Errorf("file event SourceType = %q, want %q", ev.SourceType, "repositories")
		}
		if ev.Text != "# Hello World\n" {
			t.Errorf("file event Text = %q, want %q", ev.Text, "# Hello World\n")
		}
		meta := ev.Meta
		if meta["repo_name"] == "" {
			t.Error("file event missing repo_name in meta")
		}
		if meta["relative_path"] != "README.md" {
			t.Errorf("file event relative_path = %q, want %q", meta["relative_path"], "README.md")
		}
	}

	// Verify commit event fields.
	if len(commitEvents) > 0 {
		ev := commitEvents[0]
		if ev.SourceType != "repositories" {
			t.Errorf("commit event SourceType = %q, want %q", ev.SourceType, "repositories")
		}
		if ev.Text != "initial commit\n" {
			t.Errorf("commit event Text = %q, want %q", ev.Text, "initial commit\n")
		}
		meta := ev.Meta
		if meta["author_email"] != "test@test.com" {
			t.Errorf("commit meta author_email = %v, want %q", meta["author_email"], "test@test.com")
		}
	}

	// Verify cursor was updated.
	if _, ok := s.cursors[repoDir]; !ok {
		t.Error("cursor not updated after scan")
	}
}

func TestScanRepo_MaxFileSizeSkipsLargeFiles(t *testing.T) {
	// Create a file that exceeds a small maxFileSize limit.
	repoDir := initTestRepo(t, "README.md", "# Small\n")

	// Add a large file
	largePath := filepath.Join(repoDir, "big.md")
	largeContent := make([]byte, 1024)
	for i := range largeContent {
		largeContent[i] = 'x'
	}
	if err := os.WriteFile(largePath, largeContent, 0o644); err != nil {
		t.Fatal(err)
	}
	// Commit the large file
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = repoDir
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=Test",
			"GIT_AUTHOR_EMAIL=test@test.com",
			"GIT_COMMITTER_NAME=Test",
			"GIT_COMMITTER_EMAIL=test@test.com",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v\n%s", args, err, out)
		}
	}
	run("add", "big.md")
	run("commit", "-m", "add big file")

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(repoDir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"README*", "*.md"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     512, // Only 512 bytes — big.md exceeds this
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx := context.Background()
	s.scanRepo(ctx, repoDir, events)
	close(events)

	var filePaths []string
	for ev := range events {
		if len(ev.SourceID) > 5 && ev.SourceID[:5] == "repo:" {
			if rp, ok := ev.Meta["relative_path"].(string); ok {
				filePaths = append(filePaths, rp)
			}
		}
	}

	// README.md should be included but big.md should be skipped
	found := false
	for _, p := range filePaths {
		if p == "README.md" {
			found = true
		}
		if p == "big.md" {
			t.Error("big.md should have been skipped due to maxFileSize")
		}
	}
	if !found {
		t.Error("expected README.md in file events")
	}
}

func TestScanRepo_EmptyRepo(t *testing.T) {
	// Create a repo with no commits (just initialized)
	dir := t.TempDir()
	cmd := exec.Command("git", "init")
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init failed: %v\n%s", err, out)
	}

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(dir)},
		pollInterval:    time.Minute,
		includePatterns: defaultIncludePatterns,
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx := context.Background()
	// Should not panic on a repo with no HEAD
	s.scanRepo(ctx, dir, events)
	close(events)

	count := 0
	for range events {
		count++
	}
	if count != 0 {
		t.Errorf("expected 0 events from empty repo, got %d", count)
	}
}

func TestScanRepo_BinaryFileSkipped(t *testing.T) {
	repoDir := initTestRepo(t, "image.png", "\x89PNG\r\n\x1a\n")

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(repoDir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"*"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx := context.Background()
	s.scanRepo(ctx, repoDir, events)
	close(events)

	for ev := range events {
		if len(ev.SourceID) > 5 && ev.SourceID[:5] == "repo:" {
			// Binary files should have empty text (non-text MIME)
			if rp, ok := ev.Meta["relative_path"].(string); ok && rp == "image.png" {
				if ev.Text != "" {
					t.Errorf("binary file should have empty text, got %q", ev.Text)
				}
			}
		}
	}
}

func TestScanRepo_ContextCancellation(t *testing.T) {
	repoDir := initTestRepo(t, "README.md", "# Hello\n")

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(repoDir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"README*"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Should return without hanging
	s.scanRepo(ctx, repoDir, events)
	close(events)

	// We don't assert exact event count — the point is it doesn't hang
}

func TestScanRepo_MaxCommitsLimit(t *testing.T) {
	dir := t.TempDir()
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=Test",
			"GIT_AUTHOR_EMAIL=test@test.com",
			"GIT_COMMITTER_NAME=Test",
			"GIT_COMMITTER_EMAIL=test@test.com",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v failed: %v\n%s", args, err, out)
		}
	}

	run("init")
	run("config", "user.email", "test@test.com")
	run("config", "user.name", "Test")

	// Create 5 commits
	for i := 0; i < 5; i++ {
		fpath := filepath.Join(dir, "file.txt")
		content := fmt.Sprintf("version %d\n", i)
		if err := os.WriteFile(fpath, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		run("add", "file.txt")
		run("commit", "-m", fmt.Sprintf("commit %d", i))
	}

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(dir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"*"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      2, // Limit to 2 commits
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx := context.Background()
	s.scanRepo(ctx, dir, events)
	close(events)

	var commitEvents int
	for ev := range events {
		if len(ev.SourceID) > 7 && ev.SourceID[:7] == "commit:" {
			commitEvents++
		}
	}

	if commitEvents > 2 {
		t.Errorf("expected at most 2 commit events (maxCommits=2), got %d", commitEvents)
	}
}

func TestScanRepo_CursorSkipsUnchanged(t *testing.T) {
	repoDir := initTestRepo(t, "README.md", "# Test\n")

	s := &RepoSource{
		repoRoots:       []string{filepath.Dir(repoDir)},
		pollInterval:    time.Minute,
		includePatterns: []string{"README*"},
		excludePatterns: defaultExcludePatterns,
		maxFileSize:     defaultMaxFileSize,
		maxCommits:      10,
		cursors:         make(map[string]string),
	}

	events := make(chan sources.IngestEvent, 100)
	ctx := context.Background()

	// First scan should emit events.
	s.scanRepo(ctx, repoDir, events)
	count1 := len(events)

	// Second scan with same HEAD should emit nothing.
	s.scanRepo(ctx, repoDir, events)
	count2 := len(events)

	if count2 != count1 {
		t.Errorf("second scan emitted %d events (expected 0 new), total before=%d after=%d", count2-count1, count1, count2)
	}
}
