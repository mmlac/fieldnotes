// fieldnotes-daemon is the always-on runtime for the Fieldnotes personal
// knowledge graph. It watches configured sources and dispatches IngestEvents
// to the Python worker for ML processing.
//
// Source adapters are registered via init() in their packages. Import them
// here for side effects once they are implemented:
//
//	_ "github.com/mmlac/fieldnotes/daemon/internal/sources/files"
//	_ "github.com/mmlac/fieldnotes/daemon/internal/sources/obsidian"
//	_ "github.com/mmlac/fieldnotes/daemon/internal/sources/gmail"
//	_ "github.com/mmlac/fieldnotes/daemon/internal/sources/repositories"
package main

import "fmt"

func main() {
	fmt.Println("fieldnotes-daemon: skeleton — no adapters compiled yet")
}
