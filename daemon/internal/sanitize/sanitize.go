// Package sanitize provides utilities for redacting sensitive data from log output.
package sanitize

import (
	"os"
	"strings"
)

var homeDir string

func init() {
	homeDir, _ = os.UserHomeDir()
}

// Path replaces the user's home directory with "~" in the given path string.
func Path(path string) string {
	if homeDir != "" && homeDir != "/" {
		return strings.ReplaceAll(path, homeDir, "~")
	}
	return path
}
