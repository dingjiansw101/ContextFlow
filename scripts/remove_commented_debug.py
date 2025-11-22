#!/usr/bin/env python3
"""
Script to remove commented debug statements from the codebase.
Removes:
- Commented ipdb statements (# import ipdb; ipdb.set_trace(), etc.)
- Commented jax.debug.print statements
"""

import re
from pathlib import Path
import sys

# Patterns to match commented debug lines
IPDB_PATTERNS = [
    r'^\s*#\s*import\s+ipdb\s*;\s*ipdb\.set_trace\(\)',
    r'^\s*#\s*import\s+ipdb',
    r'^\s*#\s*ipdb\.set_trace\(\)',
]

JAX_DEBUG_PATTERNS = [
    r'^\s*#\s*jax\.debug\.print\(',
]

def should_remove_line(line: str) -> bool:
    """Check if a line should be removed."""
    for pattern in IPDB_PATTERNS + JAX_DEBUG_PATTERNS:
        if re.match(pattern, line):
            return True
    return False

def clean_file(file_path: Path) -> tuple[int, int]:
    """
    Clean a single file by removing commented debug statements.
    Returns (lines_removed, total_lines).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    original_count = len(lines)
    cleaned_lines = []
    removed_count = 0

    for line in lines:
        if should_remove_line(line):
            removed_count += 1
            # Skip this line (remove it)
        else:
            cleaned_lines.append(line)

    if removed_count > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            print(f"✓ {file_path}: removed {removed_count} lines")
        except Exception as e:
            print(f"Error writing {file_path}: {e}", file=sys.stderr)
            return 0, original_count

    return removed_count, original_count

def main():
    """Main function to clean all Python files in the codebase."""
    root = Path(__file__).parent.parent

    # Find all Python files (excluding third_party and .venv)
    python_files = []
    for pattern in ['src/**/*.py', 'scripts/**/*.py', 'examples/**/*.py', 'packages/**/*.py']:
        python_files.extend(root.glob(pattern))

    # Filter out third_party and .venv
    python_files = [
        f for f in python_files
        if 'third_party' not in str(f) and '.venv' not in str(f) and 'wandb' not in str(f)
    ]

    total_removed = 0
    total_files_changed = 0

    print(f"Scanning {len(python_files)} Python files...")
    print()

    for file_path in sorted(python_files):
        removed, _ = clean_file(file_path)
        if removed > 0:
            total_removed += removed
            total_files_changed += 1

    print()
    print(f"Summary:")
    print(f"  Files modified: {total_files_changed}")
    print(f"  Lines removed: {total_removed}")

    return 0 if total_removed > 0 else 1

if __name__ == '__main__':
    sys.exit(main())
