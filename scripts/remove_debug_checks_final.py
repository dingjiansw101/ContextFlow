#!/usr/bin/env python3
"""
Script to remove debug_checks parameter and ALL related validation code.
This will:
1. Remove debug_checks parameter definitions
2. Remove entire if self.debug_checks: blocks and their contents
3. Remove debug_checks parameter from function calls
"""

import re
from pathlib import Path
import sys

def remove_debug_checks_blocks(lines):
    """Remove if self.debug_checks: blocks and their indented content."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a debug_checks conditional
        if re.match(r'^\s*if\s+self\.debug_checks.*:?\s*$', line):
            # Get the indentation of the if statement
            if_indent = len(line) - len(line.lstrip())

            # Skip the if line
            i += 1

            # Skip all lines that are more indented than the if
            while i < len(lines):
                next_line = lines[i]
                # If blank line, skip it
                if next_line.strip() == '':
                    i += 1
                    continue

                next_indent = len(next_line) - len(next_line.lstrip())
                # If this line is indented more than the if, it's part of the block - skip it
                if next_indent > if_indent:
                    i += 1
                else:
                    # We've reached the end of the block
                    break
        else:
            # Not a debug_checks block, keep the line
            result.append(line)
            i += 1

    return result

def remove_debug_checks_from_file(file_path: Path) -> bool:
    """Remove debug_checks from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    original_lines = lines[:]

    # Step 1: Remove debug_checks parameter definitions
    lines = [line for line in lines if not re.match(r'^\s*debug_checks:\s*bool\s*=.*$', line)]

    # Step 2: Remove if self.debug_checks: blocks
    lines = remove_debug_checks_blocks(lines)

    # Step 3: Remove debug_checks from parameter lists
    new_lines = []
    for line in lines:
        # Remove debug_checks=... from parameter lists
        line = re.sub(r',\s*debug_checks\s*=\s*[^,\)]+', '', line)
        line = re.sub(r'debug_checks\s*=\s*[^,\)]+,\s*', '', line)
        # Remove config.debug_fused_checks assignments
        line = re.sub(r'"debug_checks"\s*:\s*config\.debug_fused_checks,?', '', line)
        new_lines.append(line)

    lines = new_lines

    if lines != original_lines:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            removed = len(original_lines) - len(lines)
            print(f"✓ Modified {file_path} (removed {removed} lines)")
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}", file=sys.stderr)
            return False

    return False

def main():
    """Main function."""
    root = Path(__file__).parent.parent

    # Target files with debug_checks
    target_files = [
        root / "src/openpi/models/gemma_kvcache.py",
        root / "src/openpi/models/backup_gemma_kvcache.py",
        root / "src/openpi/transforms.py",
    ]

    modified_count = 0
    for file_path in target_files:
        if file_path.exists():
            if remove_debug_checks_from_file(file_path):
                modified_count += 1
        else:
            print(f"Warning: {file_path} not found")

    print(f"\nModified {modified_count} files")
    return 0

if __name__ == '__main__':
    sys.exit(main())
