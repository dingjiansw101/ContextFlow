#!/usr/bin/env python3
"""
Script to remove debug_checks parameter and related code from Python files.
Removes:
- debug_checks parameter definitions
- if self.debug_checks: blocks and their contents
"""

import re
from pathlib import Path
import sys

def remove_debug_checks_from_file(file_path: Path) -> bool:
    """
    Remove debug_checks from a single file.
    Returns True if the file was modified.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False

    original_content = content

    # Remove debug_checks parameter lines (with comments)
    content = re.sub(r'^\s*debug_checks:\s*bool\s*=\s*(True|False)\s*(#.*)?$\n?', '', content, flags=re.MULTILINE)

    # Remove if self.debug_checks: blocks (but keep the indented code inside)
    # This is tricky - we'll just comment them out for manual review
    # Pattern: find "if self.debug_checks:" and remove just that line
    content = re.sub(r'^\s*if\s+self\.debug_checks:\s*$\n', '', content, flags=re.MULTILINE)

    # Remove debug_checks parameter from function/class calls
    content = re.sub(r',?\s*debug_checks\s*=\s*[^,\)]+', '', content)
    content = re.sub(r'debug_checks\s*=\s*[^,\)]+,?\s*', '', content)

    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Modified {file_path}")
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
        root / "src/openpi/models/pi0_light_incontextv14.py",
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
