#!/usr/bin/env python3
"""
Script to remove debug_fused_checks and related code from pi0_light_incontextv14.py.
"""

import re
from pathlib import Path

def remove_debug_blocks(lines):
    """Remove if cfg_dbg: blocks and their indented content."""
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a cfg_dbg conditional
        if re.match(r'^\s*if\s+cfg_dbg.*:?\s*$', line):
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
            # Not a debug block, keep the line
            result.append(line)
            i += 1

    return result

def main():
    file_path = Path("src/openpi/models/pi0_light_incontextv14.py")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    original_count = len(lines)

    # Remove cfg_dbg assignment and if cfg_dbg blocks
    lines = [line for line in lines if not re.match(r'^\s*cfg_dbg\s*=\s*self\.debug_fused_checks\s*$', line)]

    # Remove all _dbg_ method definitions and usages
    lines = [line for line in lines if not re.match(r'^\s*def\s+_dbg_.*\(', line)]

    # Remove if cfg_dbg: blocks
    lines = remove_debug_blocks(lines)

    # Remove _dbg_ assignments
    lines = [line for line in lines if not re.search(r'self\._dbg_', line)]

    with open(file_path, 'w') as f:
        f.writelines(lines)

    removed = original_count - len(lines)
    print(f"✓ Modified {file_path} (removed {removed} lines)")

if __name__ == '__main__':
    main()
