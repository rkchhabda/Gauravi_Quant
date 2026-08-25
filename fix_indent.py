with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Find the problematic lines and fix indentation
# Lines 586-604 (0-indexed: 585-603) have wrong indentation
# The issue: lines 586-593 should be indented 16, line 594 should be 8, line 595 should be 8, lines 596-602 should be 12

# Fix lines 585-603 (0-indexed)
for i in range(585, 594):
    if lines[i].startswith('                '):
        lines[i] = ' ' * 16 + lines[i].lstrip()

# Line 594 (blank line) - should be 8 spaces
if lines[594].strip() == '':
    lines[594] = ' ' * 8 + '\n'

# Line 595 (return statement) - should be 8 spaces
lines[595] = ' ' * 8 + lines[595].lstrip()

# Lines 596-602 (dict items) - should be 12 spaces
for i in range(596, 603):
    if lines[i].startswith('                '):
        lines[i] = ' ' * 12 + lines[i].lstrip()

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed indentation')