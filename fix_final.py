with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Fix line 594 (0-indexed 593) - blank line should be 8 spaces
lines[593] = ' ' * 8 + '\n'

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed blank line')