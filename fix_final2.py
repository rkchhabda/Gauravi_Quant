with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Fix line 603 (0-indexed 602) - blank line should be 0 spaces
lines[602] = '\n'

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed')