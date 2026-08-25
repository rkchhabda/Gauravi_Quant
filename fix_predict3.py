with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Class starts at line 421 (index 420), ends at line 521 (index 520)
# predict method is at line 522 (index 521) to line ~602

# Extract predict method (from line 522 to line 602)
predict_start = 521  # 0-indexed
predict_end = 601    # inclusive

predict_lines = lines[predict_start:predict_end+1]

# Remove from current location
del lines[predict_start:predict_end+1]

# Insert before class end (line 521, index 520)
insert_idx = 520  # before the line that ends the class

# Add proper indentation (4 spaces for class method)
indented = []
for line in predict_lines:
    if line.strip() == '':
        indented.append('\n')
    elif line.startswith('    ') or line.startswith('\t'):
        indented.append('    ' + line)
    else:
        indented.append('    ' + line)

# Insert before class end
for i, line in enumerate(indented):
    lines.insert(520 + i, line)

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)

print('Moved predict method inside class')
print(f'Inserted {len(indented)} lines at index 520')