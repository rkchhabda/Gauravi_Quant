with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Find the end of the class (line 521 = index 520)
# The predict method at line 522 (index 521) should be inside the class
# Class ends at line 521 (index 520)

# Find the predict method at module level (currently at index 521)
predict_start = None
for i, line in enumerate(lines):
    if line.strip() == 'def predict(self) -> dict:' and i > 400:
        if lines[i-1].strip() == '' and not lines[i].startswith(' '):
            predict_start = i
            break

print(f'predict method at index {predict_start} (line {predict_start+1})')

# The class ends before this. Find where class ends
class_end = None
for i in range(421, 521):
    if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
        if not lines[i].startswith('class ') and not lines[i].startswith('def '):
            continue
        if lines[i].strip().startswith('def ') and i > 421:
            class_end = i
            break

print(f'Class ends at index {class_end} (line {class_end+1})')
print(f'predict method at index {predict_start}')

# Move predict method inside class (before class_end)
if predict_start and class_end and predict_start > class_end:
    # Extract predict method
    predict_lines = []
    i = predict_start
    while i < len(lines) and (lines[i].startswith(' ') or lines[i].strip() == ''):
        predict_lines.append(lines[i])
        i += 1
    # Also include the def line
    predict_lines = [lines[predict_start]] + predict_lines
    
    # Remove from current location
    for _ in range(len(predict_lines)):
        lines.pop(predict_start)
    
    # Insert before class_end
    for i, line in enumerate(predict_lines):
        lines.insert(class_end + i, line)
    
    print(f'Moved {len(predict_lines)} lines to before line {class_end+1}')

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed')