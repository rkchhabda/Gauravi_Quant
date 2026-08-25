with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Find the predict method and fix its indentation
# The class ends before line 522, so predict should be inside the class (indent 4)
# Find where the class ends and add predict inside

# Find the line where class DailyKronosPipeline ends (look for next class or function at indent 0)
class_end = None
for i in range(380, 550):
    if lines[i].startswith('class ') or (lines[i].startswith('def ') and not lines[i].startswith('    def ')):
        if i > 380:  # after the class definition
            class_end = i
            break

print(f'Class ends at line {class_end}')

# The predict method should be inside the class, so move it before class_end
# Lines 522-602 are the predict method (should be inside class)

# Fix: indent lines 522-602 by 4 spaces
for i in range(522, 603):
    if not lines[i].startswith(' ') and lines[i].strip():
        lines[i] = '    ' + lines[i]
    elif lines[i].startswith('        '):
        lines[i] = '    ' + lines[i]

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed predict method indentation')