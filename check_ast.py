import ast
import sys

with open('kronos_high_accuracy_advisor.py', 'r', encoding='utf-8') as f:
    source = f.read()

try:
    ast.parse(source)
    print('Syntax OK')
except SyntaxError as e:
    print(f'SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}')
    lines = source.splitlines()
    if e.lineno <= len(lines):
        print(f'Error line: {repr(lines[e.lineno-1])}')
        # Show context
        start = max(0, e.lineno - 10)
        end = min(len(lines), e.lineno + 5)
        for i in range(start, end):
            marker = '>>> ' if i == e.lineno - 1 else '    '
            leading = len(lines[i]) - len(lines[i].lstrip())
            print(f'{marker}{i+1:4d}: indent={leading} {repr(lines[i][:100])}')