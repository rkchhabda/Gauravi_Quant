import ast
with open('kronos_high_accuracy_advisor.py', 'r', encoding='utf-8') as f:
    source = f.read()
try:
    ast.parse(source)
    print('Syntax OK')
except SyntaxError as e:
    print('SyntaxError at line', e.lineno, ':', e.msg)
    lines = source.splitlines()
    if e.lineno <= len(lines):
        print('Line:', repr(lines[e.lineno-1]))
        # Show context
        for i in range(max(0, e.lineno-5), min(len(lines), e.lineno+5)):
            print(f'{i+1}: {repr(lines[i])}')