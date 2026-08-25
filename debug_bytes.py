with open('kronos_high_accuracy_advisor.py', 'rb') as f:
    content = f.read()

idx = content.find(b'def predict_next_candle')
if idx >= 0:
    start = max(0, idx - 150)
    end = min(len(content), idx + 50)
    chunk = content[start:end]
    for i, b in enumerate(chunk):
        if b < 32 or b > 126:
            print('  Offset {}: {} ({})'.format(start+i, hex(b), chr(b) if b < 128 else '?'))
        else:
            print('  Offset {}: {}'.format(start+i, chr(b)))