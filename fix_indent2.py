with open('daily_kronos_pipeline.py', 'r') as f:
    lines = f.readlines()

# Fix the specific indentation issues
# Lines 585-595 (0-indexed: 585-594)

# Line 586 (0-indexed 585): if statement - should be 16 (inside else block)
lines[585] = ' ' * 16 + 'if max(ml_prob, 1-ml_prob) * 100 > kronos_f["kronos_confidence"]:\n'

# Line 587: direction = ml_dir - inside if, should be 20
lines[586] = ' ' * 20 + 'direction = ml_dir\n'

# Line 588
lines[587] = ' ' * 20 + 'confidence = round(max(ml_prob, 1-ml_prob) * 100, 1)\n'

# Line 589: else: - should be 16 (matching if)
lines[588] = ' ' * 16 + 'else:\n'

# Line 590: direction = kronos_dir - inside else, 20
lines[589] = ' ' * 20 + 'direction = kronos_dir\n'

# Line 591
lines[590] = ' ' * 20 + 'confidence = round(kronos_f["kronos_confidence"], 1)\n'

# Line 592: tradeable = False
lines[591] = ' ' * 16 + 'tradeable = False\n'

# Line 593: gate = "NO TRADE"
lines[592] = ' ' * 16 + 'gate = "NO TRADE"\n'

# Line 594: blank line - 8 spaces (method level)
lines[593] = ' ' * 8 + '\n'

# Line 595: return - 8 spaces (method level)
lines[594] = ' ' * 8 + 'return {"symbol": self.symbol, "date": pd.Timestamp.now().strftime("%Y-%m-%d"),\n'

# Lines 596-602: dict items - 12 spaces
for i in range(595, 602):
    if lines[i].startswith('            ') or lines[i].startswith('                '):
        lines[i] = ' ' * 12 + lines[i].lstrip()

with open('daily_kronos_pipeline.py', 'w') as f:
    f.writelines(lines)
print('Fixed')