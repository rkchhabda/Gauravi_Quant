with open('kronos_high_accuracy_advisor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix MarketMoodEngine class structure
# Line 60 (0-indexed 59): blank line with indent 5 -> should be empty or removed
# Line 74 (0-indexed 73): @classmethod should be indent 4
# Line 75 (0-indexed 74): def compute should be indent 4 (method def), body at 8
# Line 78 (0-indexed 77): return cls._neutral() should be inside if block (indent 12)

lines[59] = '\n'  # blank line at module level
lines[73] = '    @classmethod\n'
lines[74] = '    def compute(cls, symbol: str) -> dict:\n'
lines[77] = '            return cls._neutral()\n'  # inside if block

with open('kronos_high_accuracy_advisor.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed MarketMoodEngine class structure')