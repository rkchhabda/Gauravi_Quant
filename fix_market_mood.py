with open('kronos_high_accuracy_advisor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix MarketMoodEngine class structure
# Line 74 (0-indexed 73): @classmethod should be indent 4
# Line 75 (0-indexed 74): def compute should be indent 8  
# Line 78 (0-indexed 77): return cls._neutral() should be indent 8
# Line 79 (0-indexed 78): close = df["close"] should be indent 8
# Line 82 (0-indexed 81): # EMAs comment should be indent 8
# Line 86 (0-indexed 85): # RSI(14) comment should be indent 8

lines[73] = '        @classmethod\n'
lines[74] = '    def compute(cls, symbol: str) -> dict:\n'
lines[77] = '        return cls._neutral()\n'
lines[78] = '        close = df["close"]\n'
lines[81] = '        # EMAs\n'
lines[85] = '        # RSI(14)\n'

with open('kronos_high_accuracy_advisor.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed MarketMoodEngine class')