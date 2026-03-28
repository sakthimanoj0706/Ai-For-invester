import sys

with open('d:/signal-ai/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(124, 212):
    if len(lines[i]) > 4 and lines[i].startswith('    '):
        lines[i] = lines[i][4:]

with open('d:/signal-ai/app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("done")
