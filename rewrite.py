with open("backend/api.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_func = False
for i, line in enumerate(lines):
    if line.startswith("async def _run_conversion"):
        in_func = True
        
    if in_func and line.startswith("# ───"):
        in_func = False
        
    if in_func:
        pass
        
    new_lines.append(line)
