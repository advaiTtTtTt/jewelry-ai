with open('src/backend/reconstruction/pipeline.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for idx, line in enumerate(lines):
    new_lines.append(line)
    if 'resolution=resolution,' in line:
        new_lines.append('                threshold=10.0,       # Isosurface threshold\n')
        new_lines.append('            )\n')
        new_lines.append('\n')
        new_lines.append('        self._offload_triposr()\n')
        new_lines.append('\n')
        new_lines.append('        # TripoSR returns a list of meshes (one per input image)\n')
        new_lines.append('        mesh = meshes[0]\n')
        new_lines.append('\n')

with open('src/backend/reconstruction/pipeline.py', 'w') as f:
    f.writelines(new_lines)
