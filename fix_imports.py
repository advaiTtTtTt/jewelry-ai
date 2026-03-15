import re

# Fix ring_builder.py
with open("src/backend/reconstruction/ring_builder.py", "r") as f:
    text = f.read()
text = text.replace("                import trimesh.smoothing\n", "")
with open("src/backend/reconstruction/ring_builder.py", "w") as f:
    f.write(text)

# Fix pipeline.py 
with open("src/backend/reconstruction/pipeline.py", "r") as f:
    text = f.read()
text = text.replace("        import trimesh\n", "")
with open("src/backend/reconstruction/pipeline.py", "w") as f:
    f.write(text)

print("Imports fixed.")
