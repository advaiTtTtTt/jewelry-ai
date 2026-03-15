with open("backend/api.py", "r") as f:
    text = f.read()

import re

# 1. Add acquired_lock = False at start
text = text.replace("""    try:
        from PIL import Image

        # Wait for other jobs to finish using the GPU""", """    acquired_lock = False
    try:
        from PIL import Image

        # Wait for other jobs to finish using the GPU""")

# 2. Add acquired_lock = True after acquire
text = text.replace("""        await gpu_lock.acquire()
        JOBS[job_id]["status"] = "running\"""", """        await gpu_lock.acquire()
        acquired_lock = True
        JOBS[job_id]["status"] = "running\"""")

# 3. Change finally block
text = text.replace("""    finally:
        if gpu_lock.locked():
            gpu_lock.release()""", """    finally:
        if acquired_lock:
            gpu_lock.release()""")

with open("backend/api.py", "w") as f2:
    f2.write(text)
