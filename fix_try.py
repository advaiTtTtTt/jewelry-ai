with open("backend/api.py", "r") as f:
    text = f.read()

bad_block = """        await gpu_lock.acquire()
        try:
            JOBS[job_id]["status"] = "running"
    
            # Load image
            JOBS[job_id]["message"] = "Loading image..."
            JOBS[job_id]["progress"] = 5
            image = Image.open(image_path).convert("RGB")"""

good_block = """        await gpu_lock.acquire()
        JOBS[job_id]["status"] = "running"

        # Load image
        JOBS[job_id]["message"] = "Loading image..."
        JOBS[job_id]["progress"] = 5
        image = Image.open(image_path).convert("RGB")"""

if bad_block in text:
    new_text = text.replace(bad_block, good_block)
    with open("backend/api.py", "w") as f2:
        f2.write(new_text)
    print("Fixed!")
else:
    print("Not found!")
