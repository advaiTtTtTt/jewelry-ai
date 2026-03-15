import sys
content = open('backend/api.py').read()
old_part = """        # Wait for other jobs to finish using the GPU
        JOBS[job_id]["message"] = "Waiting in queue for GPU..."
        async with gpu_lock:
            JOBS[job_id]["status"] = "running"
    
            # Load image
            JOBS[job_id]["message"] = "Loading image..."
            JOBS[job_id]["progress"] = 5
        image = Image.open(image_path).convert("RGB")"""
new_part = """        # Wait for other jobs to finish using the GPU
        JOBS[job_id]["message"] = "Waiting in queue for GPU..."

        await gpu_lock.acquire()
        try:
            JOBS[job_id]["status"] = "running"
    
            # Load image
            JOBS[job_id]["message"] = "Loading image..."
            JOBS[job_id]["progress"] = 5
            image = Image.open(image_path).convert("RGB")"""
if old_part in content:
    content = content.replace(old_part, new_part)
    content = content.replace("        # Save split GLB", "            # Save split GLB")
    # Actually wait, manual acquire/release is easier!
    open('backend/api.py', 'w').write(content)
else:
    print("Not found")
