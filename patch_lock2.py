import re

with open("backend/api.py", "r") as f:
    text = f.read()

# Fix the lock logic:
def patch():
    old = text
    # remove the bad `async with gpu_lock:` block and just replace it with acquire
    s1 = r'''        # Wait for other jobs to finish using the GPU
        JOBS\[job_id\]\["message"\] = "Waiting in queue for GPU..."
        async with gpu_lock:
            JOBS\[job_id\]\["status"\] = "running"
    
            # Load image
            JOBS\[job_id\]\["message"\] = "Loading image..."
            JOBS\[job_id\]\["progress"\] = 5
        image = Image.open\(image_path\).convert\("RGB"\)'''
    
    r1 = '''        # Wait for other jobs to finish using the GPU
        JOBS[job_id]["message"] = "Waiting in queue for GPU..."
        await gpu_lock.acquire()
        try:
            JOBS[job_id]["status"] = "running"
    
            # Load image
            JOBS[job_id]["message"] = "Loading image..."
            JOBS[job_id]["progress"] = 5
            image = Image.open(image_path).convert("RGB")'''
            
    res = re.sub(s1, r1, old)
    
    # Now we need to release the lock in a finally block at the end of _run_conversion
    # we'll find the except Exception as e: block
    s2 = '''        logger.exception("[%s] Conversion failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["message"] = f"Error: {str(e)}"'''
        
    r2 = '''        logger.exception("[%s] Conversion failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["message"] = f"Error: {str(e)}"
    finally:
        if gpu_lock.locked():
            gpu_lock.release()'''
            
    res = res.replace("        logger.exception(\"[%s] Conversion failed\", job_id)", "        logger.exception(\"[%s] Conversion failed\", job_id)")
    
    new_res = res.replace(
"""        logger.exception("[%s] Conversion failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["message"] = f"Error: {str(e)}\"""",
"""        logger.exception("[%s] Conversion failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["message"] = f"Error: {str(e)}"
    finally:
        if gpu_lock.locked():
            gpu_lock.release()""")
            
    with open("backend/api.py", "w") as f2:
        f2.write(new_res)
        
patch()
