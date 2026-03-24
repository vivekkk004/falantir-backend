import subprocess
import os

repo_path = r"c:\Users\vivek\OneDrive\画像\Desktop\BEProject\Shoplifting-Detection-main"
os.chdir(repo_path)

new_remote = "https://github.com/vivekkk004/falantir-backend.git"

with open("git_push_log.txt", "w") as f:
    f.write(f"Setting remote origin to {new_remote}\n")
    subprocess.run(["git", "remote", "set-url", "origin", new_remote], capture_output=True, text=True)
    
    f.write("Git Status:\n")
    proc = subprocess.run(["git", "status"], capture_output=True, text=True)
    f.write(proc.stdout + proc.stderr + "\n\n")
    
    f.write("Adding files...\n")
    subprocess.run(["git", "add", "."], capture_output=True, text=True)
    
    f.write("Committing...\n")
    subprocess.run(["git", "commit", "-m", "chore: prepare for Render deployment (new files + build fixes)"], capture_output=True, text=True)
    
    f.write("Pushing to origin main...\n")
    proc = subprocess.run(["git", "push", "origin", "main", "--force"], capture_output=True, text=True)
    f.write(proc.stdout + proc.stderr + "\n\n")
