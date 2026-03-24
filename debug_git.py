import subprocess
import os

repo_path = r"c:\Users\vivek\OneDrive\画像\Desktop\BEProject\Shoplifting-Detection-main"
os.chdir(repo_path)

with open("git_output.txt", "w") as f:
    f.write("REMOTE:\n")
    proc = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    f.write(proc.stdout + proc.stderr + "\n\n")
    
    f.write("LOG:\n")
    proc = subprocess.run(["git", "log", "--oneline", "-n", "5"], capture_output=True, text=True)
    f.write(proc.stdout + proc.stderr + "\n\n")
    
    f.write("STATUS:\n")
    proc = subprocess.run(["git", "status"], capture_output=True, text=True)
    f.write(proc.stdout + proc.stderr + "\n\n")
