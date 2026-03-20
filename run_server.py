import sys
import os
import uvicorn

# Add current directory to sys.path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    try:
        from api.main import app
        print("Import successful. Starting uvicorn...")
        uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        import traceback
        with open("server_error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"Error occurred: {e}. Check server_error.log for details.")
        sys.exit(1)
