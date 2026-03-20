import urllib.request
import json

data = json.dumps({
    "name": "vivek",
    "email": "test_py@gmail.com",
    "password": "Pass@123"
}).encode('utf-8')

req = urllib.request.Request("http://localhost:8000/api/auth/register", data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as response:
        print("SUCCESS:", response.read().decode())
except urllib.error.HTTPError as e:
    print("STATUS:", e.code)
    print("BODY:", e.read().decode())
except Exception as e:
    print("ERROR:", str(e))
