from flask import Flask, request, Response
import requests

app = Flask(__name__)

# The address of web server (VM 2)
BACKEND_SERVER = "http://192.168.100.3:8080"

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def proxy(path):
        try:
                resp = requests.get(f"{BACKEND_SERVER}/{path}", timeout=5)

                excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                headers = [(name, value) for (name, value) in resp.raw.headers.items()
                        if name.lower() not in excluded_headers]

                return Response(resp.content, resp.status_code, headers)
        except requests.exceptions.RequestException as e:
                return f"Proxy Error: Could not connect to backend. {e}", 502

if __name__ == '__main__':
        # Run the proxy on port 80
        app.run(host='0.0.0.0', port=80)