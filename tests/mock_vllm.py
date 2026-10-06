"""A tiny OpenAI-compatible chat server for smoke tests: always answers with a fixed reply."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def start_mock_vllm(*, reply: str, prompt_tokens: int = 11, completion_tokens: int = 7):
    """Serve on a free local port in a thread; returns (server, base_url, requests), shut down with server.shutdown()."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body})
            data = json.dumps({
                "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1", requests


if __name__ == "__main__":
    import sys
    import time

    _server, base_url, _ = start_mock_vllm(reply=sys.argv[1])
    print(base_url, flush=True)
    while True:
        time.sleep(3600)
