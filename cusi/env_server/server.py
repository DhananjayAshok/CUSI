"""
The HTTP layer: JSON in and out, a bearer token on every request.

    Every request carries the server's token, or a session token (returned when a session is opened) that reaches only
    /sessions/<that id>/...

    GET    /health                     {"env", "kind"}
    GET    /tasks                      dev: this env's train + val tasks [{task_id, split, text}]; test: []
    GET    /scenes                     dev: free-play scene names
    POST   /sessions                   {"task_id" | "scene", "state": bool} -> session description + first obs
    GET    /sessions/<id>              the session's description
    POST   /sessions/<id>/reset        back to the start -> obs
    POST   /sessions/<id>/step         {"action"} -> obs, reward, done, success (when the episode ends)
    POST   /sessions/<id>/observe      the current obs (Android re-reads the device)
    GET    /sessions/<id>/context      Android: foreground app and device time
    POST   /sessions/<id>/save         dev: -> {"handle"}
    POST   /sessions/<id>/load         dev: {"handle"} -> obs
    DELETE /sessions/<id>              close
"""
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from cusi.utils.log_handling import log_info, log_warn
from cusi.env_server.codec import encode
from cusi.env_server.sessions import SessionError, Sessions


def make_handler(*, sessions: Sessions, token: str, parameters: dict):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):     # requests are not logged one by one
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _route(self, method: str) -> None:
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            given = self.headers.get("Authorization") or ""
            master = given == f"Bearer {token}"
            # a session token reaches only its own session (not /tasks, /scenes, other sessions or opening one)
            own_token = sessions.token_of(session_id=parts[1]) if len(parts) >= 2 and parts[0] == "sessions" else None
            own = own_token is not None and given == f"Bearer {own_token}"
            if not (master or own):
                return self._send(401, {"error": "bad or missing token"})
            try:
                body = self._body() if method in ("POST",) else {}
                out = self._dispatch(method=method, parts=parts, body=body)
                self._send(200, out)
            except SessionError as e:
                self._send(e.status, {"error": str(e)})
            except Exception as e:
                log_warn(f"[env_server] {method} {self.path}: {traceback.format_exc()[-1500:]}",
                         parameters=parameters)
                self._send(500, {"error": f"{type(e).__name__}: {str(e)[:500]}"})

        def _dispatch(self, *, method: str, parts: list, body: dict) -> dict:
            p = sessions.policy
            if parts == ["health"] and method == "GET":
                return {"env": sessions.backend.env, "kind": p.kind}
            if parts == ["tasks"] and method == "GET":
                return {"tasks": p.listed(env=sessions.backend.env, parameters=parameters)}
            if parts == ["scenes"] and method == "GET":
                return {"scenes": sorted(sessions.backend.free_play_scenes()) if p.free_play else []}
            if parts == ["sessions"] and method == "POST":
                session = sessions.open(task_id=body.get("task_id"), scene=body.get("scene"),
                                        want_state=bool(body.get("state", False)))
                try:
                    with session.lock:
                        return encode({**session.describe(), **session.reset(), "session_token": session.token})
                except Exception:
                    sessions.close(session_id=session.id)     # never leave a half-opened session holding a slot
                    raise
            if len(parts) >= 2 and parts[0] == "sessions":
                sid, action = parts[1], (parts[2] if len(parts) > 2 else None)
                if method == "DELETE" and action is None:
                    sessions.close(session_id=sid)
                    return {"closed": sid}
                session = sessions.get(session_id=sid)
                with session.lock:
                    if method == "GET" and action is None:
                        return encode(session.describe())
                    if method == "GET" and action == "context":
                        return session.context()
                    if method == "POST" and action == "reset":
                        return session.reset()
                    if method == "POST" and action == "step":
                        if not isinstance(body.get("action"), str):
                            raise SessionError('step takes {"action": "<text>"}')
                        return session.step(action=body["action"])
                    if method == "POST" and action == "observe":
                        return session.observe()
                    if method == "POST" and action == "save":
                        return session.save()
                    if method == "POST" and action == "load":
                        return session.load(handle=str(body.get("handle")))
            raise SessionError(f"no route {method} /{'/'.join(parts)}", 404)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


def serve(*, sessions: Sessions, host: str, port: int, token: str, parameters: dict) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(sessions=sessions, token=token, parameters=parameters))
    server.daemon_threads = True
    log_info(f"[env_server] {sessions.backend.env} ({sessions.policy.kind}) on http://{host}:{port}",
             parameters=parameters)
    try:
        server.serve_forever()
    finally:
        sessions.close_all()
        sessions.backend.close()
