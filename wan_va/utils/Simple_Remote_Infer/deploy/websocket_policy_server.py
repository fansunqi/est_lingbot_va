import asyncio
import http
import logging
import time
import traceback
import itertools

import websockets.asyncio.server as _server
import websockets.frames

from .msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy over websocket, supporting multiple concurrent clients.

    Each connection is assigned a unique session_id, injected into every request
    so the backend can maintain per-session KV caches. An asyncio Lock serializes
    GPU access (one infer at a time).
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._infer_lock = None  # created inside event loop
        self._session_counter = itertools.count()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._infer_lock = asyncio.Lock()
        async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
                ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        session_id = f"session_{next(self._session_counter)}"
        logger.info(f"Connection from {websocket.remote_address} opened (session={session_id})")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                # Inject session_id for per-session KV cache management
                obs['_session_id'] = session_id

                # Serialize GPU access across all connections
                async with self._infer_lock:
                    infer_time = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"][
                        "prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(
                    f"Connection from {websocket.remote_address} closed (session={session_id})")
                # Notify policy to free this session's resources
                try:
                    if hasattr(self._policy, 'on_session_closed'):
                        self._policy.on_session_closed(session_id)
                except Exception:
                    pass
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason=
                    "Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection,
                  request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
