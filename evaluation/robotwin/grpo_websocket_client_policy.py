from typing import Dict, Optional, Tuple

import websockets.sync.client

from evaluation.robotwin.msgpack_numpy import Packer, unpackb


class GRPOWebsocketClientPolicy:
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=10,
        )
        metadata = unpackb(conn.recv())
        return conn, metadata

    def _send(self, payload: Dict) -> Dict:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in GRPO server:\n{response}")
        return unpackb(response)

    def reset_episode(self, *, prompt: str, task: str, seed: int, group_id: str, **metadata) -> Dict:
        payload = {
            "command": "reset_episode",
            "prompt": prompt,
            "task": task,
            "seed": seed,
            "group_id": group_id,
        }
        payload.update(metadata)
        return self._send(payload)

    def sample_action(self, obs: Dict, *, prompt: str, **metadata) -> Dict:
        payload = {"command": "sample_action", "prompt": prompt}
        payload.update(obs)
        payload.update(metadata)
        return self._send(payload)

    def commit_chunk(self, *, obs, state) -> Dict:
        return self._send({
            "command": "commit_chunk",
            "obs": obs,
            "state": state,
            "compute_kv_cache": True,
        })

    def finish_episode(self, *, success: bool, step_count: int, **metadata) -> Dict:
        step_count_int = int(step_count)
        if success:
            reward = 1.0 + 20.0 / max(step_count_int, 20)
        else:
            reward = 0.0
        payload = {
            "command": "finish_episode",
            "success": bool(success),
            "reward": reward,
            "step_count": step_count_int,
        }
        payload.update(metadata)
        return self._send(payload)

    def get_status(self) -> Dict:
        return self._send({"command": "get_status"})

    def run_pending_updates(self) -> Dict:
        return self._send({"command": "run_pending_updates"})

    def get_eval_phase(self) -> Dict:
        return self._send({"command": "get_eval_phase"})

    def end_eval_phase(self) -> Dict:
        return self._send({"command": "end_eval_phase"})

    def save_checkpoint(self) -> Dict:
        return self._send({"command": "save_checkpoint"})

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
