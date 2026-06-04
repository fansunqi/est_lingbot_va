# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from .logging import logger
from .Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer


class DistributedModelWrapper:
    """
    TODO
    """

    def __init__(self, model, source_rank=0):
        self.model = model
        self.source_rank = source_rank

    def infer(self, obs):
        return distributed_infer(self.model, obs, self.source_rank)

    def on_session_closed(self, session_id):
        """Forward session cleanup to the underlying model."""
        distributed_session_closed(self.model, session_id, self.source_rank)


def distributed_infer(model, obs, source_rank):
    """
    TODO
    """
    rank = dist.get_rank()
    assert rank == source_rank, "distributed_infer can only run on the source rank"

    cmd = torch.tensor(1,
                       dtype=torch.int64,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.broadcast(cmd, src=source_rank)

    obj_list = [obs]
    dist.broadcast_object_list(obj_list, src=source_rank)

    result = model.infer(obs)

    return result


def distributed_session_closed(model, session_id, source_rank):
    rank = dist.get_rank()
    assert rank == source_rank, "distributed_session_closed can only run on the source rank"

    cmd = torch.tensor(2,
                       dtype=torch.int64,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.broadcast(cmd, src=source_rank)

    obj_list = [session_id]
    dist.broadcast_object_list(obj_list, src=source_rank)

    if hasattr(model, 'on_session_closed'):
        model.on_session_closed(session_id)


def worker_loop(model, local_rank):
    """
    TODO
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rank = dist.get_rank()

    while True:
        cmd = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == -1:
            break
        elif cmd_val == 1:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            obs = obj_list[0]
            _ = model.infer(obs)
        elif cmd_val == 2:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            session_id = obj_list[0]
            if hasattr(model, 'on_session_closed'):
                model.on_session_closed(session_id)
        else:
            pass

    logger.info("[worker_loop] rank=%s exiting.", rank)


def run_async_server_mode(model, local_rank, host, port):
    dist_ready = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_ready else int(getattr(model, "rank", local_rank))
    logger.info("Running in ASYNC SERVER mode: rank=%s", rank)
    if not dist_ready:
        model_server = WebsocketPolicyServer(model, host=host, port=port)
        model_server.serve_forever()
        return

    # Replicated (DDP-style) mode: each rank holds a full model copy and serves
    # its own port independently. The FSDP-style rank0-broadcast / worker_loop
    # pattern doesn't apply — every rank runs its own websocket loop. Gradient
    # sync happens later, inside the training update collective.
    if not getattr(model, 'fsdp_enabled', True):
        logger.info(
            "Replicated server: rank=%s binding %s:%s", rank, host, port,
        )
        model_server = WebsocketPolicyServer(model, host=host, port=port)
        model_server.serve_forever()
        return

    if rank == 0:
        dist_model = DistributedModelWrapper(model, source_rank=0)
        model_server = WebsocketPolicyServer(dist_model, host=host, port=port)
        model_server.serve_forever()

        cmd = torch.tensor(
            -1,
            dtype=torch.int64,
            device='cuda' if torch.cuda.is_available() else 'cpu')
        dist.broadcast(cmd, src=0)
    else:
        try:
            worker_loop(model, local_rank)
        except KeyboardInterrupt:
            logger.info("Rank shutting down: rank=%s local_rank=%s", rank, local_rank)
