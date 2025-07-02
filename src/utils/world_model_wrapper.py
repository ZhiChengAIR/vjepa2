import numpy as np
import torch.nn.functional as F
import importlib


class WorldModel(object):
    def __init__(
        self,
        encoder,
        predictor,
        tokens_per_frame,
        rotation_transformer=None,
        mpc_args={
            "rollout": 2,
            "samples": 400,
            "topk": 10,
            "cem_steps": 10,
            "momentum_mean": 0.15,
            "momentum_std": 0.15,
            "maxnorm": 0.05,
            "verbose": True,
        },
        normalize_reps=True,
        device="cuda:0",
        app=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.tokens_per_frame = tokens_per_frame
        self.device = device
        self.mpc_args = mpc_args
        mcp_utils = importlib.import_module(f"app.{app}.mpc_utils")
        self.cem = mcp_utils.cem
        self.compute_new_pose = mcp_utils.compute_new_pose
        self.rotation_transformer = rotation_transformer
        pass

    def encode(self, clip):
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device, non_blocking=True)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def infer_next_action(self, rep, pose, goal_rep):

        def step_predictor(reps, actions, poses):
            B, T, N_T, D = reps.size()
            reps = reps.flatten(1, 2)
            next_rep = self.predictor(reps, actions, poses)[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
            next_rep = next_rep.view(B, 1, N_T, D)
            next_pose = self.compute_new_pose(poses[:, -1:], actions[:, -1:], self.rotation_transformer)
            return next_rep, next_pose

        mpc_action = self.cem(
            context_frame=rep,
            context_pose=pose,
            goal_frame=goal_rep,
            world_model=step_predictor,
            **self.mpc_args,
        )[0]

        return mpc_action