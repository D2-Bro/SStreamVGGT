import json
import math
import os
import time

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable, Sequence, Set
from dataclasses import dataclass

from streamvggt.utils.cache_analysis import (
    CacheAnalysisConfig,
    EvictionNNAnalysisConfig,
    LeverageScoreHistogramConfig,
    ProjectedNormHistogramConfig,
    PreEvictionSnapshotConfig,
    TokenOverlayDumpConfig,
)
from streamvggt.layers.confidence_state import make_token_confidence_gate, pack_kv_cache, sample_token_confidence, unpack_kv_cache

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None


def _filter_grouped_camera_kv_cache(
    past_key_values_camera: List[Optional[Tuple]],
    cached_frame_ids: Sequence[int],
    keep_frame_ids: Sequence[int],
    tokens_per_frame_group: int,
) -> Tuple[List[Optional[Tuple]], List[int]]:
    """Keep camera-head KV groups whose frame IDs are active anchors."""
    tokens_per_frame_group = int(tokens_per_frame_group)
    if tokens_per_frame_group <= 0:
        raise ValueError(
            "tokens_per_frame_group must be positive, "
            f"got {tokens_per_frame_group}"
        )

    keep_set = {int(frame_id) for frame_id in keep_frame_ids}
    keep_group_indices = [
        group_idx
        for group_idx, frame_id in enumerate(cached_frame_ids)
        if int(frame_id) in keep_set
    ]
    filtered_frame_ids = [int(cached_frame_ids[idx]) for idx in keep_group_indices]
    if len(keep_group_indices) == len(cached_frame_ids):
        return past_key_values_camera, filtered_frame_ids

    filtered_kv = []
    token_indices_cache = {}
    for layer_kv in past_key_values_camera:
        if layer_kv is None:
            filtered_kv.append(None)
            continue
        if len(layer_kv) != 2:
            raise ValueError(f"Expected camera KV tuple of length 2, got {len(layer_kv)}")

        k_cache, v_cache = layer_kv[:2]
        total_tokens = k_cache.shape[2]
        expected_tokens = len(cached_frame_ids) * tokens_per_frame_group
        if total_tokens != expected_tokens:
            raise ValueError(
                "Camera KV cache length does not match tracked frame groups: "
                f"cache_tokens={total_tokens}, tracked_frames={len(cached_frame_ids)}, "
                f"tokens_per_frame_group={tokens_per_frame_group}"
            )

        device = k_cache.device
        token_indices = token_indices_cache.get(device)
        if token_indices is None:
            parts = [
                torch.arange(
                    group_idx * tokens_per_frame_group,
                    (group_idx + 1) * tokens_per_frame_group,
                    device=device,
                    dtype=torch.long,
                )
                for group_idx in keep_group_indices
            ]
            token_indices = torch.cat(parts, dim=0) if parts else torch.empty(0, device=device, dtype=torch.long)
            token_indices_cache[device] = token_indices

        filtered_k = k_cache.index_select(2, token_indices)
        filtered_v = v_cache.index_select(2, token_indices.to(device=v_cache.device))
        filtered_kv.append((filtered_k, filtered_v))

    return filtered_kv, filtered_frame_ids

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, total_budget=200000):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget

    def set_layer_budget_proportions(self, proportions_or_path):
        """Validate and install fixed layer-budget proportions."""
        source = proportions_or_path
        if isinstance(proportions_or_path, (str, os.PathLike)):
            path = os.fspath(proportions_or_path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    source = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Failed to load layer budget proportions from {path!r}: {exc}") from exc
        if isinstance(source, dict):
            declared_layers = source.get("num_layers")
            if declared_layers is not None and int(declared_layers) != self.aggregator.depth:
                raise ValueError(
                    "Layer budget JSON num_layers must match aggregator depth: "
                    f"got {declared_layers}, expected {self.aggregator.depth}"
                )
            if "proportions" not in source:
                raise ValueError("Layer budget proportions JSON must contain a 'proportions' field")
            source = source["proportions"]

        try:
            proportions = torch.as_tensor(source, dtype=torch.float32).detach().reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid layer budget proportions: {exc}") from exc
        if proportions.numel() != self.aggregator.depth:
            raise ValueError(
                "Layer budget proportions length must match aggregator depth: "
                f"got {proportions.numel()}, expected {self.aggregator.depth}"
            )
        if not torch.isfinite(proportions).all():
            raise ValueError("Layer budget proportions must contain only finite values")
        if (proportions < 0).any():
            raise ValueError("Layer budget proportions must be non-negative")
        total = proportions.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("Layer budget proportions must contain at least one positive value")

        normalized = proportions / total
        try:
            target_device = next(self.parameters()).device
        except (AttributeError, StopIteration):
            target_device = self.aggregator.last_scores.device
        self.aggregator.layer_budget_proportions = normalized.to(target_device)
        return self.aggregator.layer_budget_proportions

    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        global_attn_debug: bool = False,
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            global_attn_debug=global_attn_debug,
        )
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]

    def inference(
        self,
        frames,
        query_points: torch.Tensor = None,
        past_key_values=None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        stream_chunk_size: int = 1,
        history_anchor_strategy: str = "none",
        anchor_interval: int = 250,
        min_anchor_interval: Optional[int] = 100,
        window_protect_frames: int = 0,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        camera_motion_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        history_anchor_patch_topk_per_frame: int = 0,
        total_budget=None,
        budget_frame_multiplier: Optional[float] = None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        leverage_score_histogram_config: Optional[LeverageScoreHistogramConfig] = None,
        projected_norm_histogram_config: Optional[ProjectedNormHistogramConfig] = None,
        token_overlay_dump_config: Optional[TokenOverlayDumpConfig] = None,
        eviction_policy: str = "mean",
        eviction_policy_layers: Optional[Set[int]] = None,
        profile_eviction: bool = False,
        empty_cache_interval: int = 1,
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 256,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_normalize_before_projection: bool = False,
        leverage_normalize_before_projection_headwise: bool = False,
        leverage_projected_key_cache: bool = False,
        leverage_approx_method: str = "right_sketch_ridge",
        leverage_ridge_lambda: float = 0,
        leverage_ridge_lambda_mode: str = "absolute",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: Optional[int] = None,
        rls_refresh_interval: int = 1,
        leverage_diag: bool = False,
        leverage_diag_interval: int = 0,
        leverage_random_seed: int = 42,
        leverage_eviction_selector: str = "topk",
        leverage_conf_gate: bool = False,
        leverage_conf_gate_floor: float = 0.0,
        leverage_conf_gate_depth_alpha: float = 1.0,
        leverage_conf_gate_point_beta: float = 0.0,
        leverage_conf_gate_init: str = "mean",
        leverage_attention_utility: bool = False,
        leverage_attention_beta: float = 0.3,
        leverage_attention_ema_decay: float = 0.9,
        leverage_attention_freeze_updates: int = 5,
        leverage_attention_colsum_subsample_ratio: float = 1.0,
        leverage_conf_gate_k: float = 1.0,
        leverage_conf_gate_transform: str = "sigmoid",
        leverage_conf_gate_special_mode: str = "mean",
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.7,
        layer_budget_value_norm_type: str = "mean",
        layer_budget_norm_source: str = "key",
        layer_budget_alpha: float = 0.7,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 0,
        layer_budget_log_path: Optional[str] = None,
        global_attn_debug: bool = False,
    ):
        anchor_manager = None
        window_token_count = 0
        first_frame_special_tokens_only = False
        global_anchor_token_count = None
        global_anchor_special_only_active = False
        global_anchor_special_count = 0
        global_cache_history_anchor_special_tokens_only = False
        past_key_values = [None] * self.aggregator.depth
        self.aggregator.reset_stream_state()
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        camera_cache_frame_ids = []
        camera_cache_tokens_per_frame = None
        camera_cache_anchor_frame_ids = [0]
        total_budget = self.total_budget if total_budget is None else total_budget
        stream_chunk_size = int(stream_chunk_size)
        if stream_chunk_size < 1:
            raise ValueError(f"stream_chunk_size must be >= 1, got {stream_chunk_size}")
        empty_cache_interval = int(empty_cache_interval)
        if empty_cache_interval < 0:
            raise ValueError(f"empty_cache_interval must be >= 0, got {empty_cache_interval}")
        if stream_chunk_size > 1 and bool(global_cache_history_anchor_special_tokens_only):
            raise ValueError(
                "stream_chunk_size > 1 is not supported with "
                "global_cache_history_anchor_special_tokens_only=True"
            )
        history_anchor_patch_topk_per_frame = max(int(history_anchor_patch_topk_per_frame), 0)
        history_anchor_patch_topk_enabled = history_anchor_patch_topk_per_frame > 0 and int(max_anchors) > 0
        if budget_frame_multiplier is not None and float(budget_frame_multiplier) < 0.0:
            raise ValueError(
                "budget_frame_multiplier must be >= 0 when provided, "
                f"got {budget_frame_multiplier}"
            )

        img_h, img_w = 392, 518
        if len(frames) > 0 and "img" in frames[0]:
            sample_img = frames[0]["img"]
            if sample_img.dim() == 3:
                img_h, img_w = sample_img.shape[1], sample_img.shape[2]
            elif sample_img.dim() == 4:
                img_h, img_w = sample_img.shape[2], sample_img.shape[3]
        patch_size = self.aggregator.patch_size
        patch_h = img_h // patch_size
        patch_w = img_w // patch_size
        tokens_per_frame = 1 + 4 + patch_h * patch_w
        if budget_frame_multiplier is not None:
            per_layer_budget = int(math.ceil(float(budget_frame_multiplier) * int(tokens_per_frame)))
            num_global_layers = int(self.aggregator.depth)
            total_budget = per_layer_budget * num_global_layers
            print(
                "[StreamVGGT] Frame-multiple total budget: "
                f"budget_frame_multiplier={budget_frame_multiplier}, "
                f"tokens_per_frame={tokens_per_frame}, "
                f"per_layer_budget={per_layer_budget}, "
                f"num_global_layers={num_global_layers}, "
                f"effective_total_budget={total_budget}"
            )

        all_ress = []
        processed_frames = []
        inference_profile_start = time.perf_counter() if profile_eviction else 0.0
        stream_profile_totals = {}

        def _profile_sync():
            if profile_eviction and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _profile_start():
            if profile_eviction:
                _profile_sync()
                return time.perf_counter()
            return 0.0

        def _profile_record(name, start):
            if profile_eviction:
                _profile_sync()
                stream_profile_totals[name] = stream_profile_totals.get(name, 0.0) + (time.perf_counter() - start)

        def _frame_to_images(frame: dict, target_device: torch.device) -> torch.Tensor:
            frame_img = frame["img"].to(target_device, non_blocking=True)
            if frame_img.dim() == 3:
                return frame_img.unsqueeze(0).unsqueeze(0)
            if frame_img.dim() == 4:
                return frame_img.unsqueeze(1)
            if frame_img.dim() == 5:
                return frame_img
            raise ValueError(f"Expected frame image with 3, 4, or 5 dims, got {frame_img.shape}")

        for chunk_start in range(0, len(frames), stream_chunk_size):
            chunk_frames = frames[chunk_start:chunk_start + stream_chunk_size]
            chunk_size = len(chunk_frames)
            chunk_frame_ids = [chunk_start + offset for offset in range(chunk_size)]
            chunk_last_idx = chunk_frame_ids[-1]
            cache_write_current_frame = True
            cache_evict_current_frame = True

            profile_stage_start = _profile_start()
            target_device = next(self.parameters()).device
            image_parts = [_frame_to_images(frame, target_device) for frame in chunk_frames]
            batch_sizes = {int(part.shape[0]) for part in image_parts}
            if len(batch_sizes) != 1:
                raise ValueError(f"All frames in a stream chunk must have the same batch size, got {sorted(batch_sizes)}")
            images = torch.cat(image_parts, dim=1)

            _profile_record("input_prepare", profile_stage_start)

            profile_stage_start = _profile_start()
            aggregator_output = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=chunk_start,
                current_frame_ids=chunk_frame_ids,
                current_frame_idx=chunk_last_idx,
                total_budget=total_budget,
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_nn_analysis_config=eviction_nn_analysis_config,
                leverage_score_histogram_config=leverage_score_histogram_config,
                projected_norm_histogram_config=projected_norm_histogram_config,
                token_overlay_dump_config=token_overlay_dump_config,
                eviction_policy=eviction_policy,
                eviction_policy_layers=eviction_policy_layers,
                profile_eviction=profile_eviction,
                eviction_debug=eviction_debug,
                leverage_sketch_dim=leverage_sketch_dim,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                leverage_projection=leverage_projection,
                leverage_head_mean_dim=leverage_head_mean_dim,
                leverage_normalize_rows=leverage_normalize_rows,
                leverage_normalize_before_projection=leverage_normalize_before_projection,
                leverage_normalize_before_projection_headwise=leverage_normalize_before_projection_headwise,
                leverage_projected_key_cache=leverage_projected_key_cache,
                leverage_approx_method=leverage_approx_method,
                leverage_ridge_lambda=leverage_ridge_lambda,
                leverage_ridge_lambda_mode=leverage_ridge_lambda_mode,
                leverage_ridge_score_chunk_size=leverage_ridge_score_chunk_size,
                leverage_ridge_jitter=leverage_ridge_jitter,
                leverage_ridge_dim=leverage_ridge_dim,
                rls_refresh_interval=rls_refresh_interval,
                leverage_diag=leverage_diag,
                leverage_diag_interval=leverage_diag_interval,
                leverage_random_seed=leverage_random_seed,
                leverage_eviction_selector=leverage_eviction_selector,
                leverage_conf_gate=leverage_conf_gate,
                leverage_conf_gate_floor=leverage_conf_gate_floor,
                leverage_conf_gate_depth_alpha=leverage_conf_gate_depth_alpha,
                leverage_conf_gate_point_beta=leverage_conf_gate_point_beta,
                leverage_conf_gate_init=leverage_conf_gate_init,
                leverage_attention_utility=leverage_attention_utility,
                leverage_attention_beta=leverage_attention_beta,
                leverage_attention_ema_decay=leverage_attention_ema_decay,
                leverage_attention_freeze_updates=leverage_attention_freeze_updates,
                leverage_attention_colsum_subsample_ratio=leverage_attention_colsum_subsample_ratio,
                layer_budget_strategy=layer_budget_strategy,
                layer_budget_value_gamma=layer_budget_value_gamma,
                layer_budget_value_norm_type=layer_budget_value_norm_type,
                layer_budget_norm_source=layer_budget_norm_source,
                layer_budget_alpha=layer_budget_alpha,
                layer_budget_min_tokens=layer_budget_min_tokens,
                layer_budget_eps=layer_budget_eps,
                layer_budget_log_path=layer_budget_log_path,
                cache_write_current_frame=cache_write_current_frame,
                cache_evict_current_frame=cache_evict_current_frame,
            )
            _profile_record("aggregator_forward", profile_stage_start)

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            profile_heads_start = _profile_start()
            track_all = vis_all = track_conf_all = None
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    profile_stage_start = _profile_start()
                    pose_enc_list, past_key_values_camera = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                    current_camera_tokens_per_frame = len(pose_enc_list)
                    if camera_cache_tokens_per_frame is None:
                        camera_cache_tokens_per_frame = current_camera_tokens_per_frame
                    elif camera_cache_tokens_per_frame != current_camera_tokens_per_frame:
                        raise ValueError(
                            "Camera cache tokens per frame changed during inference: "
                            f"previous={camera_cache_tokens_per_frame}, "
                            f"current={current_camera_tokens_per_frame}"
                        )
                    camera_cache_frame_ids.extend(int(frame_id) for frame_id in chunk_frame_ids)
                    camera_pose_all = pose_enc_list[-1]
                    _profile_record("camera_head", profile_stage_start)

                if self.depth_head is not None:
                    profile_stage_start = _profile_start()
                    depth_all, depth_conf_all = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    _profile_record("depth_head", profile_stage_start)

                if self.point_head is not None:
                    profile_stage_start = _profile_start()
                    pts3d_all, pts3d_conf_all = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    _profile_record("point_head", profile_stage_start)

                if self.track_head is not None and query_points is not None:
                    profile_stage_start = _profile_start()
                    track_list, vis_all, conf_all = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track_all = track_list[-1]
                    track_conf_all = conf_all
                    query_points = track_all[:, -1]
                    _profile_record("track_head", profile_stage_start)

            _profile_record("heads_total", profile_heads_start)
            profile_stage_start = _profile_start()

            camera_pose_last = camera_pose_all[:, -1, :]
            depth_last = depth_all[:, -1]

            tokens_per_frame = int(aggregated_tokens[-1].shape[2])
            for local_idx, (global_frame_idx, frame) in enumerate(zip(chunk_frame_ids, chunk_frames)):
                camera_pose = camera_pose_all[:, local_idx, :]
                depth = depth_all[:, local_idx]
                depth_conf = depth_conf_all[:, local_idx]
                pts3d = pts3d_all[:, local_idx]
                pts3d_conf = pts3d_conf_all[:, local_idx]

                if leverage_conf_gate:
                    use_depth_conf = float(leverage_conf_gate_depth_alpha) != 0.0
                    use_point_conf = float(leverage_conf_gate_point_beta) != 0.0
                    token_depth_conf = None
                    token_point_conf = None
                    if use_depth_conf:
                        token_depth_conf = sample_token_confidence(
                            depth_conf,
                            images.shape[-2:],
                            tokens_per_frame,
                            self.aggregator.patch_start_idx,
                            self.aggregator.patch_size,
                        )
                    if use_point_conf:
                        token_point_conf = sample_token_confidence(
                            pts3d_conf if self.point_head is not None else None,
                            images.shape[-2:],
                            tokens_per_frame,
                            self.aggregator.patch_start_idx,
                            self.aggregator.patch_size,
                            batch_size=depth_conf.shape[0],
                            device=depth_conf.device,
                        )
                    if token_depth_conf is None and token_point_conf is None:
                        token_conf_gate = torch.ones(
                            (depth_conf.shape[0], tokens_per_frame),
                            device=depth_conf.device,
                            dtype=torch.float32,
                        )
                    else:
                        token_conf_gate = make_token_confidence_gate(
                            token_depth_conf,
                            token_point_conf,
                            floor=leverage_conf_gate_floor,
                            depth_alpha=leverage_conf_gate_depth_alpha,
                            point_beta=leverage_conf_gate_point_beta,
                            normalizer_k=leverage_conf_gate_k,
                            confidence_transform=leverage_conf_gate_transform,
                            preserve_prefix_tokens=self.aggregator.patch_start_idx,
                            prefix_token_mode=leverage_conf_gate_special_mode,
                        )
                    for layer_id, layer_kv in enumerate(past_key_values):
                        if layer_kv is None:
                            continue
                        k_cache, v_cache, confidence_state = unpack_kv_cache(layer_kv)
                        if confidence_state is None:
                            continue
                        confidence_state.update_frame_gate(global_frame_idx, token_conf_gate)
                        past_key_values[layer_id] = pack_kv_cache(k_cache, v_cache, confidence_state)

                profile_stage_start = _profile_start()
                res_gpu = {
                    "pts3d_in_other_view": pts3d,
                    "conf": pts3d_conf,
                    "depth": depth,
                    "depth_conf": depth_conf,
                    "camera_pose": camera_pose,
                    **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                    **(
                        {
                            "track": track_all[:, local_idx],
                            "vis": vis_all[:, local_idx],
                            "track_conf": track_conf_all[:, local_idx],
                        }
                        if query_points is not None and track_all is not None
                        else {}
                    ),
                }
                _profile_record("result_pack", profile_stage_start)
                profile_stage_start = _profile_start()

                res_cpu = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in res_gpu.items()
                }
                _profile_record("cpu_transfer", profile_stage_start)

                if frame_writer is not None:
                    profile_stage_start = _profile_start()
                    frame_writer(global_frame_idx, frame, res_cpu)
                    _profile_record("frame_writer", profile_stage_start)

                if cache_results:
                    profile_stage_start = _profile_start()
                    all_ress.append(res_cpu)
                    processed_frames.append(
                        {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv for nk, nv in frame.items()}
                    )
                    _profile_record("result_cache", profile_stage_start)

                if empty_cache_interval > 0 and int(global_frame_idx) % empty_cache_interval == 0:
                    profile_stage_start = _profile_start()
                    del res_gpu
                    torch.cuda.empty_cache()
                    _profile_record("empty_cache", profile_stage_start)
                else:
                    del res_gpu

            _profile_record("anchor_cache_update", profile_stage_start)

        if profile_eviction:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_profile_time = time.perf_counter() - inference_profile_start
            profile_summary = self.aggregator.format_eviction_profile_summary(
                total_inference_time=inference_profile_time,
                frame_count=len(frames),
                stream_profile_totals=stream_profile_totals,
            )
            if profile_summary:
                print(profile_summary)

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
        )