import json
import math
import os

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
from streamvggt.utils.history_anchor import HistoryAnchorConfig, HistoryAnchorManager
from streamvggt.layers.confidence_state import make_token_confidence_gate, pack_kv_cache, sample_token_confidence, unpack_kv_cache
from streamvggt.layers.recent_merge import RecentMergeConfig, RecentSimilarityMerge
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig, VoxelCovisibilityGraph

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
        if len(layer_kv) not in (2, 3):
            raise ValueError(f"Expected camera KV tuple of length 2 or 3, got {len(layer_kv)}")

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
        if len(layer_kv) == 3:
            metadata = layer_kv[2]
            metadata_indices = token_indices.detach().cpu().view(1, 1, -1)
            metadata_indices = metadata_indices.expand(
                metadata.frame_ids.shape[0],
                metadata.frame_ids.shape[1],
                -1,
            )
            filtered_kv.append((filtered_k, filtered_v, metadata.gather(metadata_indices)))
        else:
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
        global_attn_idx_ranges: Optional[Any] = None,
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
            global_attn_idx_ranges=global_attn_idx_ranges,
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
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_projected_key_cache: bool = False,
        leverage_approx_method: str = "right_sketch",        leverage_ridge_lambda: float = 1e-3,
        leverage_ridge_lambda_mode: str = "relative",
        leverage_ridge_score_chunk_size: int = 4096,
        leverage_ridge_jitter: float = 1e-6,
        leverage_ridge_dim: Optional[int] = None,
        leverage_diag: bool = False,
        leverage_diag_interval: int = 0,
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_similarity_granularity: str = "layer",
        leverage_similarity_feature_projection: str = "raw",
        leverage_similarity_leverage_gamma: float = 1.0,
        leverage_eviction_risk_mode: str = "low_leverage",
        leverage_high_outlier_z: float = 3.0,
        leverage_dpp_candidate_multiplier: int = 2,
        leverage_dpp_greedy_block_size: int = 32,
        leverage_dpp_quality_beta: float = 1.0,
        leverage_dpp_diversity_beta: float = 1.0,
        leverage_dpp_feature_projection: str = "raw",
        leverage_dpp_recency_bonus: bool = False,
        leverage_dpp_recency_lambda: float = 0.2,
        leverage_dpp_recency_window: int = 5,
        leverage_dpp_recency_gate_power: float = 1.0,
        leverage_dpp_recency_debug: bool = False,
        leverage_conf_gate: bool = False,
        leverage_conf_gate_floor: float = 0.2,
        leverage_conf_gate_depth_alpha: float = 1.0,
        leverage_conf_gate_point_beta: float = 1.0,
        leverage_conf_gate_init: str = "mean",
        leverage_conf_gate_k: float = 1.0,
        leverage_conf_gate_special_mode: str = "mean",
        layer_budget_strategy: str = "uniform",
        layer_budget_value_gamma: float = 0.5,
        layer_budget_value_norm_type: str = "rms",
        layer_budget_norm_source: str = "value",
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_depth_mu: float = 0.5,
        layer_budget_depth_sigma: float = 0.2,
        layer_budget_depth_floor: float = 0.0,
        slots_per_direction: float = 4.0,
        hybrid_beta: float = 0.5,
        layer_budget_log_path: Optional[str] = None,
        eviction_protect_recent_frames: int = 0,
        eviction_protect_special_tokens: bool = False,
        eviction_protect_special_token_interval: int = 1,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_config: Optional[VoxelCovisConfig] = None,
        covis_log_fn: Optional[Callable[[str], None]] = None,
        global_attn_idx_ranges: Optional[Any] = None,
        global_attn_debug: bool = False,
        kf_interval: int = 1,
        evict_interval: int = 1,
        global_cache_history_anchor_special_tokens_only: bool = False,
        first_frame_special_tokens_only: bool = False,
        camera_cache_history_anchors_only: bool = False,
        camera_cache_keep_dropped_anchors: bool = False,
    ):
        past_key_values = [None] * self.aggregator.depth
        self.aggregator.reset_stream_state()
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        camera_cache_frame_ids = []
        camera_cache_tokens_per_frame = None
        camera_cache_anchor_frame_ids = [0]
        total_budget = self.total_budget if total_budget is None else total_budget
        kf_interval = max(int(kf_interval), 1)
        evict_interval = max(int(evict_interval), 1)
        eviction_protect_special_token_interval = int(eviction_protect_special_token_interval)
        history_anchor_patch_topk_per_frame = max(int(history_anchor_patch_topk_per_frame), 0)
        history_anchor_patch_topk_enabled = history_anchor_patch_topk_per_frame > 0 and int(max_anchors) > 0
        if budget_frame_multiplier is not None and float(budget_frame_multiplier) < 0.0:
            raise ValueError(
                "budget_frame_multiplier must be >= 0 when provided, "
                f"got {budget_frame_multiplier}"
            )
        if eviction_protect_special_token_interval < 1:
            raise ValueError(
                "eviction_protect_special_token_interval must be >= 1, "
                f"got {eviction_protect_special_token_interval}"
            )
        recent_merger = None
        run_recent_merge = recent_merge_config is not None and recent_merge_config.enabled
        svd_needs_geometry = (
            svd_eviction_merge_config is not None
            and svd_eviction_merge_config.enabled
            and eviction_policy == "svd_leverage"
            and svd_eviction_merge_config.geometry_gate == "voxel_neighbor"
        )
        if run_recent_merge or svd_needs_geometry:
            geometry_config = recent_merge_config if recent_merge_config is not None else RecentMergeConfig()
            if not geometry_config.enabled:
                geometry_config = RecentMergeConfig(
                    enabled=True,
                    voxel_size=geometry_config.voxel_size,
                    use_depth_confidence=geometry_config.use_depth_confidence,
                    debug=geometry_config.debug,
                )
            recent_merger = RecentSimilarityMerge(
                geometry_config,
                patch_start_idx=self.aggregator.patch_start_idx,
                patch_size=self.aggregator.patch_size,
            )
        voxel_covis_graph = None
        if voxel_covis_config is not None and voxel_covis_config.enabled:
            voxel_covis_graph = VoxelCovisibilityGraph(
                voxel_covis_config,
                patch_size=self.aggregator.patch_size,
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
        window_token_count = max(int(window_protect_frames), 0) * tokens_per_frame
        global_anchor_token_count = (
            min(int(self.aggregator.patch_start_idx), int(tokens_per_frame))
            if bool(first_frame_special_tokens_only)
            else int(tokens_per_frame)
        )

        anchor_manager = None
        if history_anchor_strategy != "none":
            anchor_config = HistoryAnchorConfig(
                strategy=history_anchor_strategy,
                interval=anchor_interval,
                min_anchor_interval=min_anchor_interval,
                max_anchors=max_anchors,
                coverage_threshold=coverage_threshold,
                camera_motion_threshold=camera_motion_threshold,
                history_protect_token_count=self.aggregator.patch_start_idx,
                global_protect_token_count=global_anchor_token_count,
                anchor_keep_ratio=anchor_keep_ratio,
                patch_topk_per_anchor_frame=history_anchor_patch_topk_per_frame,
            )
            anchor_manager = HistoryAnchorManager(anchor_config, tokens_per_frame)
            anchor_manager.image_size_hw = (img_h, img_w)

        global_anchor_special_only_active = (
            bool(global_cache_history_anchor_special_tokens_only)
            and anchor_manager is not None
        )
        global_anchor_special_count = min(int(self.aggregator.patch_start_idx), int(tokens_per_frame))

        def _select_history_anchor_token_indices(batch_size: int, device: torch.device) -> torch.Tensor:
            special_count = min(int(self.aggregator.patch_start_idx), int(tokens_per_frame))
            return torch.arange(special_count, device=device).unsqueeze(0).expand(batch_size, -1)

        all_ress = []
        processed_frames = [] 

        for i, frame in enumerate(frames):
            cache_write_current_frame = i == 0 or i % kf_interval == 0
            cached_keyframe_idx = i // kf_interval
            cache_evict_current_frame = (
                cache_write_current_frame
                and cached_keyframe_idx % evict_interval == 0
            )

            fixed_interval_registered = False
            fixed_interval_is_fifo = False
            if anchor_manager is not None and history_anchor_strategy == "fixed_interval":
                should_register, is_fifo, reason = anchor_manager.should_become_anchor(frame_idx=i)
                if should_register:
                    anchor_manager.register_anchor(i)
                    fixed_interval_registered = True
                    if int(i) not in camera_cache_anchor_frame_ids:
                        camera_cache_anchor_frame_ids.append(int(i))
                    fixed_interval_is_fifo = is_fifo
                    fifo_msg = (
                        " (leverage candidate pool)"
                        if history_anchor_patch_topk_enabled
                        else (" (FIFO: oldest demoted)" if is_fifo else "")
                    )
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

            anchor_token_count = (
                anchor_manager.get_protected_token_count()
                if anchor_manager is not None and not history_anchor_patch_topk_enabled
                else (global_anchor_token_count if bool(first_frame_special_tokens_only) else None)
            )
            effective_window_token_count = window_token_count if anchor_manager is not None else 0
            forward_anchor_token_count = anchor_token_count
            if (
                global_anchor_special_only_active
                and fixed_interval_registered
                and not fixed_interval_is_fifo
                and anchor_token_count is not None
            ):
                forward_anchor_token_count = max(
                    int(anchor_token_count) - int(global_anchor_special_count),
                    int(global_anchor_token_count),
                )
            frame_total_budget = total_budget
            if (
                global_anchor_special_only_active
                and cache_write_current_frame
                and i != 0
                and total_budget is not None
            ):
                frame_total_budget = max(int(total_budget) - int(global_anchor_special_count), 0)

            target_device = next(self.parameters()).device
            frame_img = frame["img"].to(target_device, non_blocking=True)
            if frame_img.dim() == 3:
                images = frame_img.unsqueeze(0).unsqueeze(0)
            elif frame_img.dim() == 4:
                images = frame_img.unsqueeze(0)
            elif frame_img.dim() == 5:
                images = frame_img
            else:
                raise ValueError(f"Expected frame image with 3, 4, or 5 dims, got {frame_img.shape}")
            covis_selection = None
            voxel_covis_frame_ids = None
            if voxel_covis_graph is not None:
                covis_selection = voxel_covis_graph.select_for_frame(i)
                voxel_covis_frame_ids = covis_selection.selected_frame_ids
                if voxel_covis_config.debug:
                    msg = _format_covis_debug(covis_selection, past_key_values)
                    if covis_log_fn is not None:
                        covis_log_fn(msg)
                    else:
                        print(msg)

            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i,
                total_budget=frame_total_budget,
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_nn_analysis_config=eviction_nn_analysis_config,
                leverage_score_histogram_config=leverage_score_histogram_config,
                projected_norm_histogram_config=projected_norm_histogram_config,
                token_overlay_dump_config=token_overlay_dump_config,
                eviction_policy=eviction_policy,
                eviction_policy_layers=eviction_policy_layers,
                eviction_debug=eviction_debug,
                leverage_sketch_dim=leverage_sketch_dim,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                leverage_projection=leverage_projection,
                leverage_head_mean_dim=leverage_head_mean_dim,
                leverage_normalize_rows=leverage_normalize_rows,
                leverage_projected_key_cache=leverage_projected_key_cache,
                leverage_approx_method=leverage_approx_method,                leverage_ridge_lambda=leverage_ridge_lambda,
                leverage_ridge_lambda_mode=leverage_ridge_lambda_mode,
                leverage_ridge_score_chunk_size=leverage_ridge_score_chunk_size,
                leverage_ridge_jitter=leverage_ridge_jitter,
                leverage_ridge_dim=leverage_ridge_dim,
                leverage_diag=leverage_diag,
                leverage_diag_interval=leverage_diag_interval,
                leverage_random_seed=leverage_random_seed,
                leverage_eviction_selector=leverage_eviction_selector,
                leverage_similarity_granularity=leverage_similarity_granularity,
                leverage_similarity_feature_projection=leverage_similarity_feature_projection,
                leverage_similarity_leverage_gamma=leverage_similarity_leverage_gamma,
                leverage_eviction_risk_mode=leverage_eviction_risk_mode,
                leverage_high_outlier_z=leverage_high_outlier_z,
                leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                leverage_dpp_quality_beta=leverage_dpp_quality_beta,
                leverage_dpp_diversity_beta=leverage_dpp_diversity_beta,
                leverage_dpp_feature_projection=leverage_dpp_feature_projection,
                leverage_dpp_recency_bonus=leverage_dpp_recency_bonus,
                leverage_dpp_recency_lambda=leverage_dpp_recency_lambda,
                leverage_dpp_recency_window=leverage_dpp_recency_window,
                leverage_dpp_recency_gate_power=leverage_dpp_recency_gate_power,
                leverage_dpp_recency_debug=leverage_dpp_recency_debug,
                leverage_conf_gate=leverage_conf_gate,
                leverage_conf_gate_floor=leverage_conf_gate_floor,
                leverage_conf_gate_depth_alpha=leverage_conf_gate_depth_alpha,
                leverage_conf_gate_point_beta=leverage_conf_gate_point_beta,
                leverage_conf_gate_init=leverage_conf_gate_init,
                layer_budget_strategy=layer_budget_strategy,
                layer_budget_value_gamma=layer_budget_value_gamma,
                layer_budget_value_norm_type=layer_budget_value_norm_type,
                layer_budget_norm_source=layer_budget_norm_source,
                layer_budget_alpha=layer_budget_alpha,
                layer_budget_min_tokens=layer_budget_min_tokens,
                layer_budget_eps=layer_budget_eps,
                layer_budget_depth_mu=layer_budget_depth_mu,
                layer_budget_depth_sigma=layer_budget_depth_sigma,
                layer_budget_depth_floor=layer_budget_depth_floor,
                slots_per_direction=slots_per_direction,
                hybrid_beta=hybrid_beta,
                layer_budget_log_path=layer_budget_log_path,
                eviction_protect_recent_frames=eviction_protect_recent_frames,
                eviction_protect_special_tokens=eviction_protect_special_tokens,
                eviction_protect_special_token_interval=eviction_protect_special_token_interval,
                anchor_token_count=forward_anchor_token_count,
                window_token_count=effective_window_token_count,
                recent_merge_config=recent_merge_config,
                svd_eviction_merge_config=svd_eviction_merge_config,
                voxel_covis_frame_ids=voxel_covis_frame_ids,
                voxel_covis_enabled=voxel_covis_graph is not None,
                global_attn_idx_ranges=global_attn_idx_ranges,
                global_attn_debug=global_attn_debug,
                cache_write_current_frame=cache_write_current_frame,
                cache_evict_current_frame=cache_evict_current_frame,
                global_cache_history_anchor_special_tokens_only=global_anchor_special_only_active,
                history_anchor_frame_ids=(
                    anchor_manager.history_anchor_frames
                    if history_anchor_patch_topk_enabled and anchor_manager is not None
                    else None
                ),
                history_anchor_patch_topk_per_frame=history_anchor_patch_topk_per_frame,
                history_anchor_max_frames=max_anchors,
            )

            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output        
            global_special_kv_sidecars = getattr(self.aggregator, "last_global_special_kv_sidecars", None)
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
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
                    camera_cache_frame_ids.append(i)
                    pose_enc = pose_enc_list[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            if fixed_interval_registered and not history_anchor_patch_topk_enabled:
                if global_anchor_special_only_active:
                    past_key_values = self.aggregator.sync_anchor_special_tokens_from_sidecars(
                        past_key_values,
                        global_special_kv_sidecars,
                        anchor_token_count=anchor_token_count,
                        tokens_per_frame=tokens_per_frame,
                        is_fifo=fixed_interval_is_fifo,
                        global_anchor_token_count=global_anchor_token_count,
                    )
                else:
                    anchor_token_indices = _select_history_anchor_token_indices(
                        camera_pose.shape[0],
                        camera_pose.device,
                    )
                    past_key_values = self.aggregator.sync_anchor_change(
                        past_key_values,
                        anchor_token_count=anchor_token_count,
                        tokens_per_frame=tokens_per_frame,
                        anchor_keep_ratio=anchor_keep_ratio,
                        anchor_token_indices=anchor_token_indices,
                        is_fifo=fixed_interval_is_fifo,
                        global_anchor_token_count=global_anchor_token_count,
                    )

            if anchor_manager is not None and history_anchor_strategy == "coverage":
                should_register, is_fifo, reason, coverage = anchor_manager.should_become_anchor_coverage(
                    frame_idx=i,
                    current_depth=depth[0],
                    current_pose=camera_pose[0],
                )
                if should_register:
                    anchor_manager.register_anchor_coverage(i, depth[0], camera_pose[0])
                    if int(i) not in camera_cache_anchor_frame_ids:
                        camera_cache_anchor_frame_ids.append(int(i))
                    fifo_msg = (
                        " (leverage candidate pool)"
                        if history_anchor_patch_topk_enabled
                        else (" (FIFO: oldest demoted)" if is_fifo else "")
                    )
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

                    if history_anchor_patch_topk_enabled:
                        pass
                    elif global_anchor_special_only_active:
                        past_key_values = self.aggregator.sync_anchor_special_tokens_from_sidecars(
                            past_key_values,
                            global_special_kv_sidecars,
                            anchor_token_count=anchor_manager.get_protected_token_count(),
                            tokens_per_frame=tokens_per_frame,
                            is_fifo=is_fifo,
                            global_anchor_token_count=global_anchor_token_count,
                        )
                    else:
                        anchor_token_indices = _select_history_anchor_token_indices(
                            camera_pose.shape[0],
                            camera_pose.device,
                        )
                        past_key_values = self.aggregator.sync_anchor_change(
                            past_key_values,
                            anchor_token_count=anchor_manager.get_protected_token_count(),
                            tokens_per_frame=tokens_per_frame,
                            anchor_keep_ratio=anchor_keep_ratio,
                            anchor_token_indices=anchor_token_indices,
                            is_fifo=is_fifo,
                            global_anchor_token_count=global_anchor_token_count,
                        )

            if anchor_manager is not None and history_anchor_strategy == "camera_motion":
                should_register, is_fifo, reason, _camera_motion = anchor_manager.should_become_anchor_camera_motion(
                    frame_idx=i,
                    current_pose=camera_pose[0],
                )
                if should_register:
                    anchor_manager.register_anchor_camera_motion(i, camera_pose[0])
                    if int(i) not in camera_cache_anchor_frame_ids:
                        camera_cache_anchor_frame_ids.append(int(i))
                    fifo_msg = (
                        " (leverage candidate pool)"
                        if history_anchor_patch_topk_enabled
                        else (" (FIFO: oldest demoted)" if is_fifo else "")
                    )
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

                    if history_anchor_patch_topk_enabled:
                        pass
                    elif global_anchor_special_only_active:
                        past_key_values = self.aggregator.sync_anchor_special_tokens_from_sidecars(
                            past_key_values,
                            global_special_kv_sidecars,
                            anchor_token_count=anchor_manager.get_protected_token_count(),
                            tokens_per_frame=tokens_per_frame,
                            is_fifo=is_fifo,
                            global_anchor_token_count=global_anchor_token_count,
                        )
                    else:
                        anchor_token_indices = _select_history_anchor_token_indices(
                            camera_pose.shape[0],
                            camera_pose.device,
                        )
                        past_key_values = self.aggregator.sync_anchor_change(
                            past_key_values,
                            anchor_token_count=anchor_manager.get_protected_token_count(),
                            tokens_per_frame=tokens_per_frame,
                            anchor_keep_ratio=anchor_keep_ratio,
                            anchor_token_indices=anchor_token_indices,
                            is_fifo=is_fifo,
                            global_anchor_token_count=global_anchor_token_count,
                        )

            if camera_cache_history_anchors_only and self.camera_head is not None:
                if camera_cache_keep_dropped_anchors:
                    keep_camera_frame_ids = list(camera_cache_anchor_frame_ids)
                else:
                    keep_camera_frame_ids = [0]
                    if anchor_manager is not None:
                        keep_camera_frame_ids.extend(anchor_manager.history_anchor_frames)
                keep_camera_frame_ids = list(dict.fromkeys(int(frame_id) for frame_id in keep_camera_frame_ids))
                past_key_values_camera, camera_cache_frame_ids = _filter_grouped_camera_kv_cache(
                    past_key_values_camera,
                    camera_cache_frame_ids,
                    keep_camera_frame_ids,
                    camera_cache_tokens_per_frame,
                )

            tokens_per_frame = int(aggregated_tokens[-1].shape[2])
            if leverage_conf_gate:
                token_depth_conf = sample_token_confidence(
                    depth_conf,
                    images.shape[-2:],
                    tokens_per_frame,
                    self.aggregator.patch_start_idx,
                    self.aggregator.patch_size,
                )
                token_point_conf = sample_token_confidence(
                    pts3d_conf if self.point_head is not None else None,
                    images.shape[-2:],
                    tokens_per_frame,
                    self.aggregator.patch_start_idx,
                    self.aggregator.patch_size,
                    batch_size=token_depth_conf.shape[0],
                    device=token_depth_conf.device,
                )
                token_conf_gate = make_token_confidence_gate(
                    token_depth_conf,
                    token_point_conf,
                    floor=leverage_conf_gate_floor,
                    depth_alpha=leverage_conf_gate_depth_alpha,
                    point_beta=leverage_conf_gate_point_beta,
                    normalizer_k=leverage_conf_gate_k,
                    preserve_prefix_tokens=self.aggregator.patch_start_idx,
                    prefix_token_mode=leverage_conf_gate_special_mode,
                )
                for layer_id, layer_kv in enumerate(past_key_values):
                    if layer_kv is None:
                        continue
                    k_cache, v_cache, metadata, confidence_state = unpack_kv_cache(layer_kv)
                    if confidence_state is None:
                        continue
                    confidence_state.update_frame_gate(i, token_conf_gate)
                    past_key_values[layer_id] = pack_kv_cache(k_cache, v_cache, metadata, confidence_state)

            if recent_merger is not None:
                geom = recent_merger.record_frame_geometry(
                    frame_id=i,
                    depth=depth,
                    depth_conf=depth_conf,
                    point_conf=pts3d_conf if self.point_head is not None else None,
                    pose_enc=camera_pose,
                    image_hw=images.shape[-2:],
                    tokens_per_frame=tokens_per_frame,
                )
                if geom is not None:
                    for layer_id, layer_kv in enumerate(past_key_values):
                        if layer_kv is None:
                            continue
                        k_cache, v_cache, metadata, confidence_state = unpack_kv_cache(layer_kv)
                        if metadata is None:
                            continue
                        recent_merger.update_metadata_for_frame(metadata, i)
                        if run_recent_merge:
                            k_cache, v_cache, metadata, _ = recent_merger.merge_layer(
                                k_cache,
                                v_cache,
                                metadata,
                                layer_id=layer_id,
                                frame_id=i,
                            )
                        past_key_values[layer_id] = pack_kv_cache(k_cache, v_cache, metadata, confidence_state)

            if voxel_covis_graph is not None:
                voxel_covis_graph.record_frame_geometry(
                    frame_id=i,
                    depth=depth,
                    depth_conf=depth_conf,
                    pose_enc=camera_pose,
                    image_hw=images.shape[-2:],
                )

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None
                    else {}
                ),
            }
            res_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in res_gpu.items()
            }
            if frame_writer is not None:
                frame_writer(i, frame, res_cpu)

            if cache_results:
                all_ress.append(res_cpu)
                processed_frames.append(
                    {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv for nk, nv in frame.items()}
                )

            del res_gpu
            torch.cuda.empty_cache()

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
        )


def _format_covis_debug(selection, past_key_values) -> str:
    cache_before, selected_stats, per_frame_stats = _covis_cache_stats(
        past_key_values,
        selection.selected_frame_ids,
    )
    score_summary = "none"
    if selection.scores:
        shared = [score.shared_voxels for score in selection.scores]
        overlap = [score.overlap_ratio for score in selection.scores]
        score_summary = (
            f"candidates={len(selection.scores)} "
            f"shared_min/max={min(shared)}/{max(shared)} "
            f"overlap_min/max={min(overlap):.4f}/{max(overlap):.4f}"
        )
    return (
        "[voxel-covis] "
        f"frame={selection.current_frame_id} "
        f"reference={selection.reference_frame_id} "
        f"retrieved_frames={selection.selected_frame_ids} "
        f"selected_count={len(selection.selected_frame_ids)} "
        f"fallback={selection.fallback_used} "
        f"{score_summary} "
        f"cache_tokens_before={cache_before} "
        f"selected_key_cache_tokens={selected_stats} "
        f"retrieved_key_cache_by_frame={per_frame_stats}"
    )


def _covis_cache_stats(past_key_values, selected_frame_ids) -> tuple[int, str, str]:
    for layer_kv in past_key_values:
        if layer_kv is None:
            continue
        k_cache, _, metadata, _ = unpack_kv_cache(layer_kv)
        if metadata is None:
            continue
        selected = torch.as_tensor(list(selected_frame_ids), dtype=torch.long)
        counts = []
        per_frame = {}
        for b in range(metadata.frame_ids.shape[0]):
            for h in range(metadata.frame_ids.shape[1]):
                frame_ids = metadata.frame_ids[b, h]
                if selected.numel() == 0:
                    count = 0
                else:
                    count = int(torch.isin(frame_ids, selected).sum().item())
                counts.append(count)
                for frame_id in selected_frame_ids:
                    per_frame.setdefault(int(frame_id), []).append(
                        int((frame_ids == int(frame_id)).sum().item())
                    )
        if not counts:
            return int(k_cache.shape[2]), "none", "none"
        counts_tensor = torch.tensor(counts, dtype=torch.float32)
        per_frame_parts = []
        for frame_id in selected_frame_ids:
            values = per_frame.get(int(frame_id), [])
            if not values:
                per_frame_parts.append(f"{int(frame_id)}:0/0.0/0")
                continue
            values_tensor = torch.tensor(values, dtype=torch.float32)
            per_frame_parts.append(
                f"{int(frame_id)}:"
                f"{int(values_tensor.min().item())}/"
                f"{float(values_tensor.mean().item()):.1f}/"
                f"{int(values_tensor.max().item())}"
            )
        return int(k_cache.shape[2]), (
            f"min/mean/max={int(counts_tensor.min().item())}/"
            f"{float(counts_tensor.mean().item()):.1f}/"
            f"{int(counts_tensor.max().item())}"
        ), "{" + ", ".join(per_frame_parts) + "}"
    return 0, "none", "none"
