import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable
from dataclasses import dataclass

from streamvggt.utils.cache_analysis import CacheAnalysisConfig, EvictionNNAnalysisConfig, PreEvictionSnapshotConfig
from streamvggt.utils.history_anchor import HistoryAnchorConfig, HistoryAnchorManager
from streamvggt.layers.recent_merge import RecentMergeConfig, RecentSimilarityMerge
from streamvggt.layers.svd_eviction_merge import SvdEvictionMergeConfig
from streamvggt.layers.voxel_covis import VoxelCovisConfig, VoxelCovisibilityGraph

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, total_budget=1200000):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget
    


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
        history_anchor_strategy: str = "coverage",
        anchor_interval: int = 250,
        min_anchor_interval: Optional[int] = 100,
        window_protect_frames: int = 0,
        max_anchors: int = 3,
        coverage_threshold: float = 0.2,
        anchor_keep_ratio: float = 0.05,
        total_budget=None,
        cache_analysis_config: Optional[CacheAnalysisConfig] = None,
        pre_eviction_snapshot_config: Optional[PreEvictionSnapshotConfig] = None,
        eviction_nn_analysis_config: Optional[EvictionNNAnalysisConfig] = None,
        eviction_policy: str = "mean",
        eviction_debug: bool = False,
        leverage_sketch_dim: Optional[int] = 16,
        leverage_granularity: str = "head",
        leverage_feature: str = "key",
        leverage_projection: str = "random",
        leverage_head_mean_dim: int = 1,
        leverage_normalize_rows: bool = False,
        leverage_approx_method: str = "right_sketch",
        leverage_left_sketch_dim: Optional[int] = 2048,
        leverage_right_jl_dim: Optional[int] = 64,
        leverage_random_seed: int = 0,
        leverage_eviction_selector: str = "topk",
        leverage_dpp_candidate_multiplier: int = 2,
        leverage_dpp_greedy_block_size: int = 32,
        layer_budget_strategy: str = "uniform",
        layer_budget_alpha: float = 0.5,
        layer_budget_min_tokens: int = 0,
        layer_budget_eps: float = 1e-12,
        layer_budget_debug: bool = False,
        layer_budget_log_path: Optional[str] = None,
        eviction_protect_recent_frames: int = 0,
        recent_merge_config: Optional[RecentMergeConfig] = None,
        svd_eviction_merge_config: Optional[SvdEvictionMergeConfig] = None,
        voxel_covis_config: Optional[VoxelCovisConfig] = None,
        covis_log_fn: Optional[Callable[[str], None]] = None,
        global_attn_idx_ranges: Optional[Any] = None,
        global_attn_debug: bool = False,
    ):
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        total_budget = self.total_budget
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
        window_token_count = max(int(window_protect_frames), 0) * tokens_per_frame

        anchor_manager = None
        if history_anchor_strategy != "none":
            anchor_config = HistoryAnchorConfig(
                strategy=history_anchor_strategy,
                interval=anchor_interval,
                min_anchor_interval=min_anchor_interval,
                max_anchors=max_anchors,
                coverage_threshold=coverage_threshold,
                anchor_keep_ratio=anchor_keep_ratio,
            )
            anchor_manager = HistoryAnchorManager(anchor_config, tokens_per_frame)
            anchor_manager.image_size_hw = (img_h, img_w)

        def _select_anchor_token_indices(conf_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if conf_map is None:
                return None

            special_count = self.aggregator.patch_start_idx
            anchor_chunk = max(int(tokens_per_frame * anchor_keep_ratio), 1)
            if anchor_chunk <= special_count:
                return torch.arange(anchor_chunk, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)

            keep_patches = anchor_chunk - special_count
            if conf_map.dim() == 4:
                conf_map = conf_map.squeeze(1)
            if conf_map.dim() != 3:
                return None

            pooled = F.adaptive_avg_pool2d(conf_map.unsqueeze(1), (patch_h, patch_w)).squeeze(1)
            flat = pooled.reshape(conf_map.shape[0], -1)
            keep_patches = min(keep_patches, flat.shape[1])
            if keep_patches <= 0:
                return torch.arange(anchor_chunk, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)

            topk = torch.topk(flat, k=keep_patches, dim=1).indices
            special_indices = torch.arange(special_count, device=conf_map.device).unsqueeze(0).expand(conf_map.shape[0], -1)
            return torch.cat([special_indices, topk + special_count], dim=1)

        all_ress = []
        processed_frames = [] 

        for i, frame in enumerate(frames):

            fixed_interval_registered = False
            fixed_interval_is_fifo = False
            if anchor_manager is not None and history_anchor_strategy == "fixed_interval":
                should_register, is_fifo, reason = anchor_manager.should_become_anchor(frame_idx=i)
                if should_register:
                    anchor_manager.register_anchor(i)
                    fixed_interval_registered = True
                    fixed_interval_is_fifo = is_fifo
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

            anchor_token_count = (
                anchor_manager.get_protected_token_count()
                if anchor_manager is not None
                else None
            )
            effective_window_token_count = window_token_count if anchor_manager is not None else 0

            images = frame["img"].unsqueeze(0) 
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
                total_budget=total_budget,
                cache_analysis_config=cache_analysis_config,
                pre_eviction_snapshot_config=pre_eviction_snapshot_config,
                eviction_nn_analysis_config=eviction_nn_analysis_config,
                eviction_policy=eviction_policy,
                eviction_debug=eviction_debug,
                leverage_sketch_dim=leverage_sketch_dim,
                leverage_granularity=leverage_granularity,
                leverage_feature=leverage_feature,
                leverage_projection=leverage_projection,
                leverage_head_mean_dim=leverage_head_mean_dim,
                leverage_normalize_rows=leverage_normalize_rows,
                leverage_approx_method=leverage_approx_method,
                leverage_left_sketch_dim=leverage_left_sketch_dim,
                leverage_right_jl_dim=leverage_right_jl_dim,
                leverage_random_seed=leverage_random_seed,
                leverage_eviction_selector=leverage_eviction_selector,
                leverage_dpp_candidate_multiplier=leverage_dpp_candidate_multiplier,
                leverage_dpp_greedy_block_size=leverage_dpp_greedy_block_size,
                layer_budget_strategy=layer_budget_strategy,
                layer_budget_alpha=layer_budget_alpha,
                layer_budget_min_tokens=layer_budget_min_tokens,
                layer_budget_eps=layer_budget_eps,
                layer_budget_debug=layer_budget_debug,
                layer_budget_log_path=layer_budget_log_path,
                eviction_protect_recent_frames=eviction_protect_recent_frames,
                anchor_token_count=anchor_token_count,
                window_token_count=effective_window_token_count,
                recent_merge_config=recent_merge_config,
                svd_eviction_merge_config=svd_eviction_merge_config,
                voxel_covis_frame_ids=voxel_covis_frame_ids,
                voxel_covis_enabled=voxel_covis_graph is not None,
                global_attn_idx_ranges=global_attn_idx_ranges,
                global_attn_debug=global_attn_debug,
            )

            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output        
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
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

            if fixed_interval_registered:
                anchor_token_indices = _select_anchor_token_indices(
                    pts3d_conf if self.point_head is not None else None
                )
                past_key_values = self.aggregator.sync_anchor_change(
                    past_key_values,
                    anchor_token_count=anchor_token_count,
                    tokens_per_frame=tokens_per_frame,
                    anchor_keep_ratio=anchor_keep_ratio,
                    anchor_token_indices=anchor_token_indices,
                    is_fifo=fixed_interval_is_fifo,
                )

            if anchor_manager is not None and history_anchor_strategy == "coverage":
                should_register, is_fifo, reason, coverage = anchor_manager.should_become_anchor_coverage(
                    frame_idx=i,
                    current_depth=depth[0],
                    current_pose=camera_pose[0],
                )
                if should_register:
                    anchor_manager.register_anchor_coverage(i, depth[0], camera_pose[0])
                    fifo_msg = " (FIFO: oldest demoted)" if is_fifo else ""
                    print(f"[History Anchor] Frame {i} registered{fifo_msg}: {reason}")

                    anchor_token_indices = _select_anchor_token_indices(
                        pts3d_conf if self.point_head is not None else None
                    )
                    past_key_values = self.aggregator.sync_anchor_change(
                        past_key_values,
                        anchor_token_count=anchor_manager.get_protected_token_count(),
                        tokens_per_frame=tokens_per_frame,
                        anchor_keep_ratio=anchor_keep_ratio,
                        anchor_token_indices=anchor_token_indices,
                        is_fifo=is_fifo,
                    )

            if recent_merger is not None:
                tokens_per_frame = int(aggregated_tokens[-1].shape[2])
                geom = recent_merger.record_frame_geometry(
                    frame_id=i,
                    depth=depth,
                    depth_conf=depth_conf,
                    pose_enc=camera_pose,
                    image_hw=images.shape[-2:],
                    tokens_per_frame=tokens_per_frame,
                )
                if geom is not None:
                    for layer_id, layer_kv in enumerate(past_key_values):
                        if layer_kv is None or len(layer_kv) != 3:
                            continue
                        k_cache, v_cache, metadata = layer_kv
                        recent_merger.update_metadata_for_frame(metadata, i)
                        if run_recent_merge:
                            k_cache, v_cache, metadata, _ = recent_merger.merge_layer(
                                k_cache,
                                v_cache,
                                metadata,
                                layer_id=layer_id,
                                frame_id=i,
                            )
                        past_key_values[layer_id] = (k_cache, v_cache, metadata)

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
        if layer_kv is None or len(layer_kv) != 3:
            continue
        k_cache, _, metadata = layer_kv
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
