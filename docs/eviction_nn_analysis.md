# Eviction Nearest-Retained Analysis

This opt-in diagnostic measures whether tokens removed by KV-cache eviction are close to tokens that remain in the cache.

Enable it with `--eviction_nn_analysis_dir <path>`. When the flag is unset, the model only pays a negligible `None` check and eviction decisions are unchanged.

## Reading The Metrics

Small nearest-retained cosine distance means an evicted token has a similar retained token and is likely redundant in key-feature space.

Large nearest-retained cosine distance means the evicted token may be sparse or non-replaceable. This is the failure mode to look for on KITTI if global SVD leverage removes low-density but useful tokens.

`cosine_distance_p50` is the median evicted-token distance for one eviction event. Low p50 means most evicted tokens are replaceable.

`cosine_distance_p90` highlights the tail. A high p90 means a meaningful minority of evicted tokens do not have close retained neighbors.

`frac_dist_gt_0.10` and `frac_dist_gt_0.20` are easy-to-compare tail rates. Higher values mean more evicted tokens are far from any retained token.

Frame-gap statistics use metadata when available. `frame_gap_mean` and `frame_gap_p90` measure how far away in time the nearest retained token is. `frac_same_frame` is high when evicted tokens are mostly represented by retained tokens from the same frame. `frac_nearest_recent_frame` measures how often the nearest retained token comes from the latest frame present in the cache.

## Dataset Comparison

If KITTI shows larger p50/p90 nearest distances or larger `frac_dist_gt_0.10` and `frac_dist_gt_0.20` than 7Scenes or NRGBD, that supports the hypothesis that KITTI has lower cache redundancy and SVD leverage may remove sparse useful tokens.

This is a feature-space diagnostic, not direct proof of downstream importance. Interpret it together with depth, pose, reconstruction metrics, and visualization.


## R_kNN Neighborhood Ratio

`R_kNN` compares the K-nearest-neighbor distance sum after eviction against the K-nearest-neighbor distance sum before eviction for the same analyzed evicted tokens:

`R_kNN = sum_i sum_{j in N_ret^K(i)} d(k_i, k_j) / sum_i sum_{j in N_pre^K(i)} d(k_i, k_j)`

The implementation uses cosine distance, `1 - cosine_similarity`, and excludes token `i` itself from the pre-eviction neighbor set. The default is `K=5`, configurable with `--eviction_nn_analysis_knn_k`.

`R_kNN` near `1` means the retained cache preserves roughly the same local neighborhood distance as the full pre-eviction cache.

`R_kNN > 1` means eviction made the local neighborhood farther away. Larger values indicate stronger neighborhood distortion.

`R_kNN >> 1`, especially together with high `cosine_distance_p90` or high `frac_dist_gt_0.20`, supports the concern that eviction removed sparse or non-redundant tokens.

`R_kNN < 1` can happen because the retained set may remove noisy or farther pre-neighbors, or because of ties and sampling. Treat it as a diagnostic clue rather than a direct quality guarantee.
