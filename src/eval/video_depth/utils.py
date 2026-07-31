from copy import deepcopy
import json
import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as iio
from matplotlib.figure import Figure


def save_focals(cam_dict, path):
    # convert focal to txt
    focals = cam_dict["focal"]
    np.savetxt(path, focals, fmt="%.6f")
    return focals


def save_intrinsics(cam_dict, path):
    K_raw = np.eye(3)[None].repeat(len(cam_dict["focal"]), axis=0)
    K_raw[:, 0, 0] = cam_dict["focal"]
    K_raw[:, 1, 1] = cam_dict["focal"]
    K_raw[:, :2, 2] = cam_dict["pp"]
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K, fmt="%.6f")
    return K_raw


def save_conf_maps(conf, path):
    for i, c in enumerate(conf):
        np.save(f"{path}/conf_{i}.npy", c.detach().cpu().numpy())
    return conf


def save_rgb_imgs(colors, path):
    imgs = colors
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        iio.imwrite(
            f"{path}/frame_{i:04d}.jpg", (img.cpu().numpy() * 255).astype(np.uint8)
        )
    return imgs


def save_depth_maps(pts3ds_self, path, conf_self=None):
    depth_maps = torch.stack([pts3d_self[..., -1] for pts3d_self in pts3ds_self], 0)
    min_depth = depth_maps.min()  # float(torch.quantile(out, 0.01))
    max_depth = depth_maps.max()  # float(torch.quantile(out, 0.99))
    colored_depth = colorize(
        depth_maps,
        cmap_name="Spectral_r",
        range=(min_depth, max_depth),
        append_cbar=True,
    )
    images = []

    if conf_self is not None:
        conf_selfs = torch.concat(conf_self, 0)
        min_conf = torch.log(conf_selfs.min())  # float(torch.quantile(out, 0.01))
        max_conf = torch.log(conf_selfs.max())  # float(torch.quantile(out, 0.99))
        colored_conf = colorize(
            torch.log(conf_selfs),
            cmap_name="jet",
            range=(min_conf, max_conf),
            append_cbar=True,
        )

    for i, depth_map in enumerate(colored_depth):
        # Apply color map to depth map
        img_path = f"{path}/frame_{(i):04d}.png"
        if conf_self is None:
            to_save = (depth_map * 255).detach().cpu().numpy().astype(np.uint8)
        else:
            to_save = torch.cat([depth_map, colored_conf[i]], dim=1)
            to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
        iio.imwrite(img_path, to_save)
        images.append(Image.open(img_path))
        np.save(f"{path}/frame_{(i):04d}.npy", depth_maps[i].detach().cpu().numpy())

    # comment this as it may fail sometimes
    # images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

    return depth_maps


def _depth_array_to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _reshape_depth_sequence(x, sequence_shape):
    x = _depth_array_to_numpy(x)
    if x.ndim == 3:
        return x
    if x.ndim == 2 and len(sequence_shape) == 3:
        num_frames, height, width = sequence_shape
        if x.shape == (num_frames * height, width):
            return x.reshape(num_frames, height, width)
    if x.ndim == 2:
        return x[None]
    raise ValueError(
        f"Expected depth map with shape [N,H,W] or [N*H,W], got {x.shape}"
    )


def _positive_percentile_range(error, valid_mask, percentile):
    finite_mask = valid_mask & np.isfinite(error)
    values = error[finite_mask]
    if values.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(values, percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return 0.0, vmax


def _error_stats(error, valid_mask):
    finite_mask = valid_mask & np.isfinite(error)
    values = error[finite_mask]
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _colorize_error_frame(
    error,
    valid_mask,
    value_range,
    *,
    cmap_name="magma",
    append_cbar=True,
):
    vmin, vmax = value_range
    denom = max(vmax - vmin, 1e-12)
    normalized = np.clip((np.nan_to_num(error, nan=vmin) - vmin) / denom, 0.0, 1.0)
    colored = cm.get_cmap(cmap_name)(normalized)[:, :, :3].astype(np.float32)
    colored[~valid_mask] = 0.0

    if append_cbar:
        cbar = get_vertical_colorbar(
            h=error.shape[0],
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            cbar_precision=3,
        )
        colored = np.concatenate(
            (colored, np.zeros_like(colored[:, :5, :]), cbar), axis=1
        )
    return (colored * 255).clip(0, 255).astype(np.uint8)



def _load_rgb_frame(image_path, target_shape):
    height, width = target_shape
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def _overlay_error_frame(
    image,
    error,
    valid_mask,
    value_range,
    *,
    alpha=0.55,
    cmap_name="magma",
    append_cbar=True,
):
    vmin, vmax = value_range
    denom = max(vmax - vmin, 1e-12)
    normalized = np.clip((np.nan_to_num(error, nan=vmin) - vmin) / denom, 0.0, 1.0)
    heat = cm.get_cmap(cmap_name)(normalized)[:, :, :3].astype(np.float32)
    overlay = image.copy()
    overlay[valid_mask] = (1.0 - alpha) * image[valid_mask] + alpha * heat[valid_mask]

    if append_cbar:
        cbar = get_vertical_colorbar(
            h=error.shape[0],
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            cbar_precision=3,
        )
        overlay = np.concatenate(
            (overlay, np.zeros_like(overlay[:, :5, :]), cbar), axis=1
        )
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def _write_mp4(frames, path, fps):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        frames = [
            np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            for frame in frames
        ]
        height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        try:
            iio.mimwrite(path, frames, fps=fps, macro_block_size=1)
            return
        except Exception as exc:
            raise RuntimeError(f"Failed to open mp4 writer for {path}") from exc
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_depth_error_visuals(
    depth_predict,
    depth_gt,
    relative_error,
    output_dir,
    *,
    sequence,
    sequence_shape,
    fps=10,
    percentile=99.0,
    image_paths=None,
    overlay_alpha=0.55,
):
    pred = _reshape_depth_sequence(depth_predict, sequence_shape).astype(np.float32)
    gt = _reshape_depth_sequence(depth_gt, sequence_shape).astype(np.float32)
    rel_error = _reshape_depth_sequence(relative_error, sequence_shape).astype(np.float32)

    valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    abs_error = np.abs(pred - gt)
    rel_error = np.where(np.isfinite(rel_error), rel_error, 0.0)

    abs_range = _positive_percentile_range(abs_error, valid_mask, percentile)
    rel_range = _positive_percentile_range(rel_error, valid_mask, percentile)

    abs_dir = os.path.join(output_dir, "absolute")
    rel_dir = os.path.join(output_dir, "relative")
    os.makedirs(abs_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)

    overlay_enabled = image_paths is not None
    if overlay_enabled:
        if len(image_paths) != gt.shape[0]:
            raise ValueError(
                f"Expected {gt.shape[0]} RGB frames for overlay, got {len(image_paths)}"
            )
        abs_overlay_dir = os.path.join(output_dir, "absolute_overlay")
        rel_overlay_dir = os.path.join(output_dir, "relative_overlay")
        os.makedirs(abs_overlay_dir, exist_ok=True)
        os.makedirs(rel_overlay_dir, exist_ok=True)

    abs_frames = []
    rel_frames = []
    abs_overlay_frames = []
    rel_overlay_frames = []
    for frame_idx in range(gt.shape[0]):
        abs_frame = _colorize_error_frame(
            abs_error[frame_idx],
            valid_mask[frame_idx],
            abs_range,
        )
        rel_frame = _colorize_error_frame(
            rel_error[frame_idx],
            valid_mask[frame_idx],
            rel_range,
        )
        iio.imwrite(os.path.join(abs_dir, f"frame_{frame_idx:04d}.png"), abs_frame)
        iio.imwrite(os.path.join(rel_dir, f"frame_{frame_idx:04d}.png"), rel_frame)
        abs_frames.append(abs_frame)
        rel_frames.append(rel_frame)

        if overlay_enabled:
            image = _load_rgb_frame(image_paths[frame_idx], gt.shape[1:])
            abs_overlay = _overlay_error_frame(
                image,
                abs_error[frame_idx],
                valid_mask[frame_idx],
                abs_range,
                alpha=overlay_alpha,
            )
            rel_overlay = _overlay_error_frame(
                image,
                rel_error[frame_idx],
                valid_mask[frame_idx],
                rel_range,
                alpha=overlay_alpha,
            )
            iio.imwrite(
                os.path.join(abs_overlay_dir, f"frame_{frame_idx:04d}.png"),
                abs_overlay,
            )
            iio.imwrite(
                os.path.join(rel_overlay_dir, f"frame_{frame_idx:04d}.png"),
                rel_overlay,
            )
            abs_overlay_frames.append(abs_overlay)
            rel_overlay_frames.append(rel_overlay)

    _write_mp4(abs_frames, os.path.join(output_dir, "absolute_error.mp4"), fps)
    _write_mp4(rel_frames, os.path.join(output_dir, "relative_error.mp4"), fps)
    if overlay_enabled:
        _write_mp4(
            abs_overlay_frames,
            os.path.join(output_dir, "absolute_overlay.mp4"),
            fps,
        )
        _write_mp4(
            rel_overlay_frames,
            os.path.join(output_dir, "relative_overlay.mp4"),
            fps,
        )

    summary = {
        "sequence": sequence,
        "frames": int(gt.shape[0]),
        "valid_pixels": int(np.sum(valid_mask)),
        "percentile": float(percentile),
        "overlay": bool(overlay_enabled),
        "overlay_alpha": float(overlay_alpha) if overlay_enabled else None,
        "absolute_range": [float(abs_range[0]), float(abs_range[1])],
        "relative_range": [float(rel_range[0]), float(rel_range[1])],
        "absolute_error": _error_stats(abs_error, valid_mask),
        "relative_error": _error_stats(rel_error, valid_mask),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    # fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = max(1, int(im.shape[1] / im.shape[0] * h))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = plt.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


# tensor
def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    """
    turn a grayscale image into a color image
    :param x: torch.Tensor, grayscale image, [H, W] or [B, H, W]
    :param mask: torch.Tensor or None, mask image, [H, W] or [B, H, W] or None
    """

    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out
