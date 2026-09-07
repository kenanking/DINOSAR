"""Content-aware cropping for SAR self-supervised pretraining."""

import math

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Image / content-map utilities
# ---------------------------------------------------------------------------


def _to_2d_amplitude(image):
    """Normalize an image tensor to [0, 1] float, squeeze to 2-D."""
    x = _to_2d_float(image)
    x = x - x.min()
    return x / (x.max() + 1e-6)


def _to_2d_float(image):
    """Convert an image tensor to 2-D float without changing its dynamic range."""
    x = image.float()
    if x.ndim == 3:
        x = x.squeeze(0)
    return x


def _ratio_gradient_content_map(x, scales, eps=1e-6):
    """Per-scale ``|grad(log mean)| * mean`` fused by L2 norm, then min-max normalized.

    Intensity weighting suppresses gradient response in dark speckle regions
    while preserving it over bright structural regions.
    """
    magnitudes = []
    for r in scales:
        pad = r // 2
        ih, iw = x.shape[-2:]
        mode = "reflect" if (pad < ih and pad < iw) else "replicate"
        mean_map = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode=mode), kernel_size=r, stride=1)
        off = r // 2
        mh, mw = mean_map.shape[-2:]
        smode = "reflect" if (off < mh and off < mw) else "replicate"
        mp = F.pad(mean_map, (off, off, off, off), mode=smode)
        left = mp[..., off : off + mh, 2 * off : 2 * off + mw]
        right = mp[..., off : off + mh, :mw]
        top = mp[..., 2 * off : 2 * off + mh, off : off + mw]
        bottom = mp[..., :mh, off : off + mw]
        gx = torch.log(left + eps) - torch.log(right + eps)
        gy = torch.log(top + eps) - torch.log(bottom + eps)
        magnitudes.append(torch.sqrt(gx.square() + gy.square()) * mean_map)

    stacked = torch.cat(magnitudes, dim=1)
    t = torch.linalg.norm(stacked, dim=1, keepdim=False)[0]
    t_min = t.min()
    return (t - t_min) / (t.max() - t_min + 1e-6)


def multi_scale_ratio_gradient_content_map(image, scales=(5, 9, 13, 17), eps=1e-6):
    """SAR-robust saliency map in ``[0, 1]``. Log-ratio absorbs multiplicative speckle."""
    return _ratio_gradient_content_map(_to_2d_amplitude(image)[None, None], scales, eps)


# ---------------------------------------------------------------------------
# Box / crop primitives
# ---------------------------------------------------------------------------


def _sample_area_and_aspect(image_h, image_w, scale_range, aspect_range, max_retry=20):
    """Sample ``(crop_h, crop_w)`` within scale/aspect constraints."""
    area_img = image_h * image_w
    for _ in range(max_retry):
        target_area = float(torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()) * area_img
        aspect = float(torch.empty(1).uniform_(math.log(aspect_range[0]), math.log(aspect_range[1])).exp().item())
        h = int(round(math.sqrt(target_area / aspect)))
        w = int(round(math.sqrt(target_area * aspect)))
        if 8 <= h <= image_h and 8 <= w <= image_w:
            return h, w
    side = int(round(math.sqrt(scale_range[0] * area_img)))
    side = max(8, min(side, image_h, image_w))
    return side, side


def _clip_box(top, left, height, width, image_h, image_w):
    top = max(0, min(top, image_h - height))
    left = max(0, min(left, image_w - width))
    return (top, left, top + height, left + width)


def _box_contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _union_box(boxes):
    top = min(b[0] for b in boxes)
    left = min(b[1] for b in boxes)
    bottom = max(b[2] for b in boxes)
    right = max(b[3] for b in boxes)
    return (top, left, bottom, right)


def _resize_crop(image, box, out_size):
    """Extract and bicubic-resize a crop to ``(out_size, out_size)``."""
    crop = image[box[0] : box[2], box[1] : box[3]].float()[None, None]
    crop = F.interpolate(crop, size=(out_size, out_size), mode="bicubic", align_corners=False)
    return crop[0, 0]


# ---------------------------------------------------------------------------
# Scene analysis
# ---------------------------------------------------------------------------

_ALPHA = 1.5
_GLOBAL_JITTER = 0.25
_SUPPORT_DIAG_RATIO = 0.7
_CONCENTRATION_BETA = 2.0
# Hard NMS on selected locals; guards against coverage^(1-c) collapsing when c -> 1.
_IOU_NMS_THRESHOLD = 0.5
# Anchor NMS: stricter than local NMS to keep prototypes diverse.
_ANCHOR_IOU_THRESHOLD = 0.15


def _square_max_pool2d_separable(x, kernel_size):
    """Exact square max-pooling via two 1-D passes."""
    pad = kernel_size // 2
    x = F.max_pool2d(x, kernel_size=(1, kernel_size), stride=1, padding=(0, pad))
    return F.max_pool2d(x, kernel_size=(kernel_size, 1), stride=1, padding=(pad, 0))


def _scene_concentration(content_map, beta=_CONCENTRATION_BETA, eps=1e-6):
    """``c = 1 - Var_p(position) / 2`` where ``p = content^beta`` on ``[-1, 1]^2``."""
    t = content_map.float() + eps
    p_beta = t.pow(beta) if beta != 1.0 else t
    p_beta = p_beta / p_beta.sum()

    h, w = p_beta.shape
    Y, X = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=p_beta.device, dtype=p_beta.dtype),
        torch.linspace(-1.0, 1.0, steps=w, device=p_beta.device, dtype=p_beta.dtype),
        indexing="ij",
    )
    mu_x = (p_beta * X).sum()
    mu_y = (p_beta * Y).sum()
    variance = (p_beta * ((X - mu_x) ** 2 + (Y - mu_y) ** 2)).sum()
    return float(torch.clamp(1.0 - variance / 2.0, min=0.0, max=1.0).item())


def _extract_anchor_boxes(content_map, ref_h, ref_w, max_anchors=4, peak_q=0.9, nms_kernel=11, salience_factor=3.0):
    """Detect content peaks and return anchor boxes centered on them.

    Peaks below ``salience_factor * mean(content_map)`` are dropped (except the top-1),
    so background speckle is rejected while genuine secondary targets survive.
    """
    cmap = content_map.float()
    thresh = float(torch.quantile(cmap.reshape(-1), q=peak_q).item())
    pooled = _square_max_pool2d_separable(cmap[None, None], nms_kernel)[0, 0]
    salience_floor = salience_factor * float(cmap.mean().item())
    peak_mask = (cmap >= pooled - 1e-12) & (cmap >= thresh)
    ys, xs = torch.where(peak_mask)
    peaks = sorted(
        [(float(cmap[y, x0].item()), int(y), int(x0)) for y, x0 in zip(ys.tolist(), xs.tolist())],
        reverse=True,
    )
    out = []
    box_area = float(ref_h * ref_w)
    for rank, (score, y, x0) in enumerate(peaks):
        if rank > 0 and score < salience_floor:
            break
        box = _clip_box(
            int(round(y - ref_h / 2)),
            int(round(x0 - ref_w / 2)),
            ref_h,
            ref_w,
            content_map.shape[0],
            content_map.shape[1],
        )
        good = True
        for b in out:
            iw = max(0, min(box[3], b[3]) - max(box[1], b[1]))
            ih = max(0, min(box[2], b[2]) - max(box[0], b[0]))
            inter = iw * ih
            iou = inter / (2.0 * box_area - inter + 1e-6)
            if iou > _ANCHOR_IOU_THRESHOLD:
                good = False
                break
        if good:
            out.append(box)
        if len(out) >= max_anchors:
            break
    if not out:
        h, w = content_map.shape
        out = [_clip_box(h // 2 - ref_h // 2, w // 2 - ref_w // 2, ref_h, ref_w, h, w)]
    return out


# ---------------------------------------------------------------------------
# Vectorized helpers for batch scoring
# ---------------------------------------------------------------------------


def _build_integral(t):
    """Padded integral image: integral[i+1, j+1] = sum of t[0..i, 0..j]."""
    h, w = t.shape
    out = torch.zeros(h + 1, w + 1, dtype=t.dtype, device=t.device)
    out[1:, 1:] = t.cumsum(0).cumsum(1)
    return out


def _integral_box_mean(integral, tops, lefts, heights, widths):
    """Vectorized O(1)-per-box mean using integral image."""
    bottoms = tops + heights
    rights = lefts + widths
    sums = integral[bottoms, rights] - integral[tops, rights] - integral[bottoms, lefts] + integral[tops, lefts]
    areas = (heights * widths).float().clamp(min=1)
    return sums / areas


def _batch_sample_sizes(n, image_h, image_w, scale_range, aspect_range):
    """Generate *n* random ``(height, width)`` crop sizes as int tensors."""
    area = image_h * image_w
    target_areas = torch.empty(n).uniform_(scale_range[0], scale_range[1]) * area
    aspects = torch.empty(n).uniform_(math.log(aspect_range[0]), math.log(aspect_range[1])).exp()
    hs = torch.sqrt(target_areas / aspects).round().long()
    ws = torch.sqrt(target_areas * aspects).round().long()
    hs = hs.clamp(8, image_h)
    ws = ws.clamp(8, image_w)
    return hs, ws


def _batch_iou_one(tops, lefts, heights, widths, ref):
    """IoU of *N* boxes (int tensors) against one reference Box."""
    bottoms = tops + heights
    rights = lefts + widths
    it = torch.clamp(tops, min=ref[0])
    il = torch.clamp(lefts, min=ref[1])
    ib = torch.clamp(bottoms, max=ref[2])
    ir = torch.clamp(rights, max=ref[3])
    ih = (ib - it).clamp(min=0)
    iw = (ir - il).clamp(min=0)
    inter = (ih * iw).float()
    areas = (heights * widths).float()
    ref_area = float((ref[2] - ref[0]) * (ref[3] - ref[1]))
    union = areas + ref_area - inter
    return inter / union.clamp(min=1.0)


def _sample_global_containing_region(image_h, image_w, region_box, area_range, aspect_range, jitter, max_retry=50):
    """Sample a global crop that geometrically contains *region_box*."""
    img_area = image_h * image_w
    rb_h, rb_w = region_box[2] - region_box[0], region_box[3] - region_box[1]
    region_cy = (region_box[0] + region_box[2]) / 2.0
    region_cx = (region_box[1] + region_box[3]) / 2.0
    for _ in range(max_retry):
        target_area = float(torch.empty(1).uniform_(area_range[0], area_range[1]).item()) * img_area
        aspect = float(torch.empty(1).uniform_(math.log(aspect_range[0]), math.log(aspect_range[1])).exp().item())
        gh = int(round(math.sqrt(target_area / aspect)))
        gw = int(round(math.sqrt(target_area * aspect)))
        gh = max(gh, rb_h)
        gw = max(gw, rb_w)
        gh = min(gh, image_h)
        gw = min(gw, image_w)
        min_top = max(0, region_box[2] - gh)
        max_top = min(region_box[0], image_h - gh)
        min_left = max(0, region_box[3] - gw)
        max_left = min(region_box[1], image_w - gw)
        if min_top > max_top or min_left > max_left:
            continue
        center_jitter_y = int(round(jitter * rb_h))
        center_jitter_x = int(round(jitter * rb_w))
        prefer_top = int(round(region_cy - gh / 2 + torch.randint(-center_jitter_y, center_jitter_y + 1, (1,)).item()))
        prefer_left = int(round(region_cx - gw / 2 + torch.randint(-center_jitter_x, center_jitter_x + 1, (1,)).item()))
        top = max(min_top, min(prefer_top, max_top))
        left = max(min_left, min(prefer_left, max_left))
        box = _clip_box(top, left, gh, gw, image_h, image_w)
        if _box_contains(box, region_box):
            return box
    return _clip_box(
        region_box[0],
        region_box[1],
        min(image_h, rb_h * 2),
        min(image_w, rb_w * 2),
        image_h,
        image_w,
    )


# ---------------------------------------------------------------------------
# SARContentAwareCropper
# ---------------------------------------------------------------------------


class SARContentAwareCropper:
    """Multiplicative-scoring local cropper for SAR pretraining.

    Score: ``q_i(b) = T̄(b) * A(b)^c * U_i(b)^{1-c}`` where ``T̄`` is content mean,
    ``A`` the anchor prior (Gaussian over distance to nearest anchor), ``U_i`` the
    coverage prior (``1 - max IoU`` with earlier picks), and ``c`` the scene
    concentration. Sparse scenes weight the anchor term; dense scenes weight coverage.

    Two globals always cover the anchors: if anchors span most of the image they
    are split into two clusters, otherwise both globals wrap the shared support.
    All locals are constrained to lie inside at least one global.
    """

    def __init__(
        self,
        n_local=8,
        global_out_size=224,
        local_out_size=96,
        local_area_scale=(0.05, 0.15),
        global_area_scale=(0.32, 1.0),
        local_aspect_range=(0.75, 1.3333),
        global_aspect_range=(0.75, 1.3333),
        coarse_scales=(3, 7),
        max_anchor_prototypes=4,
        num_candidates=256,
        normalize_amplitude=True,
        fixed_concentration=None,
        disable_content=False,
        disable_proximity=False,
        disable_coverage=False,
    ):
        if num_candidates < n_local:
            raise ValueError("num_candidates must be at least n_local")
        self.n_local = n_local
        self.global_out_size = global_out_size
        self.local_out_size = local_out_size
        self.local_area_scale = local_area_scale
        self.global_area_scale = global_area_scale
        self.local_aspect_range = local_aspect_range
        self.global_aspect_range = global_aspect_range
        self.coarse_scales = coarse_scales
        self.max_anchor_prototypes = max_anchor_prototypes
        self.num_candidates = num_candidates
        self.normalize_amplitude = normalize_amplitude
        self.fixed_concentration = fixed_concentration
        self.disable_content = disable_content
        self.disable_proximity = disable_proximity
        self.disable_coverage = disable_coverage

    def sample(self, image, *, return_metadata=False):
        """Generate 2 global + N local crops, optionally with metadata."""
        image = _to_2d_amplitude(image) if self.normalize_amplitude else _to_2d_float(image)
        h, w = image.shape
        dev = image.device

        # ---- Phase 1: Coarse content map ----
        # Scoring runs at coarse resolution; boxes are upscaled for final extraction.
        ds = 4
        c_h, c_w = max(32, h // ds), max(32, w // ds)
        sy, sx = h / c_h, w / c_w
        coarse_img = F.interpolate(
            image[None, None],
            size=(c_h, c_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        coarse_content = _ratio_gradient_content_map(coarse_img[None, None], scales=self.coarse_scales)

        # ---- Phase 2: Scene analysis ----
        concentration = (
            self.fixed_concentration
            if self.fixed_concentration is not None
            else _scene_concentration(coarse_content, beta=_CONCENTRATION_BETA)
        )
        ref_h, ref_w = _sample_area_and_aspect(c_h, c_w, self.local_area_scale, self.local_aspect_range)
        ref_diag = math.sqrt(float(ref_h**2 + ref_w**2))

        # ---- Phase 3: Integral image (coarse) ----
        integral = _build_integral(coarse_content)

        # ---- Phase 4: Anchors (coarse) ----
        coarse_anchors = _extract_anchor_boxes(
            coarse_content,
            ref_h,
            ref_w,
            max_anchors=self.max_anchor_prototypes,
        )
        anchors_t = torch.tensor(coarse_anchors, dtype=torch.long, device=dev)
        anchor_cy = (anchors_t[:, 0] + anchors_t[:, 2]).float() / 2.0
        anchor_cx = (anchors_t[:, 1] + anchors_t[:, 3]).float() / 2.0

        # ---- Phase 5: Globals (coarse) ----
        lo, hi = self.global_area_scale
        step = (hi - lo) / 2
        global_boxes = []

        if len(coarse_anchors) >= 2:
            union = _union_box(coarse_anchors)
            union_diag = math.sqrt((union[2] - union[0]) ** 2 + (union[3] - union[1]) ** 2)
            image_diag = math.sqrt(c_h**2 + c_w**2)
            anchors_are_close = union_diag < _SUPPORT_DIAG_RATIO * image_diag
        else:
            anchors_are_close = True

        if anchors_are_close:
            support = _union_box(coarse_anchors) if len(coarse_anchors) >= 2 else coarse_anchors[0]
            supports = [support, support]
            area_ranges = [(lo, lo + step), (lo + step, hi)]
        else:
            # Farthest-point seeding for 2-group clustering
            seed0 = coarse_anchors[0]
            farthest_idx = 1
            farthest_dist = -1.0
            s0_cy, s0_cx = (seed0[0] + seed0[2]) / 2.0, (seed0[1] + seed0[3]) / 2.0
            for i, a in enumerate(coarse_anchors[1:], 1):
                acy, acx = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
                d = math.sqrt((acy - s0_cy) ** 2 + (acx - s0_cx) ** 2)
                if d > farthest_dist:
                    farthest_dist, farthest_idx = d, i
            seed1 = coarse_anchors[farthest_idx]
            s1_cy, s1_cx = (seed1[0] + seed1[2]) / 2.0, (seed1[1] + seed1[3]) / 2.0
            group0, group1 = [seed0], [seed1]
            for a in coarse_anchors:
                if a is seed0 or a is seed1:
                    continue
                cy, cx = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
                d0 = math.sqrt((cy - s0_cy) ** 2 + (cx - s0_cx) ** 2)
                d1 = math.sqrt((cy - s1_cy) ** 2 + (cx - s1_cx) ** 2)
                (group0 if d0 <= d1 else group1).append(a)
            supports = [_union_box(group0), _union_box(group1)]
            area_ranges = [(lo, hi), (lo, hi)]

        for gi in range(2):
            global_boxes.append(
                _sample_global_containing_region(
                    c_h,
                    c_w,
                    supports[gi],
                    area_ranges[gi],
                    self.global_aspect_range,
                    _GLOBAL_JITTER,
                )
            )

        # ---- Phase 6: Batch candidate proposal (coarse) ----
        n_cand = self.num_candidates

        global_mask = torch.zeros(c_h, c_w, dtype=torch.bool, device=dev)
        for gb in global_boxes:
            global_mask[gb[0] : gb[2], gb[1] : gb[3]] = True

        flat_probs = (coarse_content * global_mask.float() + 1e-6).pow(_ALPHA).reshape(-1)
        flat_probs = flat_probs / flat_probs.sum()
        center_idx = torch.multinomial(flat_probs, n_cand, replacement=True)
        center_y = center_idx // c_w
        center_x = center_idx % c_w

        cand_hs, cand_ws = _batch_sample_sizes(n_cand, c_h, c_w, self.local_area_scale, self.local_aspect_range)
        cand_tops = (center_y - cand_hs // 2).clamp(min=0)
        cand_lefts = (center_x - cand_ws // 2).clamp(min=0)
        cand_tops = torch.min(cand_tops, (c_h - cand_hs).clamp(min=0))
        cand_lefts = torch.min(cand_lefts, (c_w - cand_ws).clamp(min=0))

        # ---- Phase 7: Batch scoring (coarse) ----
        cand_scores = _integral_box_mean(integral, cand_tops, cand_lefts, cand_hs, cand_ws)

        # Containment: each local must fit inside at least one global
        valid = torch.zeros(n_cand, dtype=torch.bool, device=dev)
        for gb in global_boxes:
            valid |= (
                (cand_tops >= gb[0])
                & (cand_lefts >= gb[1])
                & (cand_tops + cand_hs <= gb[2])
                & (cand_lefts + cand_ws <= gb[3])
            )
        cand_scores[~valid] = -1.0

        # Batch anchor prior: A(b) = max_m exp(-d^2 / d_ref^2)
        box_cy = cand_tops.float() + cand_hs.float() / 2.0
        box_cx = cand_lefts.float() + cand_ws.float() / 2.0
        dy = box_cy.unsqueeze(1) - anchor_cy.unsqueeze(0)
        dx = box_cx.unsqueeze(1) - anchor_cx.unsqueeze(0)
        dist_normed = torch.sqrt(dy**2 + dx**2) / max(ref_diag, 1.0)
        anchor_scores = torch.exp(-(dist_normed**2))
        anchor_priors, anchor_ids = anchor_scores.max(dim=1)

        # ---- Phase 8: Greedy selection with coverage prior ----
        # Warm-start: rank anchors by content strength
        anchor_strength = _integral_box_mean(
            integral,
            anchors_t[:, 0],
            anchors_t[:, 1],
            anchors_t[:, 2] - anchors_t[:, 0],
            anchors_t[:, 3] - anchors_t[:, 1],
        )
        warm_anchor_ids = torch.argsort(anchor_strength, descending=True)[: min(anchors_t.shape[0], self.n_local)]

        local_boxes = []
        local_scores = []
        local_details = []
        alive = cand_scores.clone()
        max_iou = torch.zeros(n_cand, device=dev)

        for lid in range(self.n_local):
            coverage = (1.0 - max_iou).clamp(min=1e-6) if lid > 0 else torch.ones(n_cand, device=dev)
            t_term = torch.ones(n_cand, device=dev) if self.disable_content else alive
            a_term = torch.ones(n_cand, device=dev) if self.disable_proximity else anchor_priors
            u_term = torch.ones(n_cand, device=dev) if self.disable_coverage else coverage
            combined = t_term * (a_term**concentration) * (u_term ** (1.0 - concentration))
            combined[alive < 0] = -float("inf")  # always mask invalid/already-selected candidates

            if lid < len(warm_anchor_ids):
                target = int(warm_anchor_ids[lid].item())
                masked = combined.clone()
                masked[anchor_ids != target] = -float("inf")
                best = int(masked.argmax().item()) if torch.isfinite(masked).any() else int(combined.argmax().item())
            else:
                best = int(combined.argmax().item())

            top = int(cand_tops[best].item())
            left = int(cand_lefts[best].item())
            box = (top, left, top + int(cand_hs[best].item()), left + int(cand_ws[best].item()))
            local_boxes.append(box)
            if return_metadata:
                cs = float(alive[best].item())
                local_scores.append(cs)
                local_details.append(
                    {
                        "score": float(combined[best].item()),
                        "content_score": cs,
                        "anchor_prior": float(anchor_priors[best].item()),
                        "coverage_prior": float(coverage[best].item()),
                    }
                )
            alive[best] = -1.0
            new_iou = _batch_iou_one(cand_tops, cand_lefts, cand_hs, cand_ws, box)
            max_iou = torch.max(max_iou, new_iou)
            alive[new_iou > _IOU_NMS_THRESHOLD] = -1.0

        # ---- Phase 9: Upscale boxes to full-res and extract crops ----
        def _upscale(b):
            top = int(round(b[0] * sy))
            left = int(round(b[1] * sx))
            height = max(1, int(round((b[2] - b[0]) * sy)))
            width = max(1, int(round((b[3] - b[1]) * sx)))
            return _clip_box(top, left, height, width, h, w)

        global_boxes = [_upscale(gb) for gb in global_boxes]
        local_boxes = [_upscale(lb) for lb in local_boxes]

        global_crops = [_resize_crop(image, b, self.global_out_size) for b in global_boxes]
        local_crops = [_resize_crop(image, b, self.local_out_size) for b in local_boxes]

        result = {
            "global_crops": global_crops,
            "local_crops": local_crops,
        }
        if not return_metadata:
            return result

        content_map = F.interpolate(
            coarse_content[None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        result.update(
            {
                "image": image,
                "content_map": content_map,
                "scene_concentration": concentration,
                "anchor_boxes": [_upscale(a) for a in coarse_anchors],
                "global_boxes": global_boxes,
                "local_boxes": local_boxes,
                "local_scores": local_scores,
                "local_details": local_details,
                "selection_stats": {
                    "num_valid_candidates": int((cand_scores >= 0).sum().item()),
                },
            }
        )
        return result


# ---------------------------------------------------------------------------
# ContentAwareMultiCropTransform -- training-compatible wrapper
# ---------------------------------------------------------------------------


class ContentAwareMultiCropTransform:
    """Training-compatible wrapper around :class:`SARContentAwareCropper`.

    Matches the ``MultiCropTransform`` ``__call__`` API, returning
    ``[(1, G, G), (1, G, G), (1, L, L), ...]``.
    """

    def __init__(
        self,
        global_crop_size=224,
        local_crop_size=96,
        global_crop_scale=(0.6, 1.0),
        local_crop_scale=(0.1, 0.5),
        num_local_crops=6,
        global_aspect_range=(0.75, 1.3333),
        local_aspect_range=(0.75, 1.3333),
        coarse_scales=(3, 7),
        max_anchor_prototypes=4,
        num_candidates=256,
        normalize_amplitude=True,
        fixed_concentration=None,
        disable_content=False,
        disable_proximity=False,
        disable_coverage=False,
        post_crop_transforms_global_1=None,
        post_crop_transforms_global_2=None,
        post_crop_transforms_local=None,
    ):
        self.cropper = SARContentAwareCropper(
            n_local=max(1, num_local_crops),
            global_out_size=global_crop_size,
            local_out_size=local_crop_size,
            local_area_scale=local_crop_scale,
            global_area_scale=global_crop_scale,
            local_aspect_range=local_aspect_range,
            global_aspect_range=global_aspect_range,
            coarse_scales=tuple(coarse_scales),
            max_anchor_prototypes=max_anchor_prototypes,
            num_candidates=num_candidates,
            normalize_amplitude=normalize_amplitude,
            fixed_concentration=fixed_concentration,
            disable_content=disable_content,
            disable_proximity=disable_proximity,
            disable_coverage=disable_coverage,
        )
        self.post_crop_transforms_global_1 = post_crop_transforms_global_1
        self.post_crop_transforms_global_2 = post_crop_transforms_global_2
        self.post_crop_transforms_local = post_crop_transforms_local

    def __call__(self, image, *, return_metadata=False):
        """Generate 2 global + N local crops; optionally return boxes/content map."""
        result = self.cropper.sample(image, return_metadata=return_metadata)

        global_crops = [c.unsqueeze(0) for c in result["global_crops"]]
        local_crops = [c.unsqueeze(0) for c in result["local_crops"]]

        if self.post_crop_transforms_global_1 is not None:
            global_crops[0] = self.post_crop_transforms_global_1(global_crops[0])
        if self.post_crop_transforms_global_2 is not None:
            global_crops[1] = self.post_crop_transforms_global_2(global_crops[1])
        if self.post_crop_transforms_local is not None:
            local_crops = [self.post_crop_transforms_local(c) for c in local_crops]

        views = global_crops + local_crops

        if return_metadata:
            return {
                "views": views,
                "image": result["image"],
                "content_map": result["content_map"],
                "scene_concentration": result["scene_concentration"],
                "anchor_boxes": result["anchor_boxes"],
                "global_boxes": result["global_boxes"],
                "local_boxes": result["local_boxes"],
                "local_scores": result["local_scores"],
                "local_details": result["local_details"],
                "selection_stats": result["selection_stats"],
            }
        return views
