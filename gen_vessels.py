import argparse
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import frangi
from skimage.graph import route_through_array
from skimage.morphology import ball, binary_dilation, remove_small_objects, skeletonize_3d


def load_nifti(path):
    nii = nib.load(path)
    arr = np.asanyarray(nii.dataobj)
    return arr, nii.affine, nii.header


def save_nifti(arr, affine, header, out_path):
    out = nib.Nifti1Image(arr.astype(np.uint8), affine, header)
    nib.save(out, out_path)


def crop_bbox(mask, margin=80):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Input mask is empty.")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)
    return tuple(slice(mins[i], maxs[i]) for i in range(3))


def keep_connected_to_seed(candidate, seed):
    structure = ndi.generate_binary_structure(3, 2)
    labeled, _ = ndi.label(candidate.astype(bool), structure=structure)
    seed_labels = np.unique(labeled[seed.astype(bool)])
    seed_labels = seed_labels[seed_labels != 0]
    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)
    return np.isin(labeled, seed_labels)

def make_vessel_candidate(
    ct_crop,
    sma_valid,
    vein_valid,
    hu_min=95,
    hu_max=370,
    vesselness_thr=0.0012,
    use_frangi=True,
    territory_ratio=1.8,
    dilate_radius=1,
    air_hu=-500,
    air_radius=2,
):
    ct_f = ct_crop.astype(np.float32)

    candidate_core = (ct_f >= hu_min) & (ct_f <= hu_max)

    if use_frangi:
        print("Computing Frangi vesselness...")
        vness = frangi(
            ct_f,
            sigmas=[0.2, 0.3, 0.5, 0.8, 1.2],
            black_ridges=False,
        )
        candidate_core = candidate_core & (vness >= vesselness_thr)

    dist_to_sma = ndi.distance_transform_edt(~sma_valid)
    dist_to_vein = ndi.distance_transform_edt(~vein_valid)

    artery_territory = dist_to_sma < (dist_to_vein * territory_ratio)
    candidate_core = candidate_core & artery_territory

    # Remove only long bowel-boundary-like components near air, not all bowel-adjacent vessels
    candidate_core = remove_bowel_boundary_like_components(
        mask=candidate_core,
        ct_crop=ct_crop,
        protected=sma_valid,
        air_hu=air_hu,
        air_radius=air_radius,
        min_area=20,
        min_long_axis=12,
        min_aspect=2.5,
        axis=2,
    )

    candidate_core = candidate_core | sma_valid

    # Grow mask: slightly more permissive for connectivity
    candidate_grow = candidate_core.copy()

    if dilate_radius > 0:
        candidate_grow = binary_dilation(candidate_grow, ball(dilate_radius))
        candidate_grow = ndi.binary_closing(candidate_grow, structure=ball(1))
        candidate_grow = candidate_grow & artery_territory

        # Apply the bowel-boundary filter again after dilation,
        # because dilation can reconnect bowel-wall edges.
        candidate_grow = remove_bowel_boundary_like_components(
            mask=candidate_grow,
            ct_crop=ct_crop,
            protected=sma_valid,
            air_hu=air_hu,
            air_radius=air_radius,
            min_area=20,
            min_long_axis=12,
            min_aspect=2.5,
            axis=2,
        )

        candidate_grow = candidate_grow | sma_valid

    return (
        candidate_core.astype(bool),
        candidate_grow.astype(bool),
        artery_territory.astype(bool),
    )

def competitive_grow_gpu(candidate, sma_seed, vein_seed, max_iter=700, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cand = torch.from_numpy(candidate.astype(np.bool_)).to(device)
    sma = torch.from_numpy(sma_seed.astype(np.bool_)).to(device) & cand
    vein = torch.from_numpy(vein_seed.astype(np.bool_)).to(device) & cand
    cand = cand | sma | vein

    kernel = torch.ones((1, 1, 3, 3, 3), device=device)
    # 18-neighborhood: corner 제외, center 제외
    for i in [0, 2]:
        for j in [0, 2]:
            for k in [0, 2]:
                kernel[0, 0, i, j, k] = 0
    kernel[0, 0, 1, 1, 1] = 0

    sma_cur = sma[None, None]
    vein_cur = vein[None, None]
    sma_front = sma_cur.clone()
    vein_front = vein_cur.clone()
    cand = cand[None, None]

    for i in range(max_iter):
        occupied = sma_cur | vein_cur
        sma_grown = F.conv3d(sma_front.float(), kernel, padding=1) > 0
        vein_grown = F.conv3d(vein_front.float(), kernel, padding=1) > 0

        sma_new = sma_grown & cand & (~occupied)
        vein_new = vein_grown & cand & (~occupied)

        conflict = sma_new & vein_new
        sma_new = sma_new & (~conflict)
        vein_new = vein_new & (~conflict)

        n_new = int(sma_new.sum().item() + vein_new.sum().item())
        if n_new == 0:
            print(f"\nConverged at iteration {i}")
            break

        sma_cur |= sma_new
        vein_cur |= vein_new
        sma_front = sma_new
        vein_front = vein_new

        if i % 20 == 0:
            print(f"\rIter {i} | SMA: {int(sma_cur.sum())} | SMV: {int(vein_cur.sum())}", end="", flush=True)

    print(flush=True)
    return (
        sma_cur.squeeze().detach().cpu().numpy().astype(bool),
        vein_cur.squeeze().detach().cpu().numpy().astype(bool),
    )


def filter_by_sma_cross_section_area(vessel, sma, axis=2, scale=1.8):
    sma_areas = []
    for z in range(sma.shape[axis]):
        sma_slice = np.take(sma, z, axis=axis)
        sma_areas.append(sma_slice.sum())

    max_allowed_area = max(sma_areas) * scale
    filtered = np.zeros_like(vessel, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(vessel.shape[axis]):
        vessel_slice = np.take(vessel, z, axis=axis)
        labeled, n = ndi.label(vessel_slice, structure=structure2d)
        kept_slice = np.zeros_like(vessel_slice, dtype=bool)
        for lab in range(1, n + 1):
            comp = labeled == lab
            if comp.sum() <= max_allowed_area:
                kept_slice |= comp
        if axis == 0:
            filtered[z, :, :] = kept_slice
        elif axis == 1:
            filtered[:, z, :] = kept_slice
        else:
            filtered[:, :, z] = kept_slice
    return filtered


def connect_nearby_branches_by_path(
    artery_tree,
    candidate,
    ct_crop,
    max_dist=10,
    hu_min=90,
    hu_max=400,
    dilate_radius=0,
    max_components=10,
    max_comp_size=500,
):
    passable = candidate | ((ct_crop >= hu_min) & (ct_crop <= hu_max))
    passable = passable & (ct_crop >= 0)

    labeled, _ = ndi.label(candidate)
    main = artery_tree.astype(bool)
    main_coords = np.argwhere(main)
    if len(main_coords) == 0:
        return artery_tree

    tree = cKDTree(main_coords)
    out = main.copy()
    objs = ndi.find_objects(labeled)
    processed = 0

    for lab, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        comp = labeled[slc] == lab
        comp_size = int(comp.sum())
        if comp_size < 3 or comp_size > max_comp_size:
            continue
        comp_global = np.argwhere(comp) + np.array([s.start for s in slc])
        dists, idxs = tree.query(comp_global, k=1)
        min_i = np.argmin(dists)
        if dists[min_i] > max_dist:
            continue

        start = tuple(main_coords[idxs[min_i]])
        end = tuple(comp_global[min_i])
        lo = np.maximum(np.minimum(start, end) - max_dist - 5, 0)
        hi = np.minimum(np.maximum(start, end) + max_dist + 6, candidate.shape)
        local_slices = tuple(slice(lo[d], hi[d]) for d in range(3))
        local_passable = passable[local_slices]
        local_start = tuple(np.array(start) - lo)
        local_end = tuple(np.array(end) - lo)
        cost = np.where(local_passable, 1.0, 1e5)

        try:
            path, cost_val = route_through_array(cost, local_start, local_end, fully_connected=True)
        except Exception:
            continue
        if cost_val > max_dist * 4:
            continue

        path = np.array(path) + lo
        out[tuple(path.T)] = True
        out[tuple(comp_global.T)] = True
        processed += 1
        if processed >= max_components:
            break

    if dilate_radius > 0:
        out = binary_dilation(out, ball(dilate_radius))
        out = out & passable

    print("Connected components:", processed)
    return out
def add_tube_path(out, path, radius=2):
    tube = np.zeros_like(out, dtype=bool)
    path = np.asarray(path)
    tube[tuple(path.T)] = True
    tube = binary_dilation(tube, ball(radius))
    return out | tube


def connect_valid_vessel_gaps(
    artery_tree,
    candidate,
    ct_crop,
    max_dist=8,
    hu_min=-30,
    hu_max=400,
    min_align=0.5,
):
    skel = skeletonize_3d(artery_tree).astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoints = skel & (neighbor_count == 1)
    ep_coords = np.argwhere(endpoints)

    if len(ep_coords) < 2:
        print("Valid vessel gap connections: 0")
        return artery_tree

    passable = candidate | ((ct_crop >= hu_min) & (ct_crop <= hu_max))
    passable = passable & (ct_crop >= hu_min)

    def endpoint_dir(p, r=5):
        p = np.asarray(p)
        lo = np.maximum(p - r, 0)
        hi = np.minimum(p + r + 1, skel.shape)
        slc = tuple(slice(lo[d], hi[d]) for d in range(3))

        pts = np.argwhere(skel[slc]) + lo
        if len(pts) < 3:
            return None

        pts0 = pts - p
        cov = pts0.T @ pts0
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]

        mean_vec = pts0.mean(axis=0)
    
        if np.dot(v, mean_vec) < 0:
            v = -v

        # v는 endpoint에서 혈관 내부로 향하는 방향.
        # gap 연결에는 바깥으로 나가는 방향이 필요하므로 뒤집는다.
        v = -v

        return v / (np.linalg.norm(v) + 1e-6)
    

    dirs = [endpoint_dir(p) for p in ep_coords]

    tree = cKDTree(ep_coords)
    pairs = list(tree.query_pairs(r=max_dist))
    pairs.sort(key=lambda ij: np.linalg.norm(ep_coords[ij[0]] - ep_coords[ij[1]]))

    out = artery_tree.copy()
    connected = 0

    for i, j in pairs:
        p1 = ep_coords[i]
        p2 = ep_coords[j]

        v1 = dirs[i]
        v2 = dirs[j]

        if v1 is None or v2 is None:
            continue

        gap = p2 - p1
        dist = np.linalg.norm(gap)

        if dist < 2 or dist > max_dist:
            continue

        gap_dir = gap / dist

        if np.dot(v1, gap_dir) < min_align:
            continue
        if np.dot(v2, -gap_dir) < min_align:
            continue

        lo = np.maximum(np.minimum(p1, p2) - 5, 0)
        hi = np.minimum(np.maximum(p1, p2) + 6, artery_tree.shape)

        slc = tuple(slice(lo[d], hi[d]) for d in range(3))
        local_passable = passable[slc]

        start = tuple(p1 - lo)
        end = tuple(p2 - lo)

        cost = np.where(local_passable, 1.0, 1e6)

        try:
            path, cost_val = route_through_array(
                cost,
                start,
                end,
                fully_connected=True,
            )
        except Exception:
            continue

        if cost_val > max_dist * 2.5:
            continue

        path = np.array(path) + lo
        out = add_tube_path(out, path, radius=2)
        connected += 1

    print("Valid vessel gap connections:", connected)
    return out | artery_tree
def remove_bowel_boundary_like_components(
    mask,
    ct_crop,
    protected=None,
    air_hu=-500,
    air_radius=2,
    min_area=20,
    min_long_axis=12,
    min_aspect=2.5,
    axis=2,
):
    """
    Remove long/linear components near air-filled bowel lumen.
    This does NOT remove all vessels near bowel.
    It removes only air-adjacent components that look like long bowel-wall boundaries in 2D slices.
    """
    air = ct_crop <= air_hu
    near_air = binary_dilation(air, ball(air_radius))

    suspect = mask & near_air

    if protected is None:
        protected = np.zeros_like(mask, dtype=bool)

    remove = np.zeros_like(mask, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for z in range(mask.shape[axis]):
        s = np.take(suspect, z, axis=axis)
        p = np.take(protected, z, axis=axis)

        labeled, n = ndi.label(s, structure=structure2d)
        remove_slice = np.zeros_like(s, dtype=bool)

        for lab in range(1, n + 1):
            comp = labeled == lab
            area = int(comp.sum())
            if area < min_area:
                continue

            coords = np.argwhere(comp)
            if len(coords) == 0:
                continue

            span = coords.max(axis=0) - coords.min(axis=0) + 1
            long_axis = int(span.max())
            short_axis = int(max(span.min(), 1))
            aspect = long_axis / short_axis

            # bowel boundary: long, thin/arc-like component near air
            if long_axis >= min_long_axis and aspect >= min_aspect:
                remove_slice |= comp

        # never remove protected seed voxels
        remove_slice = remove_slice & (~p)

        if axis == 0:
            remove[z, :, :] = remove_slice
        elif axis == 1:
            remove[:, z, :] = remove_slice
        else:
            remove[:, :, z] = remove_slice

    print("Bowel-boundary-like voxels removed:", int(remove.sum()))

    out = mask & (~remove)
    out = out | protected
    return out
def bridge_tiny_gaps_with_grow_candidate(
    artery_tree,
    candidate_grow,
    candidate_core,
    artery_territory,
    vein_exclude,
    sma_valid,
    ct_crop,
    air_hu=-500,
    air_radius=2,
    n_iter=1,
):
    """
    Recover only tiny gaps using candidate_grow.
    This does NOT use broad endpoint filling.
    It only adds voxels created by local closing within the already bowel-filtered grow candidate.
    """

    out = artery_tree.astype(bool).copy()

    safe_mask = candidate_grow & artery_territory
    safe_mask = safe_mask & (~vein_exclude)
    safe_mask = safe_mask | sma_valid

    # 다시 한 번 bowel boundary-like component 제거
    safe_mask = remove_bowel_boundary_like_components(
        mask=safe_mask,
        ct_crop=ct_crop,
        protected=sma_valid,
        air_hu=air_hu,
        air_radius=air_radius,
        min_area=20,
        min_long_axis=12,
        min_aspect=2.5,
        axis=2,
    )

    total_added = 0

    for _ in range(n_iter):
        # use only ball(1)
        closed = ndi.binary_closing(out, structure=ball(1))

        bridge = closed & safe_mask & (~out)

        # prevent spreading too wide
        bridge = bridge & binary_dilation(out, ball(1))

        # only allow voxels within grow_candidate
        added = int(bridge.sum())
        out = out | bridge
        total_added += added

        if added == 0:
            break

    out = out | sma_valid
    print("Tiny bridge voxels added:", total_added)

    return out
def keep_connected_to_seed_6(candidate, seed):
    structure6 = ndi.generate_binary_structure(3, 1)
    labeled, _ = ndi.label(candidate.astype(bool), structure=structure6)

    seed_labels = np.unique(labeled[seed.astype(bool)])
    seed_labels = seed_labels[seed_labels != 0]

    if len(seed_labels) == 0:
        return np.zeros_like(candidate, dtype=bool)

    return np.isin(labeled, seed_labels)


def connect_detached_large_components(
    artery_tree,
    candidate_grow,
    artery_territory,
    vein_exclude,
    sma_valid,
    ct_crop,
    air_hu=-500,
    air_radius=2,
    min_comp_size=100,
    max_dist=25,
    max_components=5,
    apply_bowel_filter=True,
    direct_bridge_hu_min=30,
    direct_bridge_hu_max=370,
    hard_forbid=None,
):
    """
    Connect only sizeable detached vessel components to the SMA tree.
    This avoids broad endpoint/gap filling and does not dilate the whole tree.
    """

    structure6 = ndi.generate_binary_structure(3, 1)

    safe_mask = candidate_grow & artery_territory
    safe_mask = safe_mask & (~vein_exclude)
    safe_mask = safe_mask | sma_valid

    if apply_bowel_filter:
        safe_mask = remove_bowel_boundary_like_components(
            mask=safe_mask,
            ct_crop=ct_crop,
            protected=sma_valid,
            air_hu=air_hu,
            air_radius=air_radius,
            min_area=20,
            min_long_axis=12,
            min_aspect=2.5,
            axis=2,
        )
    out = artery_tree.astype(bool).copy()

    if hard_forbid is None:
        hard_forbid = np.zeros_like(out, dtype=bool)
    else:
        hard_forbid = hard_forbid.astype(bool)

    labeled, n = ndi.label(out, structure=structure6)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    seed_labels = np.unique(labeled[sma_valid.astype(bool)])
    seed_labels = seed_labels[seed_labels != 0]

    if len(seed_labels) == 0:
        print("Connected detached components: 0")
        return out

    main_label = seed_labels[np.argmax(sizes[seed_labels])]
    main = labeled == main_label
    main_coords = np.argwhere(main)

    if len(main_coords) == 0:
        print("Connected detached components: 0")
        return out

    connected = 0

    # connect a larger component first
    comp_labels = np.argsort(sizes)[::-1]

    for lab in comp_labels:
        if lab == 0 or lab == main_label:
            continue

        comp_size = int(sizes[lab])
        if comp_size < min_comp_size:
            continue

        comp = labeled == lab
        comp_coords = np.argwhere(comp)

        tree = cKDTree(main_coords)
        dists, idxs = tree.query(comp_coords, k=1)

        min_i = int(np.argmin(dists))
        min_dist = float(dists[min_i])

        print(f"Detached comp label={lab}, size={comp_size}, min_dist={min_dist:.2f}")

        if min_dist > max_dist:
            print("  skipped: too far")
            continue

        start = tuple(main_coords[idxs[min_i]])
        end = tuple(comp_coords[min_i])

        lo = np.maximum(np.minimum(start, end) - max_dist - 3, 0)
        hi = np.minimum(np.maximum(start, end) + max_dist + 4, out.shape)

        slc = tuple(slice(lo[d], hi[d]) for d in range(3))

        local_safe = safe_mask[slc]
        local_start = tuple(np.array(start) - lo)
        local_end = tuple(np.array(end) - lo)

        cost = np.where(local_safe, 1.0, 1e6)

        try:
            path, cost_val = route_through_array(
                cost,
                local_start,
                local_end,
                fully_connected=False,
            )
        except Exception as e:
            print("  skipped: no path", e)
            continue

        print(f"  path cost={cost_val:.2f}")

        if comp_size >= 800 and min_dist <= 5:
            allowed_cost = max(max_dist * 6.0, 160)
        elif comp_size >= 300 and min_dist <= 5:
            allowed_cost = max(max_dist * 4.0, 120)
        else:
            allowed_cost = max_dist * 2.5

        if cost_val > allowed_cost and comp_size >= 300 and min_dist <= 4.0:
            p0 = np.array(start)
            p1 = np.array(end)

            n_steps = int(np.ceil(np.linalg.norm(p1 - p0))) + 1
            line = np.linspace(p0, p1, n_steps)
            line = np.round(line).astype(int)
            line = np.unique(line, axis=0)

            valid = np.all((line >= 0) & (line < np.array(out.shape)), axis=1)
            line = line[valid]

            if len(line) > 0:
                line_mask = np.zeros_like(out, dtype=bool)
                line_mask[tuple(line.T)] = True
                line_tube = binary_dilation(line_mask, ball(1))

                direct_safe = (
                    (ct_crop >= direct_bridge_hu_min) &
                    (ct_crop <= direct_bridge_hu_max) &
                    artery_territory &
                    (~vein_exclude) &
                    (~hard_forbid)
                )

                direct_tube = line_tube & direct_safe

                # require that most of the direct line is passable
                line_passable = direct_safe[tuple(line.T)]
                pass_ratio = float(line_passable.mean()) if len(line_passable) > 0 else 0.0

                required_ratio = 0.3 if min_dist <= 3.5 else 0.7

                if pass_ratio >= required_ratio:
                    test_out = out | direct_tube
                    test_out[comp] = True
                    test_out = test_out | sma_valid

                    test_main = keep_connected_to_seed_6(test_out, sma_valid)

                    if np.any(comp & test_main):
                        print(
                            f"  direct short bridge accepted and connected, "
                            f"pass_ratio={pass_ratio:.2f}, required={required_ratio:.2f}"
                        )

                        out = test_out
                        main = test_main
                        main_coords = np.argwhere(main)

                        connected += 1

                        if connected >= max_components:
                            break

                        continue

                    else:
                        print(
                            f"  direct short bridge accepted by ratio but NOT connected, "
                            f"pass_ratio={pass_ratio:.2f}, required={required_ratio:.2f}"
                        )

                        # For extremely close gaps, try a slightly thicker direct bridge.
                        # This is only allowed for very short gaps to avoid weird long connections.
                        if min_dist <= 3.5:
                            line_tube2 = binary_dilation(line_mask, ball(2))

                            direct_tube2 = line_tube2 & direct_safe

                            test_out2 = out | direct_tube2
                            test_out2[comp] = True
                            test_out2 = test_out2 | sma_valid

                            test_main2 = keep_connected_to_seed_6(test_out2, sma_valid)
                            if np.any(comp & test_main2):
                                print("  thicker direct bridge accepted and connected")

                                out = test_out2
                                main = test_main2
                                main_coords = np.argwhere(main)

                                connected += 1

                                if connected >= max_components:
                                    break

                                continue

                            else:
                                print("  thicker direct bridge also NOT connected")

                                # Last resort: extremely short forced local bridge.
                                # Only for very close gaps. This avoids long, weird connections.
                                if min_dist <= 3.5:
                                    force_tube = binary_dilation(line_mask, ball(2))

                                    # Hard safety only:
                                    # - do not pass through vein exclusion
                                    # - do not pass through bowel interior / hard forbidden region
                                    # - avoid obvious air/negative regions
                                    force_safe = (
                                        (ct_crop >= -50) &
                                        (ct_crop <= direct_bridge_hu_max) &
                                        (~vein_exclude) &
                                        (~hard_forbid)
                                    )

                                    force_tube = force_tube & force_safe

                                    # Require at least some support from original safe/candidate/tree/component
                                    support = force_tube & (direct_safe | candidate_grow | out | comp)
                                    support_ratio = float(support.sum()) / max(float(force_tube.sum()), 1.0)

                                    test_out3 = out | force_tube
                                    test_out3[comp] = True
                                    test_out3 = test_out3 | sma_valid

                                    test_main3 = keep_connected_to_seed_6(test_out3, sma_valid)

                                    if support_ratio >= 0.15 and np.any(comp & test_main3):
                                        print(
                                            f"  forced very-short bridge accepted and connected, "
                                            f"support_ratio={support_ratio:.2f}"
                                        )

                                        out = test_out3
                                        main = test_main3
                                        main_coords = np.argwhere(main)

                                        connected += 1

                                        if connected >= max_components:
                                            break

                                        continue

                                    else:
                                        print(
                                            f"  forced very-short bridge rejected, "
                                            f"support_ratio={support_ratio:.2f}"
                                        )

                else:
                    print(
                        f"  direct short bridge rejected, "
                        f"pass_ratio={pass_ratio:.2f}, required={required_ratio:.2f}"
                    )
                    # Last resort even when direct pass_ratio is low.
                    # For extremely close detached components, allow a local bridge
                    # if it does not cross vein or bowel interior.
                    if min_dist <= 2.0 and comp_size >= 100:
                        # Use a slightly thicker local bridge for extremely short gaps.
                        # This is not global dilation; it is restricted to the local line zone.
                        bridge_radius = 3 if min_dist <= 2.0 else 2

                        line_zone = binary_dilation(line_mask, ball(bridge_radius + 1))

                        force_safe = (
                            (ct_crop >= -80) &
                            (ct_crop <= direct_bridge_hu_max) &
                            artery_territory &
                            (~vein_exclude) &
                            (~hard_forbid)
                        )

                        # Initial bridge tube
                        force_tube = binary_dilation(line_mask, ball(bridge_radius))
                        force_tube = force_tube & force_safe

                        # Local closing: fill the small lumen-like gap between main and detached branch
                        local_obj = (out | comp | force_tube) & line_zone
                        local_closed = ndi.binary_closing(local_obj, structure=ball(2))

                        # Keep only safe voxels in the very local bridge zone
                        local_closed = local_closed & line_zone & force_safe

                        test_out4 = out | comp | force_tube | local_closed
                        test_out4 = test_out4 | sma_valid

                        test_main4 = keep_connected_to_seed_6(test_out4, sma_valid)

                        if np.any(comp & test_main4):
                            print("  forced ultra-short bridge accepted despite low pass_ratio")

                            out = test_out4
                            main = test_main4
                            main_coords = np.argwhere(main)

                            connected += 1

                            if connected >= max_components:
                                break

                            continue
                        else:
                            print("  forced ultra-short bridge failed")

        if cost_val > allowed_cost:
            print(f"  skipped: path cost too high > {allowed_cost:.2f}")
            continue

        path = np.array(path) + lo
        tube = np.zeros_like(out, dtype=bool)
        tube[tuple(path.T)] = True
        tube = binary_dilation(tube, ball(1))
        tube = tube & safe_mask

        out = out | tube
        out[comp] = True
        out = out | sma_valid

        main = keep_connected_to_seed_6(out, sma_valid)
        main_coords = np.argwhere(main)

        connected += 1

        if connected >= max_components:
            break

    print("Connected detached components:", connected)

    # out = keep_connected_to_seed_6(out, sma_valid)
    out = out | sma_valid

    return out
def prune_terminal_boundary_branches(
    artery_tree,
    sma_valid,
    min_branch_size=80,
    max_branch_size=5000,
    endpoint_dilate_radius=2,
    neck_radius=1,
):
    """
    Remove terminal false-positive branches that are attached to the main tree by a thin neck.
    This is useful for bowel-wall/boundary-like structures that remain connected to the vessel tree.
    """

    out = artery_tree.astype(bool).copy()

    # skeletonize current tree
    skel = skeletonize_3d(out).astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoints = skel & (neighbor_count == 1)
    endpoints = endpoints & (~binary_dilation(sma_valid, ball(3)))

    ep_zone = binary_dilation(endpoints, ball(endpoint_dilate_radius))

    # Candidate terminal parts near endpoints
    terminal_zone = out & ep_zone

    # Remove a small neck around the main tree, then identify terminal blobs
    neck = binary_dilation(skel & (neighbor_count >= 2), ball(neck_radius))
    terminal_candidates = out & (~neck)

    structure26 = ndi.generate_binary_structure(3, 3)
    labeled, n = ndi.label(terminal_candidates, structure=structure26)

    remove = np.zeros_like(out, dtype=bool)

    for lab in range(1, n + 1):
        comp = labeled == lab
        size = int(comp.sum())

        if size < min_branch_size or size > max_branch_size:
            continue

        # Only remove terminal components that touch endpoints
        if not np.any(comp & ep_zone):
            continue

        # Never remove SMA seed
        if np.any(comp & sma_valid):
            continue

        coords = np.argwhere(comp)
        if len(coords) < 3:
            continue

        # shape check: boundary-like pieces tend to be wide/flat or irregular,
        # not a compact round vessel segment
        span = coords.max(axis=0) - coords.min(axis=0) + 1
        long_axis = span.max()
        short_axis = max(span.min(), 1)
        aspect = long_axis / short_axis

        if aspect >= 3.0:
            remove |= comp

    print("Terminal boundary-like branch voxels removed:", int(remove.sum()))

    out = out & (~remove)
    out = out | sma_valid
    return out

def remove_bowel_interior_only(
    artery_tree,
    bowel_crop,
    sma_valid,
    interior_radius=2,
):
    """
    Remove only voxels inside the small-bowel interior.
    Keep vessels touching or adjacent to the bowel wall.
    """

    if bowel_crop is None:
        return artery_tree

    bowel_mask = bowel_crop.astype(bool)

    if interior_radius > 0:
        bowel_interior = ndi.binary_erosion(
            bowel_mask,
            structure=ball(interior_radius)
        )
    else:
        bowel_interior = bowel_mask

    # protect SMA seed
    remove = artery_tree & bowel_interior & (~sma_valid)

    out = artery_tree & (~remove)
    out = out | sma_valid

    print("Bowel interior voxels removed:", int(remove.sum()))

    return out
def prune_bowel_wall_edges_keep_vessel_contacts(
    artery_tree,
    bowel_crop,
    sma_valid,
    interior_radius=2,
    wall_radius=1,
    protect_radius=2,
    min_area=20,
    min_long_axis=10,
    min_aspect=2.0,
    axes=(2,),
):
    """
    Remove long bowel-wall edge-like false positives,
    while keeping vessel endpoints that touch the bowel wall.

    Logic:
    - bowel_interior is removed elsewhere.
    - bowel_wall_band is the shell around the bowel wall.
    - long, thin components inside the wall band are likely bowel wall edges.
    - contact zones connected to vessels outside the wall band are protected.
    """

    if bowel_crop is None:
        return artery_tree

    out = artery_tree.astype(bool).copy()
    bowel_mask = bowel_crop.astype(bool)

    # Bowel interior
    if interior_radius > 0:
        bowel_interior = ndi.binary_erosion(
            bowel_mask,
            structure=ball(interior_radius),
        )
    else:
        bowel_interior = bowel_mask.copy()

    # Bowel wall shell: bowel mask minus eroded interior
    bowel_wall = bowel_mask & (~bowel_interior)

    # Slightly expand wall shell to account for imperfect bowel segmentation
    if wall_radius > 0:
        bowel_wall_band = binary_dilation(bowel_wall, ball(wall_radius))
    else:
        bowel_wall_band = bowel_wall.copy()

    # Do not include deep bowel interior in wall band
    bowel_wall_band = bowel_wall_band & (~bowel_interior)

    # Protect vessel segments that approach the bowel wall from outside
    outside_wall_vessel = out & (~bowel_wall_band) & (~bowel_interior)
    contact_protect = binary_dilation(outside_wall_vessel, ball(protect_radius)) & bowel_wall_band
    contact_protect = contact_protect | sma_valid

    total_remove = np.zeros_like(out, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for axis in axes:
        suspect = out & bowel_wall_band & (~sma_valid)

        for z in range(out.shape[axis]):
            s = np.take(suspect, z, axis=axis)
            p = np.take(contact_protect, z, axis=axis)

            labeled, n = ndi.label(s, structure=structure2d)
            remove_slice = np.zeros_like(s, dtype=bool)

            for lab in range(1, n + 1):
                comp = labeled == lab
                area = int(comp.sum())

                if area < min_area:
                    continue

                coords = np.argwhere(comp)
                if len(coords) < 3:
                    continue

                span = coords.max(axis=0) - coords.min(axis=0) + 1
                long_axis = int(span.max())
                short_axis = int(max(span.min(), 1))
                aspect = long_axis / short_axis

                # Long, thin component along bowel wall = likely bowel edge
                if long_axis >= min_long_axis and aspect >= min_aspect:
                    # Keep the local contact point where a real vessel reaches the bowel wall
                    remove_slice |= (comp & (~p))

            if axis == 0:
                total_remove[z, :, :] |= remove_slice
            elif axis == 1:
                total_remove[:, z, :] |= remove_slice
            else:
                total_remove[:, :, z] |= remove_slice

    # Never remove SMA seed
    total_remove = total_remove & (~sma_valid)

    out = out & (~total_remove)
    out = out | sma_valid

    print("Bowel wall edge-like voxels removed:", int(total_remove.sum()))

    return out

def prune_fragmented_air_lumen_rim_final(
    artery_tree,
    ct_crop,
    sma_valid,
    air_hu=-500,
    air_radius=5,
    protect_radius=1,
    group_radius=1,
    min_area=3,
    min_long_axis=6,
    min_aspect=1.5,
    axes=(0, 1, 2),
):
    """
    Final-only cleanup.
    Remove fragmented air-lumen rim / bowel-edge false positives.

    Key idea:
    - fragmented red edge pieces may be too small individually
    - group nearby suspect voxels first
    - remove only original artery voxels inside grouped rim-like components
    """

    out = artery_tree.astype(bool).copy()

    air = ct_crop <= air_hu
    near_air = binary_dilation(air, ball(air_radius))

    # True vessels may touch bowel wall.
    # Protect only very local contact points from vessels coming from outside near-air zone.
    outside_air_vessel = out & (~near_air)
    if protect_radius > 0:
        contact_protect = binary_dilation(outside_air_vessel, ball(protect_radius)) & near_air
    else:
        contact_protect = np.zeros_like(out, dtype=bool)

    contact_protect = contact_protect | binary_dilation(sma_valid, ball(3))

    suspect = out & near_air & (~sma_valid)

    total_remove = np.zeros_like(out, dtype=bool)
    structure2d = ndi.generate_binary_structure(2, 2)

    for axis in axes:
        for z in range(out.shape[axis]):
            s = np.take(suspect, z, axis=axis)
            p = np.take(contact_protect, z, axis=axis)

            # Group fragmented rim pieces before shape analysis
            grouped = s.copy()
            if group_radius > 0:
                for _ in range(group_radius):
                    grouped = ndi.binary_dilation(grouped, structure=structure2d)
                grouped = ndi.binary_closing(grouped, structure=structure2d)

            labeled, n = ndi.label(grouped, structure=structure2d)
            remove_slice = np.zeros_like(s, dtype=bool)

            for lab in range(1, n + 1):
                group = labeled == lab

                # Only remove original artery voxels, not the dilated grouping mask
                orig = s & group
                area = int(orig.sum())

                if area < min_area:
                    continue

                coords = np.argwhere(group)
                if len(coords) < 3:
                    continue

                span = coords.max(axis=0) - coords.min(axis=0) + 1
                long_axis = int(span.max())
                short_axis = int(max(span.min(), 1))
                aspect = long_axis / short_axis

                # fragmented lumen rim: grouped shape is elongated near air
                if long_axis >= min_long_axis and aspect >= min_aspect:
                    remove_slice |= (orig & (~p))

            if axis == 0:
                total_remove[z, :, :] |= remove_slice
            elif axis == 1:
                total_remove[:, z, :] |= remove_slice
            else:
                total_remove[:, :, z] |= remove_slice

    total_remove = total_remove & (~sma_valid)

    out = out & (~total_remove)
    out = out | sma_valid

    print("Fragmented air-lumen rim voxels removed:", int(total_remove.sum()))

    return out
def connect_local_endpoint_gaps_safe(
    artery_tree,
    candidate_core,
    candidate_grow,
    artery_territory,
    vein_exclude,
    sma_valid,
    ct_crop,
    bowel_crop=None,
    bowel_interior_radius=2,
    max_dist=16,
    min_align=0.25,
    max_cost_factor=5.0,
    max_connections=30,
    path_radius=0,
):
    """
    Connect short local gaps between vessel endpoints.
    This is different from connecting detached components.
    It is designed for visually broken vessels within the same vessel tree.
    """

    out = artery_tree.astype(bool).copy()

    # Safe path mask: prefer core candidate, allow grow candidate only locally.
    safe_mask = (candidate_core | candidate_grow)
    safe_mask = safe_mask & artery_territory
    safe_mask = safe_mask & (~vein_exclude)
    safe_mask = safe_mask | sma_valid

    # Do not pass through bowel interior.
    if bowel_crop is not None:
        bowel_mask = bowel_crop.astype(bool)
        if bowel_interior_radius > 0:
            bowel_interior = ndi.binary_erosion(
                bowel_mask,
                structure=ball(bowel_interior_radius),
            )
        else:
            bowel_interior = bowel_mask

        safe_mask = safe_mask & (~bowel_interior)
        safe_mask = safe_mask | sma_valid

    skel = skeletonize_3d(out).astype(bool)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = ndi.convolve(
        skel.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

    endpoints = skel & (neighbor_count == 1)
    endpoints = endpoints & (~binary_dilation(sma_valid, ball(3)))

    ep_coords = np.argwhere(endpoints)

    if len(ep_coords) < 2:
        print("Local endpoint gap connections:", 0)
        return out

    def endpoint_dir(p, r=5):
        p = np.asarray(p)
        lo = np.maximum(p - r, 0)
        hi = np.minimum(p + r + 1, skel.shape)
        slc = tuple(slice(lo[d], hi[d]) for d in range(3))

        pts = np.argwhere(skel[slc]) + lo
        if len(pts) < 3:
            return None

        pts0 = pts - p
        cov = pts0.T @ pts0
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]

        mean_vec = pts0.mean(axis=0)

        if np.dot(v, mean_vec) < 0:
            v = -v

        # v from endpoint toward the core vessel components
        v = -v

        return v / (np.linalg.norm(v) + 1e-6)

    dirs = [endpoint_dir(p) for p in ep_coords]

    tree = cKDTree(ep_coords)
    pairs = list(tree.query_pairs(r=max_dist))
    pairs.sort(key=lambda ij: np.linalg.norm(ep_coords[ij[0]] - ep_coords[ij[1]]))

    used = set()
    connected = 0

    for i, j in pairs:
        if i in used or j in used:
            continue

        p1 = ep_coords[i]
        p2 = ep_coords[j]

        v1 = dirs[i]
        v2 = dirs[j]

        if v1 is None or v2 is None:
            continue

        gap = p2 - p1
        dist = np.linalg.norm(gap)

        if dist < 2 or dist > max_dist:
            continue

        gap_dir = gap / (dist + 1e-6)

        # Endpoint directions should face each other.
        if np.dot(v1, gap_dir) < min_align:
            continue
        if np.dot(v2, -gap_dir) < min_align:
            continue

        lo = np.maximum(np.minimum(p1, p2) - max_dist - 3, 0)
        hi = np.minimum(np.maximum(p1, p2) + max_dist + 4, out.shape)

        slc = tuple(slice(lo[d], hi[d]) for d in range(3))

        local_core = candidate_core[slc]
        local_grow = candidate_grow[slc]
        local_safe = safe_mask[slc]

        local_start = tuple(p1 - lo)
        local_end = tuple(p2 - lo)

        # Prefer core candidate, allow grow candidate, forbid everything else.
        cost = np.full(local_safe.shape, 1e6, dtype=np.float32)
        cost[local_grow & local_safe] = 3.0
        cost[local_core & local_safe] = 1.0

        try:
            path, cost_val = route_through_array(
                cost,
                local_start,
                local_end,
                fully_connected=False,  # 6-connectivity path
            )
        except Exception:
            continue

        if cost_val > dist * max_cost_factor:
            continue

        path = np.array(path) + lo

        if path_radius > 0:
            tube = np.zeros_like(out, dtype=bool)
            tube[tuple(path.T)] = True
            tube = binary_dilation(tube, ball(path_radius))
            tube = tube & safe_mask
            out = out | tube
        else:
            out[tuple(path.T)] = True

        used.add(i)
        used.add(j)
        connected += 1

        if connected >= max_connections:
            break

    out = out | sma_valid

    print("Local endpoint gap connections:", connected)

    return out

def absorb_small_candidate_gaps(
    artery_tree,
    final_connect_candidate,
    sma_valid,
    vein_exclude,
    hard_forbid=None,
    near_radius=3,
    max_gap_size=300,
    min_contact_voxels=2,
    max_components=50,
):
    """
    Add small missing candidate components that sit right next to the current artery tree.
    This is for gaps that are already present in final_connect_candidate but missing in artery_tree.
    """

    out = artery_tree.astype(bool).copy()

    if hard_forbid is None:
        hard_forbid = np.zeros_like(out, dtype=bool)
    else:
        hard_forbid = hard_forbid.astype(bool)

    main = keep_connected_to_seed_6(out | sma_valid, sma_valid)

    near_tree = binary_dilation(main, ball(near_radius))

    missing = (
        final_connect_candidate &
        (~out) &
        near_tree &
        (~vein_exclude) &
        (~hard_forbid)
    )

    structure26 = ndi.generate_binary_structure(3, 3)
    labeled, n = ndi.label(missing, structure=structure26)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    labels = np.argsort(sizes)[::-1]

    added_total = 0
    added_comps = 0

    tree_touch_zone = binary_dilation(main, ball(1))

    for lab in labels:
        if lab == 0:
            continue

        size = int(sizes[lab])

        if size == 0 or size > max_gap_size:
            continue

        comp = labeled == lab

        contact = comp & tree_touch_zone
        contact_voxels = int(contact.sum())

        if contact_voxels < min_contact_voxels:
            continue

        test_out = out | comp | sma_valid
        test_main = keep_connected_to_seed_6(test_out, sma_valid)

        # Keep only if this candidate gap becomes part of the SMA-connected tree
        if np.any(comp & test_main):
            out = test_out
            added_total += size
            added_comps += 1

        if added_comps >= max_components:
            break

    print(
        f"Small candidate gaps absorbed: comps={added_comps}, voxels={added_total}"
    )

    return out

def main():
    start_time = time.time()
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", default="CT image path")
    parser.add_argument("--seg", default="00004_00001_3_sma_smv.nii.gz")
    parser.add_argument("--out", default="00004_00001_3_gt_algo.nii.gz")
    parser.add_argument("--hu_min", type=float, default=95)
    parser.add_argument("--hu_max", type=float, default=370)
    parser.add_argument("--vesselness_thr", type=float, default=0.0012)
    parser.add_argument("--margin", type=int, default=95)
    parser.add_argument("--max_iter", type=int, default=700)
    parser.add_argument("--min_size", type=int, default=25)
    parser.add_argument("--vein_block_radius", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--connect_branches", type=int, default=0)
    parser.add_argument("--air_hu", type=float, default=-500)
    parser.add_argument("--air_radius", type=int, default=2)
    parser.add_argument("--bowel_seg", default=None)
    parser.add_argument("--bowel_interior_radius", type=int, default=2)
    args = parser.parse_args()

    ct, affine, header = load_nifti(args.ct)

    print("shape:", ct.shape)
    print("affine:")
    print(affine)

    sma_smv, _, _ = load_nifti(args.seg)

    sma = sma_smv == 1
    vein = (sma_smv == 2) | (sma_smv == 3)

    roi_mask = sma | vein
    slices = crop_bbox(roi_mask, margin=args.margin)
    ct_crop = ct[slices]
    sma_crop = sma[slices]
    vein_crop = vein[slices]

    bowel_crop = None

    # ITK-SNAP cursor coordinate
    full_xyz = np.array([156, 268, 345])
    starts = np.array([s.start for s in slices])
    crop_xyz = full_xyz - starts

    cursor_inside = np.all(crop_xyz >= 0) and np.all(crop_xyz < np.array(ct_crop.shape))

    if cursor_inside:
        x, y, z = crop_xyz.astype(int)
    else:
        x, y, z = 0, 0, 0

    if args.bowel_seg is not None:
        bowel_seg, _, _ = load_nifti(args.bowel_seg)
        bowel = bowel_seg > 0
        bowel_crop = bowel[slices]
        print("Bowel voxels in crop:", int(bowel_crop.sum()))

    sma_valid = sma_crop & (ct_crop >= 0)
    vein_valid = vein_crop & (ct_crop >= 0)

    print("Artery seed voxels:", int(sma_valid.sum()))
    print("Vein seed voxels:", int(vein_valid.sum()))

    candidate_core, candidate, artery_territory = make_vessel_candidate(
        ct_crop=ct_crop,
        sma_valid=sma_valid,
        vein_valid=vein_valid,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        vesselness_thr=args.vesselness_thr,
        use_frangi=True,
        territory_ratio=2.0,
        dilate_radius=1,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
    )

    print("Candidate core voxels:", int(candidate_core.sum()))
    print("Candidate grow voxels:", int(candidate.sum()))

    vein_exclude = binary_dilation(vein_valid, ball(args.vein_block_radius))
    vein_exclude = vein_exclude & (~sma_valid)
    print("Vein-exclude voxels:", int(vein_exclude.sum()))

    candidate = candidate & (~vein_exclude)
    candidate = candidate | sma_valid

    candidate_core = candidate_core & (~vein_exclude)
    candidate_core = candidate_core | sma_valid

    print("Growing SMA arterial tree with competitive vein growth...")
    artery_tree, vein_tree = competitive_grow_gpu(
        candidate=candidate,
        sma_seed=sma_valid,
        vein_seed=vein_valid,
        max_iter=args.max_iter,
        device=args.device,
    )

    print("Raw artery tree voxels:", int(artery_tree.sum()))
    print("Raw vein tree voxels:", int(vein_tree.sum()))

    artery_tree = filter_by_sma_cross_section_area(
        vessel=artery_tree,
        sma=sma_valid,
        axis=2,
        scale=1.8,
    )
    print("After area filter voxels:", int(artery_tree.sum()))

    artery_tree = keep_connected_to_seed(candidate=artery_tree | sma_valid, seed=sma_valid)
    print("After keep connected voxels:", int(artery_tree.sum()))

    artery_tree = artery_tree & (~vein_tree)
    artery_tree = remove_small_objects(artery_tree.astype(bool), min_size=args.min_size)
    artery_tree = artery_tree | sma_valid
    print("After vein/small-object removal voxels:", int(artery_tree.sum()))

    # ============================================================
    # Strict cleanup only
    # No skeleton refinement, no endpoint gap fill, no branch connection
    # ============================================================

    near_tree = binary_dilation(artery_tree, ball(1))

    strict_mask = candidate_core | (candidate & near_tree)
    strict_mask = strict_mask & artery_territory
    strict_mask = strict_mask & (~vein_exclude)
    strict_mask = strict_mask | sma_valid

    # Do not add new voxels.
    # Only keep the already-grown artery inside the strict candidate mask.
    artery_tree = artery_tree & strict_mask
    artery_tree = artery_tree | sma_valid

    artery_tree = remove_small_objects(
        artery_tree.astype(bool),
        min_size=args.min_size
    )
    artery_tree = remove_bowel_boundary_like_components(
        mask=artery_tree,
        ct_crop=ct_crop,
        protected=sma_valid,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        min_area=20,
        min_long_axis=12,
        min_aspect=2.5,
        axis=2,
    )

    artery_tree = artery_tree | sma_valid

    print("After strict cleanup voxels:", int(artery_tree.sum()))

    artery_tree = bridge_tiny_gaps_with_grow_candidate(
        artery_tree=artery_tree,
        candidate_grow=candidate,
        candidate_core=candidate_core,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        n_iter=2,
    )

    print("After tiny gap bridge voxels:", int(artery_tree.sum()))

    artery_tree = connect_detached_large_components(
        artery_tree=artery_tree,
        candidate_grow=candidate,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        min_comp_size=200,
        max_dist=25,
        max_components=5,
    )
    print("DEBUG artery_tree at cursor after final detached connection:", bool(artery_tree[x, y, z]))
    print("After detached component connection voxels:", int(artery_tree.sum()))

    # Remove only bowel interior, not bowel wall-adjacent vessels
    artery_tree = remove_bowel_interior_only(
        artery_tree=artery_tree,
        bowel_crop=bowel_crop,
        sma_valid=sma_valid,
        interior_radius=args.bowel_interior_radius,
    )

    print("After bowel interior removal voxels:", int(artery_tree.sum()))
    artery_tree = prune_bowel_wall_edges_keep_vessel_contacts(
        artery_tree=artery_tree,
        bowel_crop=bowel_crop,
        sma_valid=sma_valid,
        interior_radius=args.bowel_interior_radius,
        wall_radius=1,
        protect_radius=2,
        min_area=20,
        min_long_axis=10,
        min_aspect=2.0,
        axes=(2,),
    )

    print("After bowel wall-edge cleanup voxels:", int(artery_tree.sum()))
    artery_tree = prune_fragmented_air_lumen_rim_final(
        artery_tree=artery_tree,
        ct_crop=ct_crop,
        sma_valid=sma_valid,
        air_hu=args.air_hu,
        air_radius=5,
        protect_radius=1,
        group_radius=1,
        min_area=3,
        min_long_axis=6,
        min_aspect=1.5,
        axes=(0, 1, 2),
    )
    print("After fragmented air-lumen rim cleanup voxels:", int(artery_tree.sum()))

    artery_tree = connect_detached_large_components(
        artery_tree=artery_tree,
        candidate_grow=candidate,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        min_comp_size=300,
        max_dist=25,
        max_components=10,
    )
    print("DEBUG artery_tree at cursor after final detached connection:", bool(artery_tree[x, y, z]))
    print("After post-cleanup detached component connection voxels:", int(artery_tree.sum()))

    artery_tree = connect_local_endpoint_gaps_safe(
        artery_tree=artery_tree,
        candidate_core=candidate_core,
        candidate_grow=candidate,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        bowel_crop=bowel_crop,
        bowel_interior_radius=args.bowel_interior_radius,
        max_dist=22,
        min_align=0.15,
        max_cost_factor=8.0,
        max_connections=15,
        path_radius=1,
    )

    print("After post-cleanup local endpoint gap connection voxels:", int(artery_tree.sum()))

    # ============================================================
    # FINAL cleanup again after all connection steps
    # Do not run any connection after this block
    # ============================================================

    artery_tree = remove_bowel_interior_only(
        artery_tree=artery_tree,
        bowel_crop=bowel_crop,
        sma_valid=sma_valid,
        interior_radius=args.bowel_interior_radius,
    )

    artery_tree = prune_bowel_wall_edges_keep_vessel_contacts(
        artery_tree=artery_tree,
        bowel_crop=bowel_crop,
        sma_valid=sma_valid,
        interior_radius=args.bowel_interior_radius,
        wall_radius=2,
        protect_radius=1,
        min_area=10,
        min_long_axis=8,
        min_aspect=1.5,
        axes=(0, 1, 2),
    )
    artery_tree = prune_fragmented_air_lumen_rim_final(
        artery_tree=artery_tree,
        ct_crop=ct_crop,
        sma_valid=sma_valid,
        air_hu=args.air_hu,
        air_radius=5,
        protect_radius=1,
        group_radius=1,
        min_area=3,
        min_long_axis=6,
        min_aspect=1.5,
        axes=(0, 1, 2),
    )

    print("After FINAL bowel / lumen cleanup voxels:", int(artery_tree.sum()))

    near_final_tree = binary_dilation(artery_tree, ball(6))

    hu_bridge_min = max(30, args.hu_min - 40)

    hu_bridge = (
        (ct_crop >= hu_bridge_min) &
        (ct_crop <= args.hu_max)
    )

    final_connect_candidate = (
        ((candidate_core | candidate | hu_bridge) & near_final_tree) |
        artery_tree |
        sma_valid
    )

    final_connect_candidate = final_connect_candidate & artery_territory
    final_connect_candidate = final_connect_candidate & (~vein_exclude)
    final_connect_candidate = final_connect_candidate | artery_tree | sma_valid


    bowel_interior_for_bridge = None

    if bowel_crop is not None:
        bowel_mask = bowel_crop.astype(bool)
        bowel_interior_for_bridge = ndi.binary_erosion(
            bowel_mask,
            structure=ball(args.bowel_interior_radius),
        )
        final_connect_candidate = final_connect_candidate & (~bowel_interior_for_bridge)
        final_connect_candidate = final_connect_candidate | artery_tree | sma_valid

    artery_tree = connect_local_endpoint_gaps_safe(
        artery_tree=artery_tree,
        candidate_core=final_connect_candidate,
        candidate_grow=final_connect_candidate,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        bowel_crop=bowel_crop,
        bowel_interior_radius=args.bowel_interior_radius,
        max_dist=14,
        min_align=0.0,
        max_cost_factor=12.0,
        max_connections=30,
        path_radius=1,
    )

    print("After AUTO final endpoint gap filling voxels:", int(artery_tree.sum()))
    
    artery_tree = absorb_small_candidate_gaps(
        artery_tree=artery_tree,
        final_connect_candidate=final_connect_candidate,
        sma_valid=sma_valid,
        vein_exclude=vein_exclude,
        hard_forbid=bowel_interior_for_bridge,
        near_radius=3,
        max_gap_size=800,
        min_contact_voxels=1,
        max_components=20,
    )
    print("DEBUG artery_tree at cursor after absorption:", bool(artery_tree[x, y, z]))
    print("After small candidate gap absorption voxels:", int(artery_tree.sum()))


    print("Full xyz:", full_xyz)
    print("Crop starts:", starts)
    print("Crop xyz:", crop_xyz)
    print("Crop shape:", ct_crop.shape)

    

    if np.all(crop_xyz >= 0) and np.all(crop_xyz < np.array(ct_crop.shape)):
        print("CT HU at cursor:", ct_crop[x, y, z])
        print("candidate_core:", bool(candidate_core[x, y, z]))
        print("candidate_grow:", bool(candidate[x, y, z]))
        print("artery_territory:", bool(artery_territory[x, y, z]))
        print("vein_exclude:", bool(vein_exclude[x, y, z]))

        if bowel_crop is not None:
            bowel_mask = bowel_crop.astype(bool)
            bowel_interior = ndi.binary_erosion(
                bowel_mask,
                structure=ball(args.bowel_interior_radius),
            )
            print("bowel_crop:", bool(bowel_crop[x, y, z]))
            print("bowel_interior:", bool(bowel_interior[x, y, z]))

        near_final_tree = binary_dilation(artery_tree, ball(6))
        print("near_final_tree:", bool(near_final_tree[x, y, z]))

        hu_bridge_min = max(30, args.hu_min - 40)
        hu_ok = (ct_crop[x, y, z] >= hu_bridge_min) and (ct_crop[x, y, z] <= args.hu_max)
        print("hu_bridge_min:", hu_bridge_min)
        print("HU bridge OK:", bool(hu_ok))
    else:
        print("Cursor is outside crop.")

    artery_tree = connect_detached_large_components(
        artery_tree=artery_tree,
        candidate_grow=final_connect_candidate,
        artery_territory=artery_territory,
        vein_exclude=vein_exclude,
        sma_valid=sma_valid,
        ct_crop=ct_crop,
        air_hu=args.air_hu,
        air_radius=args.air_radius,
        min_comp_size=300,
        max_dist=25,
        max_components=8,
        apply_bowel_filter=False,
        direct_bridge_hu_min=30,
        direct_bridge_hu_max=args.hu_max,
        hard_forbid=bowel_interior_for_bridge,
    )
    print("After FINAL safe detached connection voxels:", int(artery_tree.sum()))
    print("After FINAL safe detached connection voxels:", int(artery_tree.sum()))
    print("DEBUG artery_tree at cursor after final detached connection:", bool(artery_tree[x, y, z]))

    structure6_dbg = ndi.generate_binary_structure(3, 1)

    labeled_dbg, n_dbg = ndi.label(artery_tree, structure=structure6_dbg)

    sizes_dbg = np.bincount(labeled_dbg.ravel())
    sizes_dbg[0] = 0
    top_dbg = sorted(sizes_dbg[sizes_dbg > 0], reverse=True)[:20]

    print("Before keep_connected_to_seed_6 components:", n_dbg)
    print("Top 20 sizes before keep_connected_to_seed_6:", top_dbg)

    artery_tree = keep_connected_to_seed_6(
        candidate=artery_tree | sma_valid,
        seed=sma_valid
    )
    structure6 = ndi.generate_binary_structure(3, 1)
    labeled6, n6_pre = ndi.label(artery_tree, structure=structure6)

    sizes6 = np.bincount(labeled6.ravel())
    sizes6[0] = 0
    top_sizes6 = sorted(sizes6[sizes6 > 0], reverse=True)[:10]

    print("Before final 6-connectivity components:", n6_pre)
    print("Top 10 component sizes before final:", top_sizes6)

    structure6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(artery_tree, structure=structure6)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    main_label = sizes.argmax()
    artery_tree = labeled == main_label

    artery_tree = artery_tree | sma_valid

    structure6 = ndi.generate_binary_structure(3, 1)
    labeled, n = ndi.label(artery_tree, structure=structure6)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    main_label = sizes.argmax()
    artery_tree = labeled == main_label

    _, n6 = ndi.label(artery_tree, structure=structure6)
    print("Final connected components 6-connectivity:", n6)

    result = np.zeros(sma.shape, dtype=np.uint8)
    result[slices] = artery_tree.astype(np.uint8)
    save_nifti(result, affine, header, args.out)


    debug_core = np.zeros(sma.shape, dtype=np.uint8)
    debug_core[slices] = candidate_core.astype(np.uint8)
    save_nifti(debug_core, affine, header, "debug_candidate_core.nii.gz")

    debug_grow = np.zeros(sma.shape, dtype=np.uint8)
    debug_grow[slices] = candidate.astype(np.uint8)
    save_nifti(debug_grow, affine, header, "debug_candidate_grow.nii.gz")

    debug_final = np.zeros(sma.shape, dtype=np.uint8)
    debug_final[slices] = final_connect_candidate.astype(np.uint8)
    save_nifti(debug_final, affine, header, "debug_final_connect_candidate.nii.gz")

    print("DEBUG artery_tree at cursor:", bool(artery_tree[x, y, z]))
    print("DEBUG final_connect_candidate:", bool(final_connect_candidate[x, y, z]))

    print("Saved:", args.out)
    print("Output voxels:", int(result.sum()))
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Total runtime: {minutes} min {seconds:.2f} sec")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
