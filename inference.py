

import json
import os
import pickle
import time
import warnings
from glob import glob
from pathlib import Path
from scipy import ndimage

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_PATH  = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH  = Path("/opt/ml/model")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOWNSAMPLE   = 4
TARGET_SHAPE = (78, 50, 50)
T_NAMES      = ["T1", "T2", "T3", "T4"]
N_NAMES      = ["N0", "N1", "N2", "N3"]
CROP_BOX_SIZE = (200, 200, 310)

CLINICAL_DIM_TN  = 30   # task2_v7 — has AJCC N-proxy features appended
CLINICAL_DIM_RFS = 26   # original task3 — no AJCC proxies

NNUNET_DATASET_ID = 100
NNUNET_CONFIG     = "3d_fullres"
NNUNET_PLANS      = "nnUNetResEncUNetLPlans"
NNUNET_TRAINER    = "nnUNetTrainer_250epochs"
NNUNET_FOLDS      = (0, 1, 2, 3, 4)

# Used for segmentation mask post-processing (cleaning predicted
# components) — NOT for either radiomics pipeline's node counting.
FULL_CONNECTIVITY_3D = ndimage.generate_binary_structure(3, 3)

# --- TN staging (task2_v7) normalization constants -------------------------
# Confirmed by direct recomputation from clinical_with_radiomics_pred.csv:
# out[col] = df[col].fillna(0.0) / (p99 + 1e-8), p99 = df[col].quantile(0.99)
# computed ONCE over the full training set (782 patients). Frozen.
RADIOMICS_P99_TN = {
    "GTVp_volume_cm3":  58.415579999999885,
    "GTVn_volume_cm3":  84.93970999999995,
    "GTVp_max_diam_mm": 65.4795929254689,
    "GTVn_max_diam_mm": 142.71604976776328,
    "GTVn_node_count":  8.0,
    "GTVp_SUVmax":      44601.94746093685,
    "GTVn_SUVmax":      42224.06285156241,
    "GTVp_SUVmean":     22023.230624999946,
    "GTVn_SUVmean":     17961.93808593746,
    "GTVp_TLG":         152734.7233574974,
    "GTVn_TLG":         273401.8183658203,
}
AGE_MEAN_TN = 59.987468030690536
AGE_STD_TN  = 9.211645553938592


# =============================================================================
# Preprocessing — SHARED by all three subtasks, run ONCE
# =============================================================================

def get_bounding_boxes(ct_sitk, pet_sitk):
    ct_origin  = np.array(ct_sitk.GetOrigin())
    pet_origin = np.array(pet_sitk.GetOrigin())
    ct_max  = ct_origin  + np.array(ct_sitk.GetSize())  * np.array(ct_sitk.GetSpacing())
    pet_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())
    return np.concatenate([np.maximum(ct_origin, pet_origin),
                           np.minimum(ct_max, pet_max)])


def ensure_3d(img_sitk):
    arr = sitk.GetArrayFromImage(img_sitk)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
        out = sitk.GetImageFromArray(arr)
        out.SetSpacing(list(img_sitk.GetSpacing()) + [1.0])
        out.SetOrigin(list(img_sitk.GetOrigin()) + [0.0])
        out.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        return out
    return img_sitk


def resample_images(ct_sitk, pet_sitk):
    ct_sitk  = ensure_3d(ct_sitk)
    pet_sitk = ensure_3d(pet_sitk)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing([1, 1, 1])
    bb   = get_bounding_boxes(ct_sitk, pet_sitk)
    size = np.round((bb[3:] - bb[:3])).astype(int)
    size = np.maximum(size, 1)
    resampler.SetOutputOrigin(bb[:3].tolist())
    resampler.SetSize([int(k) for k in size])
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct_sitk)
    pt = resampler.Execute(pet_sitk)
    return ct, pt


def register_pet_to_ct(ct_sitk, pet_sitk):
    ct_f  = sitk.Cast(ct_sitk,  sitk.sitkFloat32)
    pet_f = sitk.Cast(pet_sitk, sitk.sitkFloat32)
    transform = sitk.CenteredTransformInitializer(
        ct_f, pet_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform)
    final_transform = reg.Execute(ct_f, pet_f)
    return sitk.Resample(pet_sitk, ct_sitk, final_transform, sitk.sitkLinear, 0.0, pet_sitk.GetPixelID())


def get_roi_center(pet_np_xyz, z_top_fraction=0.75, z_score_threshold=1.0):
    crop_z_start = int(z_top_fraction * pet_np_xyz.shape[2])
    top_of_scan  = pet_np_xyz[..., crop_z_start:]
    std = top_of_scan.std()
    mask = ((top_of_scan - top_of_scan.mean()) / (std + 1e-8)) > z_score_threshold

    if not mask.any():
        warnings.warn("No high-intensity region found. Using geometric center.")
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        labeled_mask, num_features = ndimage.label(mask, structure=FULL_CONNECTIVITY_3D)
        if num_features > 0:
            sizes    = np.bincount(labeled_mask.ravel())[1:]
            comp_idx = np.argwhere(labeled_mask == np.argmax(sizes) + 1)
        else:
            comp_idx = np.argwhere(mask)
        center_in_top = np.mean(comp_idx, axis=0).astype(int)

    return (center_in_top + np.array([0, 0, crop_z_start])).astype(int)


def crop_neck_region(ct_sitk, pet_sitk, crop_box_size=CROP_BOX_SIZE):
    pet_np_zyx = sitk.GetArrayFromImage(pet_sitk)
    pet_np_xyz = np.transpose(pet_np_zyx, (2, 1, 0))

    crop_box_size = np.asarray(crop_box_size, dtype=int)
    center    = get_roi_center(pet_np_xyz)
    img_shape = np.asarray(pet_np_xyz.shape)

    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    box_end   = np.clip(box_start + crop_box_size,   0, img_shape)
    box_start = np.maximum(box_end - crop_box_size, 0)

    index = [int(i) for i in box_start]
    size  = [int(e - s) for s, e in zip(box_start, box_end)]

    ct_crop  = sitk.RegionOfInterest(ct_sitk,  size=size, index=index)
    pet_crop = sitk.RegionOfInterest(pet_sitk, size=size, index=index)
    return ct_crop, pet_crop, box_start, box_end


def preprocess_once(ct_path, pet_path):
    ct_sitk  = sitk.ReadImage(str(ct_path))
    pet_sitk = sitk.ReadImage(str(pet_path))

    print(f"[PREP] CT size={ct_sitk.GetSize()} dim={ct_sitk.GetDimension()}")
    print(f"[PREP] PET size={pet_sitk.GetSize()} dim={pet_sitk.GetDimension()}")

    ct_sitk  = ensure_3d(ct_sitk)
    pet_sitk = ensure_3d(pet_sitk)

    try:
        ct_rs, pet_rs = resample_images(ct_sitk, pet_sitk)
        print(f"[PREP] Resampled OK: CT={ct_rs.GetSize()} PET={pet_rs.GetSize()}")
    except Exception as e:
        print(f"[PREP] Direct resample failed: {e}")
        try:
            pet_reg = register_pet_to_ct(ct_sitk, pet_sitk)
            ct_rs, pet_rs = resample_images(ct_sitk, pet_reg)
            print(f"[PREP] Registration fallback resample OK")
        except Exception as e2:
            print(f"[PREP] Fallback also failed: {e2} — using raw images")
            ct_rs, pet_rs = ct_sitk, pet_sitk

    ct_crop, pet_crop, box_start, box_end = crop_neck_region(ct_rs, pet_rs)
    print(f"[PREP] Crop size: {ct_crop.GetSize()}")

    return {
        "ct_sitk":  ct_sitk,
        "ct_rs":    ct_rs,
        "ct_crop":  ct_crop,
        "pet_crop": pet_crop,
        "box_start": box_start,
        "box_end":   box_end,
    }


def downsample_and_pad(arr, factor=DOWNSAMPLE, target=TARGET_SHAPE):
    ds = arr[::factor, ::factor, ::factor]
    result = np.zeros(target, dtype=arr.dtype)
    slices_src, slices_dst = [], []
    for i in range(3):
        s, t = ds.shape[i], target[i]
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    result[tuple(slices_dst)] = ds[tuple(slices_src)]
    return result


def prepare_tensors(ct_arr, pet_arr, mask_arr):
    ct  = np.clip(ct_arr, -200, 300)
    ct  = (ct  - ct.mean())  / (ct.std()  + 1e-8)
    pet = (pet_arr - pet_arr.mean()) / (pet_arr.std() + 1e-8)

    ct_ds  = downsample_and_pad(ct)
    pet_ds = downsample_and_pad(pet)
    msk_ds = downsample_and_pad(mask_arr)

    img = torch.tensor(np.stack([ct_ds, pet_ds], axis=0)[None], dtype=torch.float32)
    mask_onehot = torch.tensor(np.stack([
        (msk_ds == 0).astype(np.float32),
        (msk_ds == 1).astype(np.float32),
        (msk_ds == 2).astype(np.float32)
    ], axis=0)[None], dtype=torch.float32)
    return img, mask_onehot


def get_max_diameter_mm(binary_mask):
    """1x1x1mm spacing guaranteed by resample_images, so no explicit
    voxel_spacing multiply is needed here — voxel-index distance == mm."""
    if binary_mask.sum() == 0:
        return 0.0
    coords = np.argwhere(binary_mask).astype(np.float32)
    if len(coords) > 2000:
        idx = np.random.choice(len(coords), 2000, replace=False)
        coords = coords[idx]
    max_dist = 0.0
    for i in range(0, len(coords), 10):
        diffs = coords[i+1:] - coords[i]
        dists = np.sqrt((diffs**2).sum(axis=1))
        if len(dists) > 0:
            max_dist = max(max_dist, dists.max())
    return float(max_dist)


# =============================================================================
# Radiomics — TN STAGING VERSION (matches compute_radiomics.py exactly:
# ndimage.label(gtvn) with NO structure= arg -> scipy default 6-connectivity)
# =============================================================================

def extract_radiomics_tn(mask_arr, pet_arr):
    feats = {k: 0.0 for k in [
        "GTVp_volume_cm3", "GTVn_volume_cm3",
        "GTVp_SUVmax", "GTVn_SUVmax",
        "GTVp_SUVmean", "GTVn_SUVmean",
        "GTVp_max_diam_mm", "GTVn_max_diam_mm",
        "GTVn_node_count", "GTVp_TLG", "GTVn_TLG"
    ]}
    voxel_vol = 0.001

    gtvp = (mask_arr == 1)
    gtvn = (mask_arr == 2)

    if gtvp.sum() > 0:
        feats["GTVp_volume_cm3"] = float(gtvp.sum() * voxel_vol)
        pet_p = pet_arr[gtvp]
        feats["GTVp_SUVmax"]      = float(pet_p.max())
        feats["GTVp_SUVmean"]     = float(pet_p.mean())
        feats["GTVp_TLG"]         = feats["GTVp_volume_cm3"] * feats["GTVp_SUVmean"]
        feats["GTVp_max_diam_mm"] = get_max_diameter_mm(gtvp)

    if gtvn.sum() > 0:
        feats["GTVn_volume_cm3"] = float(gtvn.sum() * voxel_vol)
        pet_n = pet_arr[gtvn]
        feats["GTVn_SUVmax"]      = float(pet_n.max())
        feats["GTVn_SUVmean"]     = float(pet_n.mean())
        feats["GTVn_TLG"]         = feats["GTVn_volume_cm3"] * feats["GTVn_SUVmean"]
        feats["GTVn_max_diam_mm"] = get_max_diameter_mm(gtvn)
        # 6-connectivity — matches compute_radiomics.py exactly.
        _, n_comp = ndimage.label(gtvn)
        feats["GTVn_node_count"] = float(n_comp)

    return feats


def compute_ajcc_proxies(radiomics):
    gtvn_vol   = radiomics.get("GTVn_volume_cm3", 0.0)
    gtvn_count = radiomics.get("GTVn_node_count", 0.0)
    gtvn_diam  = radiomics.get("GTVn_max_diam_mm", 0.0)

    n0 = float(gtvn_vol < 0.5)
    n1 = float((gtvn_count == 1) and (gtvn_diam < 30))
    n2 = float(((gtvn_count > 1) or (gtvn_diam >= 30)) and (gtvn_diam < 60))
    n3 = float(gtvn_diam >= 60)
    return {"N0_proxy": n0, "N1_proxy": n1, "N2_proxy": n2, "N3_proxy": n3}


EHR_KEY_ALIASES = {
    "age":                ["Age", "age"],
    "gender":             ["Gender", "gender"],
    "tobacco":            ["Tobacco Consumption", "Tobacco", "tobacco"],
    "alcohol":            ["Alcohol Consumption", "Alcohol", "alcohol"],
    "hpv":                ["HPV Status", "HPV", "hpv"],
    "treatment":          ["Treatment", "treatment"],
    "performance_status": ["Performance Status", "PerformanceStatus", "performance_status"],
    "center_id":          ["CenterID", "Center_ID", "center_id"],
}


def _encode_categorical(key, raw_value, encoders, default=-1.0):
    cat_maps = encoders.get("categorical_maps", {})
    if key in cat_maps and raw_value in cat_maps[key]:
        return float(cat_maps[key][raw_value])
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        warnings.warn(f"[CLIN-TN] Could not encode '{key}'={raw_value!r}; using default={default}.")
        return default


def build_clinical_vector_tn(ehr, radiomics, encoders):
    """TN staging (task2_v7) clinical vector — 30-dim, AJCC N-proxies
    appended, hardcoded p99/age normalization constants."""
    def get_ehr_raw(key, default=None):
        for alias in EHR_KEY_ALIASES.get(key, [key]):
            if alias in ehr:
                return ehr[alias]
        return default

    age_raw  = get_ehr_raw("age", AGE_MEAN_TN)
    age_norm = (float(age_raw) - AGE_MEAN_TN) / (AGE_STD_TN + 1e-8)

    center_raw = get_ehr_raw("center_id", 5)
    center_id  = int(_encode_categorical("center_id", center_raw, encoders, default=5))
    center_vec = [1.0 if i == center_id else 0.0 for i in range(1, 9)]

    def norm_rad(col_name, val):
        val = 0.0 if val is None else float(val)
        p99 = RADIOMICS_P99_TN[col_name]
        return val / (p99 + 1e-8)

    rad_vec = [
        norm_rad("GTVp_volume_cm3",  radiomics["GTVp_volume_cm3"]),
        norm_rad("GTVn_volume_cm3",  radiomics["GTVn_volume_cm3"]),
        norm_rad("GTVp_max_diam_mm", radiomics["GTVp_max_diam_mm"]),
        norm_rad("GTVn_max_diam_mm", radiomics["GTVn_max_diam_mm"]),
        norm_rad("GTVn_node_count",  radiomics["GTVn_node_count"]),
        norm_rad("GTVp_SUVmax",      radiomics["GTVp_SUVmax"]),
        norm_rad("GTVn_SUVmax",      radiomics["GTVn_SUVmax"]),
        norm_rad("GTVp_SUVmean",     radiomics["GTVp_SUVmean"]),
        norm_rad("GTVn_SUVmean",     radiomics["GTVn_SUVmean"]),
        norm_rad("GTVp_TLG",         radiomics["GTVp_TLG"]),
        norm_rad("GTVn_TLG",         radiomics["GTVn_TLG"]),
    ]

    ajcc = compute_ajcc_proxies(radiomics)
    proxy_vec = [ajcc["N0_proxy"], ajcc["N1_proxy"], ajcc["N2_proxy"], ajcc["N3_proxy"]]

    vec = [
        age_norm,
        _encode_categorical("gender", get_ehr_raw("gender"), encoders),
        _encode_categorical("tobacco", get_ehr_raw("tobacco"), encoders),
        _encode_categorical("alcohol", get_ehr_raw("alcohol"), encoders),
        _encode_categorical("hpv", get_ehr_raw("hpv"), encoders),
        _encode_categorical("treatment", get_ehr_raw("treatment"), encoders),
        _encode_categorical("performance_status", get_ehr_raw("performance_status"), encoders),
    ] + center_vec + rad_vec + proxy_vec

    assert len(vec) == CLINICAL_DIM_TN, (
        f"[CLIN-TN] Built {len(vec)}-dim vector but expected {CLINICAL_DIM_TN}.")
    print(f"[CLIN-TN] vec({len(vec)})={[round(v,3) for v in vec]}")
    return torch.tensor([vec], dtype=torch.float32)


# =============================================================================
# Radiomics — RFS/PROGNOSIS VERSION (matches the file that scored 0.6821:
# ndimage.label(gtvn, structure=FULL_CONNECTIVITY_3D) -> 26-connectivity)
# =============================================================================

def extract_radiomics_rfs(mask_arr, pet_arr):
    feats = {k: 0.0 for k in [
        "GTVp_volume_cm3", "GTVn_volume_cm3",
        "GTVp_SUVmax", "GTVn_SUVmax",
        "GTVp_SUVmean", "GTVn_SUVmean",
        "GTVp_max_diam_mm", "GTVn_max_diam_mm",
        "GTVn_node_count", "GTVp_TLG", "GTVn_TLG"
    ]}
    voxel_vol = 0.001

    gtvp = (mask_arr == 1)
    gtvn = (mask_arr == 2)

    if gtvp.sum() > 0:
        feats["GTVp_volume_cm3"] = float(gtvp.sum() * voxel_vol)
        pet_p = pet_arr[gtvp]
        feats["GTVp_SUVmax"]      = float(pet_p.max())
        feats["GTVp_SUVmean"]     = float(pet_p.mean())
        feats["GTVp_TLG"]         = feats["GTVp_volume_cm3"] * feats["GTVp_SUVmean"]
        feats["GTVp_max_diam_mm"] = get_max_diameter_mm(gtvp)

    if gtvn.sum() > 0:
        feats["GTVn_volume_cm3"] = float(gtvn.sum() * voxel_vol)
        pet_n = pet_arr[gtvn]
        feats["GTVn_SUVmax"]      = float(pet_n.max())
        feats["GTVn_SUVmean"]     = float(pet_n.mean())
        feats["GTVn_TLG"]         = feats["GTVn_volume_cm3"] * feats["GTVn_SUVmean"]
        # 26-connectivity — matches the validated 0.6821 RFS pipeline.
        _, n_comp = ndimage.label(gtvn, structure=FULL_CONNECTIVITY_3D)
        feats["GTVn_node_count"] = float(n_comp)
        feats["GTVn_max_diam_mm"] = get_max_diameter_mm(gtvn)

    return feats


def build_clinical_vector_rfs(ehr, radiomics, encoders):
    """Prognosis (original RFSModel) clinical vector — 26-dim, no AJCC
    proxies, radiomics/age stats loaded from clinical_encoders.pkl."""
    stats = encoders.get("radiomics_stats", {})
    if not stats:
        warnings.warn("[CLIN-RFS] encoders['radiomics_stats'] missing/empty — "
                       "falling back to p99=1.0 for all radiomics features. "
                       "This will NOT match the 0.6821 result. Check clinical_encoders.pkl.")

    def get_ehr(key, default=-1):
        for alias in EHR_KEY_ALIASES.get(key, [key]):
            if alias in ehr:
                v = ehr[alias]
                return float(v) if v is not None else default
        return default

    age_mean = encoders.get("age_mean", 60.0)
    age_std  = encoders.get("age_std",  10.0)
    if "age_mean" not in encoders or "age_std" not in encoders:
        warnings.warn("[CLIN-RFS] encoders missing age_mean/age_std — using "
                       "defaults (60.0/10.0), will NOT match 0.6821 result.")
    age_norm = (get_ehr("age", 60) - age_mean) / (age_std + 1e-8)

    # ehr.json does not include CenterID at test time in the validated
    # pipeline; default to center 5 (most common training center).
    center_id  = int(get_ehr("center_id", 5))
    center_vec = [1.0 if i == center_id else 0.0 for i in range(1, 9)]

    def norm_rad(stat_key, val):
        p99 = stats.get(stat_key, 1.0)
        return float(val) / (float(p99) + 1e-8)

    rad_vec = [
        norm_rad("vol_p99",          radiomics["GTVp_volume_cm3"]),
        norm_rad("voln_p99",         radiomics["GTVn_volume_cm3"]),
        norm_rad("dp_p99",           radiomics["GTVp_max_diam_mm"]),
        norm_rad("dn_p99",           radiomics["GTVn_max_diam_mm"]),
        norm_rad("nc_p99",           radiomics["GTVn_node_count"]),
        norm_rad("GTVp_SUVmax_p99",  radiomics["GTVp_SUVmax"]),
        norm_rad("GTVn_SUVmax_p99",  radiomics["GTVn_SUVmax"]),
        norm_rad("GTVp_SUVmean_p99", radiomics["GTVp_SUVmean"]),
        norm_rad("GTVn_SUVmean_p99", radiomics["GTVn_SUVmean"]),
        norm_rad("GTVp_TLG_p99",     radiomics["GTVp_TLG"]),
        norm_rad("GTVn_TLG_p99",     radiomics["GTVn_TLG"]),
    ]

    vec = [
        age_norm,
        get_ehr("gender", -1),
        get_ehr("tobacco", -1),
        get_ehr("alcohol", -1),
        get_ehr("hpv", -1),
        get_ehr("treatment", -1),
        get_ehr("performance_status", -1),
    ] + center_vec + rad_vec

    assert len(vec) == CLINICAL_DIM_RFS, (
        f"[CLIN-RFS] Built {len(vec)}-dim vector but expected {CLINICAL_DIM_RFS}.")
    print(f"[CLIN-RFS] vec({len(vec)})={[round(v,3) for v in vec]}")
    return torch.tensor([vec], dtype=torch.float32)


# =============================================================================
# Model definitions
# =============================================================================

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.InstanceNorm3d(out_ch, affine=True)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.leaky_relu(
            self.bn2(self.conv2(F.leaky_relu(self.bn1(self.conv1(x))))) +
            self.downsample(x), inplace=True)


class DualBranchFusionResNet(nn.Module):
    """TN staging model — task2_v7. clinical_dim comes from the checkpoint
    (should be 30)."""
    def __init__(self, clinical_dim=CLINICAL_DIM_TN, t_classes=4, n_classes=4):
        super().__init__()
        self.img_stem  = nn.Sequential(nn.Conv3d(2,32,3,padding=1,bias=False),
            nn.InstanceNorm3d(32,affine=True),nn.LeakyReLU(inplace=True),nn.MaxPool3d(2))
        self.mask_conv = nn.Sequential(nn.Conv3d(3,32,3,padding=1,bias=False),
            nn.InstanceNorm3d(32,affine=True),nn.LeakyReLU(inplace=True),nn.MaxPool3d(2))
        self.layer1 = nn.Sequential(ResBlock3D(32,32),ResBlock3D(32,32))
        self.layer2 = nn.Sequential(ResBlock3D(32,64,stride=2),ResBlock3D(64,64))
        self.layer3 = nn.Sequential(ResBlock3D(64,128,stride=2),ResBlock3D(128,128))
        self.layer4 = nn.Sequential(ResBlock3D(128,256,stride=2),ResBlock3D(256,256))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim,64),nn.BatchNorm1d(64),nn.ReLU(inplace=True),
            nn.Linear(64,32),nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(288,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),
            nn.Dropout(0.3),nn.Linear(256,128),nn.ReLU(inplace=True))
        self.t_head = nn.Linear(128,t_classes)
        self.n_head = nn.Linear(128,n_classes)

    def forward(self,img,mask_onehot,clinical):
        x=self.img_stem(img)+self.mask_conv(mask_onehot)
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x)
        img_feat=self.pool(x).flatten(1)
        clin_feat=self.clinical_mlp(clinical)
        fused=self.fusion(torch.cat([img_feat,clin_feat],dim=1))
        return self.t_head(fused),self.n_head(fused)


class RFSModel(nn.Module):
    """Prognosis model — original single-head version. clinical_dim comes
    from the checkpoint (should be 26)."""
    def __init__(self, clinical_dim=CLINICAL_DIM_RFS):
        super().__init__()
        self.img_stem  = nn.Sequential(nn.Conv3d(2, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True), nn.LeakyReLU(inplace=True), nn.MaxPool3d(2))
        self.mask_conv = nn.Sequential(nn.Conv3d(3, 32, 3, padding=1, bias=False),
            nn.InstanceNorm3d(32, affine=True), nn.LeakyReLU(inplace=True), nn.MaxPool3d(2))
        self.layer1 = nn.Sequential(ResBlock3D(32,32), ResBlock3D(32,32))
        self.layer2 = nn.Sequential(ResBlock3D(32,64,stride=2), ResBlock3D(64,64))
        self.layer3 = nn.Sequential(ResBlock3D(64,128,stride=2), ResBlock3D(128,128))
        self.layer4 = nn.Sequential(ResBlock3D(128,256,stride=2), ResBlock3D(256,256))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim,64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64,32), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(288,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256,128), nn.ReLU(inplace=True),
            nn.Linear(128,1))

    def forward(self, img, mask_onehot, clinical):
        x = self.img_stem(img) + self.mask_conv(mask_onehot)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        img_feat  = self.pool(x).flatten(1)
        clin_feat = self.clinical_mlp(clinical)
        return self.fusion(torch.cat([img_feat, clin_feat], dim=1)).squeeze(1)


# =============================================================================
# Subtask 1 — Segmentation
# =============================================================================

def run_segmentation(ct_crop, pet_crop):
    print(f"[SEG] Starting at {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    os.environ["nnUNet_results"]      = str(MODEL_PATH / "nnunet")
    os.environ["nnUNet_raw"]          = "/tmp/nnunet_raw"
    os.environ["nnUNet_preprocessed"] = "/tmp/nnunet_preprocessed"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    tmp_in  = Path("/tmp/nnunet_input")
    tmp_out = Path("/tmp/nnunet_output_dir")
    tmp_in.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(ct_crop,  str(tmp_in / "case_0000.nii.gz"))
    sitk.WriteImage(pet_crop, str(tmp_in / "case_0001.nii.gz"))

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=DEVICE,
            verbose=False,
            verbose_preprocessing=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(MODEL_PATH / "nnunet" /
                f"Dataset{NNUNET_DATASET_ID:03d}_HECKTOR2026" /
                f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIG}"),
            use_folds=NNUNET_FOLDS,
            checkpoint_name="checkpoint_best.pth",
        )
        tmp_mask = tmp_out / "case.nii.gz"
        predictor.predict_from_files(
            [[str(tmp_in / "case_0000.nii.gz"), str(tmp_in / "case_0001.nii.gz")]],
            [str(tmp_mask)],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
    except Exception as e:
        import traceback
        print(f"[SEG] ERROR: {e}")
        print(traceback.format_exc())
        return np.zeros(sitk.GetArrayFromImage(ct_crop).shape, dtype=np.uint8)

    mask_files = [f for f in tmp_out.glob("*.nii.gz") if "probabilities" not in f.name]
    if not mask_files:
        print("[SEG] No mask output found")
        return np.zeros(sitk.GetArrayFromImage(ct_crop).shape, dtype=np.uint8)

    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_files[0]))).astype(np.uint8)

    result_arr = np.zeros_like(mask_arr)
    gtvp = (mask_arr == 1)
    if gtvp.sum() >= 10:
        labeled, n = ndimage.label(gtvp, structure=FULL_CONNECTIVITY_3D)
        if n > 0:
            sizes = ndimage.sum(gtvp, labeled, range(1, n+1))
            result_arr[labeled == np.argmax(sizes)+1] = 1
    gtvn = (mask_arr == 2)
    if gtvn.sum() >= 10:
        labeled, n = ndimage.label(gtvn, structure=FULL_CONNECTIVITY_3D)
        for i in range(1, n+1):
            if (labeled == i).sum() >= 10:
                result_arr[labeled == i] = 2

    print(f"[SEG] Done in {time.time()-t0:.0f}s | GTVp={(result_arr==1).sum()} GTVn={(result_arr==2).sum()}")
    return result_arr


def map_crop_mask_to_ct_space(mask_crop_arr_zyx, ct_rs, ct_sitk, box_start):
    x0, y0, z0 = [int(v) for v in box_start]
    z1 = z0 + mask_crop_arr_zyx.shape[0]
    y1 = y0 + mask_crop_arr_zyx.shape[1]
    x1 = x0 + mask_crop_arr_zyx.shape[2]

    full_shape = sitk.GetArrayFromImage(ct_rs).shape
    full_pred  = np.zeros(full_shape, dtype=np.uint8)
    full_pred[z0:z1, y0:y1, x0:x1] = mask_crop_arr_zyx

    full_img = sitk.GetImageFromArray(full_pred)
    full_img.CopyInformation(ct_rs)

    final = sitk.Resample(full_img, ct_sitk, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(final)


# =============================================================================
# Subtask 2 — TN Staging (task2_v7 checkpoints, 30-dim clinical vector)
# =============================================================================

def run_tn_staging(ct_arr_xyz, pet_arr_xyz, mask_arr_xyz, ehr, encoders):
    """
    TTA (test-time augmentation) via flips: matches the random axis-0/1/2
    flip augmentation used during task2_v7 training
    (HECKTORStagingDataset.__getitem__), so the model is being asked for
    predictions on variants it was explicitly trained to be invariant to.

    Averages softmax probabilities across the identity view plus all 3
    single-axis flips (4 views total) BEFORE the existing 5-fold ensemble
    average. Clinical vector and radiomics are computed ONCE from the
    un-flipped mask (flipping ct/pet/mask together keeps them spatially
    consistent for the img_stem/mask_conv branches; radiomics/clinical
    features are orientation-invariant scalars so recomputing per flip
    would be redundant, not wrong).

    Set TTA_FLIPS=False below to instantly revert to the original
    single-view behavior if this needs to be ruled out.
    """
    print(f"[TN] Starting at {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    TTA_FLIPS = True  # flip to False to disable TTA with one edit

    radiomics   = extract_radiomics_tn(mask_arr_xyz.astype(np.uint8), pet_arr_xyz)
    clin        = build_clinical_vector_tn(ehr, radiomics, encoders).to(DEVICE)
    img, mask_onehot = prepare_tensors(ct_arr_xyz, pet_arr_xyz, mask_arr_xyz)
    img         = img.to(DEVICE)
    mask_onehot = mask_onehot.to(DEVICE)

    # img shape: (1, 2, D, H, W) ; mask_onehot shape: (1, 3, D, H, W)
    # Spatial dims are axes 2,3,4 -- correspond to the 0/1/2 axis flips
    # used in training (which flipped the pre-batch (D,H,W) array).
    if TTA_FLIPS:
        spatial_axes = [2, 3, 4]
        views = [(img, mask_onehot)]  # identity view first
        for ax in spatial_axes:
            views.append((torch.flip(img, dims=[ax]), torch.flip(mask_onehot, dims=[ax])))
    else:
        views = [(img, mask_onehot)]

    task2_dir   = MODEL_PATH / "task2"
    t_probs_all = []
    n_probs_all = []

    for fold in range(5):
        ckpt_path = task2_dir / f"fold{fold}_best.pth"
        if not ckpt_path.exists():
            print(f"[TN] fold{fold} missing")
            continue
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = DualBranchFusionResNet(clinical_dim=ckpt.get("clinical_dim", CLINICAL_DIM_TN)).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        fold_t_probs = []
        fold_n_probs = []
        with torch.no_grad():
            for view_img, view_mask in views:
                t_logits, n_logits = model(view_img, view_mask, clin)
                fold_t_probs.append(torch.softmax(t_logits, dim=1).cpu().numpy())
                fold_n_probs.append(torch.softmax(n_logits, dim=1).cpu().numpy())
        # Average across TTA views for this fold, THEN treat like before
        t_probs_all.append(np.mean(fold_t_probs, axis=0))
        n_probs_all.append(np.mean(fold_n_probs, axis=0))
        del model
        torch.cuda.empty_cache()

    if not t_probs_all:
        print("[TN] WARNING: no Task 2 checkpoints found, returning default stage")
        return "T2", "N2"

    t_stage = T_NAMES[int(np.argmax(np.mean(t_probs_all, axis=0), axis=1)[0])]
    n_stage = N_NAMES[int(np.argmax(np.mean(n_probs_all, axis=0), axis=1)[0])]
    print(f"[TN] Done in {time.time()-t0:.0f}s | T={t_stage} N={n_stage} | TTA_FLIPS={TTA_FLIPS}")
    return t_stage, n_stage


# =============================================================================
# Subtask 3 — Prognosis (original RFSModel checkpoints, 26-dim clinical vector)
# =============================================================================

def run_prognosis(ct_arr_xyz, pet_arr_xyz, mask_arr_xyz, ehr, encoders, t_stage, n_stage):
    """
    Two optional, no-retrain post-processing steps, both gated by flags
    below so either can be instantly disabled if it doesn't pan out:

    TTA_FLIPS: averages risk across the identity view + 3 single-axis
    flips (matches training augmentation in the original task3.py,
    which flipped img/mask_onehot on axes [1,2,3] pre-batch -- same
    spatial axes as TN's TTA, dims=[2,3,4] here since img is batched).

    WEIGHTED_ENSEMBLE: weights each fold's risk contribution by that
    fold's own validation C-index (stored in the checkpoint as
    ckpt["cindex"] by the original training script), instead of a
    plain unweighted mean across folds. Folds are normalized so
    weights sum to 1. Falls back to equal weighting for any fold
    missing a "cindex" key.

    IMPORTANT: unlike TN staging (balanced accuracy), this task is
    scored by C-index, which only depends on the RELATIVE ORDER of
    risk scores across patients -- so any purely monotonic
    rescaling would NOT change the score. Both changes below CAN
    change relative ordering (TTA by changing each patient's estimate
    based on new information; weighting by shifting which folds'
    opinions dominate) -- that's why they're worth trying, unlike
    e.g. calibration/scaling.
    """
    print(f"[RFS] Starting at {time.strftime('%H:%M:%S')}")
    t0 = time.time()

    TTA_FLIPS         = True  # flip to False to disable
    WEIGHTED_ENSEMBLE  = True  # flip to False to disable (reverts to plain mean)

    radiomics   = extract_radiomics_rfs(mask_arr_xyz.astype(np.uint8), pet_arr_xyz)
    clin        = build_clinical_vector_rfs(ehr, radiomics, encoders).to(DEVICE)
    img, mask_onehot = prepare_tensors(ct_arr_xyz, pet_arr_xyz, mask_arr_xyz)
    img         = img.to(DEVICE)
    mask_onehot = mask_onehot.to(DEVICE)

    if TTA_FLIPS:
        spatial_axes = [2, 3, 4]
        views = [(img, mask_onehot)]
        for ax in spatial_axes:
            views.append((torch.flip(img, dims=[ax]), torch.flip(mask_onehot, dims=[ax])))
    else:
        views = [(img, mask_onehot)]

    task3_dir     = MODEL_PATH / "task3"
    risk_scores   = []
    fold_weights  = []

    for fold in range(5):
        ckpt_path = task3_dir / f"fold{fold}_best.pth"
        if not ckpt_path.exists():
            print(f"[RFS] fold{fold} missing")
            continue
        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = RFSModel(clinical_dim=ckpt.get("clinical_dim", CLINICAL_DIM_RFS)).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        view_risks = []
        with torch.no_grad():
            for view_img, view_mask in views:
                view_risks.append(model(view_img, view_mask, clin).cpu().item())
        fold_risk = float(np.mean(view_risks))
        risk_scores.append(fold_risk)

        fold_cindex = ckpt.get("cindex", None)
        fold_weights.append(fold_cindex if fold_cindex is not None else np.nan)

        print(f"[RFS] fold{fold}: risk={fold_risk:.4f} "
              f"(views={len(view_risks)}) stored_cindex={fold_cindex}")
        del model
        torch.cuda.empty_cache()

    if not risk_scores:
        print("[RFS] WARNING: no Task 3 checkpoints found, returning default risk")
        return 0.0

    risk_scores  = np.array(risk_scores)
    fold_weights = np.array(fold_weights, dtype=np.float64)

    if WEIGHTED_ENSEMBLE and not np.all(np.isnan(fold_weights)):
        # Missing/NaN weights fall back to the mean of the known weights,
        # so a fold without a stored cindex doesn't get zeroed out.
        if np.any(np.isnan(fold_weights)):
            fill = np.nanmean(fold_weights)
            fold_weights = np.where(np.isnan(fold_weights), fill, fold_weights)
        fold_weights = fold_weights / fold_weights.sum()
        raw_rfs_score = float(np.sum(risk_scores * fold_weights))
        print(f"[RFS] Weighted ensemble weights: {np.round(fold_weights, 3).tolist()}")
    else:
        raw_rfs_score = float(np.mean(risk_scores))

    flipped_rfs_score = -raw_rfs_score   # validated: negation is what scored 0.6821/0.6964

    print(f"[RFS] Done in {time.time()-t0:.0f}s | "
          f"raw={raw_rfs_score:.4f} flipped(used)={flipped_rfs_score:.4f} | "
          f"TTA_FLIPS={TTA_FLIPS} WEIGHTED_ENSEMBLE={WEIGHTED_ENSEMBLE}")
    return flipped_rfs_score


# =============================================================================
# Main
# =============================================================================

def run():
    total_start = time.time()
    print(f"[HECKTOR2026] Starting at {time.strftime('%H:%M:%S')} | Device: {DEVICE}")

    ct_path  = get_image_file(INPUT_PATH / "images/ct")
    pet_path = get_image_file(INPUT_PATH / "images/pet")
    ehr      = load_json(INPUT_PATH / "ehr.json")
    print(f"[HECKTOR2026] EHR: {ehr}")

    with open(MODEL_PATH / "clinical_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    print(f"[HECKTOR2026] encoders keys: {list(encoders.keys())}")

    print(f"[PREP] Starting at {time.strftime('%H:%M:%S')}")
    t0   = time.time()
    prep = preprocess_once(ct_path, pet_path)
    print(f"[PREP] Done in {time.time()-t0:.0f}s")

    ct_arr_zyx  = sitk.GetArrayFromImage(prep["ct_crop"]).astype(np.float32)
    pet_arr_zyx = sitk.GetArrayFromImage(prep["pet_crop"]).astype(np.float32)
    ct_arr_xyz  = np.transpose(ct_arr_zyx,  (2, 1, 0))
    pet_arr_xyz = np.transpose(pet_arr_zyx, (2, 1, 0))

    mask_zyx = run_segmentation(prep["ct_crop"], prep["pet_crop"])
    final_mask = map_crop_mask_to_ct_space(
        mask_zyx, prep["ct_rs"], prep["ct_sitk"], prep["box_start"])

    ct_filename = Path(ct_path).name
    for suffix in [".nii.gz", ".mha", ".tif", ".tiff"]:
        ct_filename = ct_filename.replace(suffix, "")
    patient_id = ct_filename.replace("_0000", "").replace("_0001", "")
    print(f"[HECKTOR2026] Patient ID: {patient_id}")

    write_segmentation(
        location=OUTPUT_PATH / "images/head-neck-tumor-segmentation",
        array=final_mask, reference_path=ct_path,
        patient_id=patient_id)

    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.float32)

    t_stage, n_stage = run_tn_staging(ct_arr_xyz, pet_arr_xyz, mask_xyz, ehr, encoders)
    write_json(OUTPUT_PATH / "t-stage.json", t_stage)
    write_json(OUTPUT_PATH / "n-stage.json", n_stage)

    rfs_score = run_prognosis(ct_arr_xyz, pet_arr_xyz, mask_xyz, ehr, encoders, t_stage, n_stage)
    write_json(OUTPUT_PATH / "rfs.json", float(rfs_score))

    print(f"[HECKTOR2026] Total: {time.time()-total_start:.0f}s")
    return 0


# =============================================================================
# I/O utilities
# =============================================================================

def get_image_file(location):
    files = (glob(str(location / "*.mha")) + glob(str(location / "*.tif"))
             + glob(str(location / "*.tiff")) + glob(str(location / "*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No image file found in {location}")
    return files[0]


def load_json(location):
    with open(location) as f:
        return json.load(f)


def write_json(location, data):
    Path(location).parent.mkdir(parents=True, exist_ok=True)
    with open(location, "w") as f:
        json.dump(data, f, indent=2)


def write_segmentation(location, array, reference_path, patient_id="output"):
    Path(location).mkdir(parents=True, exist_ok=True)
    reference = sitk.ReadImage(str(reference_path))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    try:
        if img.GetDimension() == reference.GetDimension():
            img.CopyInformation(reference)
        else:
            img.SetSpacing([1.0] * img.GetDimension())
            img.SetOrigin([0.0] * img.GetDimension())
    except Exception as e:
        print(f"[SEG] CopyInformation skipped: {e}")
    sitk.WriteImage(img, str(Path(location) / f"{patient_id}.mha"), useCompression=True)


if __name__ == "__main__":
    raise SystemExit(run())
