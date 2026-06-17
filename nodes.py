import sys
import os
import torch
import comfy
from comfy_api.latest import io

# =====================================================================
# ULTRA-ROBUST SIBLING MODULE RESOLVER (Handles custom folder naming & AVSplit)
# =====================================================================
def _get_ltx_module(module_name):
    import sys
    import importlib
    from pathlib import Path

    # Find custom_nodes directory from our own folder path
    current_file_path = Path(__file__).resolve()
    custom_nodes_dir = current_file_path.parent.parent

    # Find the LTXVideo folder with high priority exact matches
    ltx_video_dir = None
    
    # Priority 1: Exact match (preferred)
    for p in custom_nodes_dir.iterdir():
        if p.is_dir() and p.name in ("ComfyUI-LTXVideo", "ComfyUI_LTXVideo"):
            ltx_video_dir = p
            break
            
    # Priority 2: Substring matches (ignoring alternative modules like AVSplit)
    if ltx_video_dir is None:
        for p in custom_nodes_dir.iterdir():
            if p.is_dir() and ("ComfyUI-LTXVideo" in p.name or "ComfyUI_LTXVideo" in p.name):
                if "AVSplit" not in p.name:
                    ltx_video_dir = p
                    break

    # Priority 3: Fallback search in sys.path
    if ltx_video_dir is None:
        for p_str in sys.path:
            p = Path(p_str)
            if "ComfyUI-LTXVideo" in p.name or "ComfyUI_LTXVideo" in p.name:
                if "AVSplit" not in p.name:
                    ltx_video_dir = p
                    break

    if ltx_video_dir is None:
        raise ImportError(
            f"Could not find the 'ComfyUI-LTXVideo' custom nodes directory. "
            "Please ensure it is installed."
        )

    ltx_dir_str = str(ltx_video_dir)
    pkg_name = ltx_video_dir.name

    # If the parent directory (custom_nodes) is not in sys.path, add it briefly
    custom_nodes_str = str(custom_nodes_dir)
    added_to_sys_path = False
    if custom_nodes_str not in sys.path:
        sys.path.insert(0, custom_nodes_str)
        added_to_sys_path = True

    try:
        # Import it as a sub-module of the package
        full_module_name = f"{pkg_name}.{module_name}"
        if full_module_name in sys.modules:
            return sys.modules[full_module_name]

        return importlib.import_module(full_module_name)
    finally:
        if added_to_sys_path:
            sys.path.remove(custom_nodes_str)

# =====================================================================
# STANDALONE HELPER FUNCTIONS (No dependency on ComfyUI-LTXVideo)
# =====================================================================
def normalize_mask(mask):
    """Normalize a ComfyUI MASK to (1, 1, F, H, W) for downstream processing.
    ComfyUI MASK type is typically (B, H, W) or (H, W).
    """
    if mask is None:
        return None
    if mask.dim() == 2:  # (H, W) -> single frame
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:  # (F, H, W) -> video mask
        return mask.unsqueeze(0).unsqueeze(0)
    return mask


def _get_guide_attention_entries(conditioning):
    """Read the current guide_attention_entries list from conditioning."""
    for t in conditioning:
        entries = t[1].get("guide_attention_entries", None)
        if entries is not None:
            return entries
    return []


def _set_guide_attention_entries(conditioning, entries):
    """Write guide_attention_entries into conditioning (immutable update)."""
    import node_helpers
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": entries}
    )


def append_guide_attention_entry(
    conditioning,
    pre_filter_count,
    latent_shape,
    attention_strength=1.0,
    attention_mask=None,
):
    """Append a new guide attention entry to conditioning metadata."""
    existing_entries = _get_guide_attention_entries(conditioning)
    entries = [*existing_entries]
    entries.append(
        {
            "pre_filter_count": pre_filter_count,
            "strength": attention_strength,
            "pixel_mask": attention_mask,
            "latent_shape": latent_shape,
        }
    )
    return _set_guide_attention_entries(conditioning, entries)


def _dilate_latent(samples, scale_factor):
    """Dilates a latent by a scale factor and returns dilated samples and noise_mask."""
    if scale_factor == 1.0:
        return samples, None

    scale_factor = int(scale_factor)
    dilated_shape = samples.shape[:3] + (
        samples.shape[3] * scale_factor,
        samples.shape[4] * scale_factor,
    )

    dilated_samples = torch.zeros(
        dilated_shape,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_samples[..., ::scale_factor, ::scale_factor] = samples

    dilated_mask_shape = (
        dilated_samples.shape[0],
        1,
        dilated_samples.shape[2],
        dilated_samples.shape[3],
        dilated_samples.shape[4],
    )
    dilated_mask = torch.full(
        dilated_mask_shape,
        -1.0,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_mask[..., ::scale_factor, ::scale_factor] = 1.0

    return dilated_samples, dilated_mask


def _encode_guide_latent(
    vae,
    latent_width,
    latent_height,
    images,
    scale_factors,
    latent_downscale_factor,
    crop,
    use_tiled_encode,
    tile_size,
    tile_overlap,
):
    """Encodes the guide images into latent space using VAE (with optional tiled encoding and downscaling)."""
    time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
    num_frames_to_keep = (
        (images.shape[0] - 1) // time_scale_factor
    ) * time_scale_factor + 1
    images = images[:num_frames_to_keep]

    target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
    target_height = int(latent_height * height_scale_factor / latent_downscale_factor)

    pixels = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        target_width,
        target_height,
        "bilinear",
        crop=crop,
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    if use_tiled_encode:
        guide_latent = vae.encode_tiled(
            encode_pixels,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=tile_overlap,
        )
    else:
        guide_latent = vae.encode(encode_pixels)
    return encode_pixels, guide_latent


# =====================================================================
# STANDALONE NODE IMPLEMENTATION WITH USER-FRIENDLY TOOLTIPS
# =====================================================================
class LTXMSRICLoRAFLF(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXMSRICLoRAFLF",
            display_name="🅛🅣🅧 MSR IC LORA FLF (Standalone)",
            category="Lightricks/IC-LoRA",
            description="Standalone custom node combining Licon MSR sequence creation and IC-LoRA conditioning for First, Last, and Subject frames.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="The positive text conditioning. The reference frames and IC-LoRA attention metadata will be appended to this."),
                io.Conditioning.Input("negative", tooltip="The negative text conditioning. The reference frames and IC-LoRA attention metadata will be appended to this."),
                io.Vae.Input("vae", tooltip="The VAE model used to encode your reference images (First Frame, Last Frame, and MSR images) into latent space."),
                io.Latent.Input("latent", tooltip="The main video latent container from your sampler/empty latent node. This defines the target video length, width, and height."),
                
                # MSR Inputs & Settings
                io.Int.Input("msr_width", default=1920, min=32, max=8192, step=32, tooltip="Width to resize all MSR (Multi-Subject Reference) images to before encoding them. Usually set to match your target video width."),
                io.Int.Input("msr_height", default=1088, min=32, max=8192, step=32, tooltip="Height to resize all MSR (Multi-Subject Reference) images to before encoding them. Usually set to match your target video height."),
                io.Combo.Input("msr_frame_count", options=["17", "25", "33", "41"], default="41", tooltip="The total number of frames in the generated MSR sequence. The input images will be duplicated evenly to fill this length."),
                io.Image.Input("msr_image_1", optional=True, tooltip="The first reference image of your subject. Will be placed at the start of the MSR sequence."),
                io.Image.Input("msr_image_2", optional=True, tooltip="The second reference image of your subject (optional). Allows conditioning on different angles or states."),
                io.Image.Input("msr_image_3", optional=True, tooltip="The third reference image of your subject (optional)."),
                io.Image.Input("msr_image_4", optional=True, tooltip="The fourth reference image of your subject (optional)."),
                io.Image.Input("msr_background", optional=True, tooltip="Optional background reference image. If provided, it is placed at the end of the MSR sequence to guide the background style."),
                io.Float.Input("msr_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="The injection strength of MSR latents directly into the latent space (0.0 = none, 1.0 = maximum). Controls how strongly the subject features are forced into the latent pixels."),
                io.Float.Input("msr_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="How strongly the generator's cross-attention is guided by the MSR sequence. Higher values make the model pay more attention to the subject references."),
                
                # First Frame Inputs & Settings
                io.Image.Input("first_frame", optional=True, tooltip="The starting image of your video. Used to establish the exact beginning of the video sequence."),
                io.Boolean.Input("lock_first_frame", default=False, tooltip="If enabled, freezes the first frame's latents to force the video to start exactly with your first_frame image, preventing any modification by the sampler."),
                io.Int.Input("lock_first_frame_len", default=1, min=1, max=16, step=1, tooltip="The number of latent frames to freeze (1 latent frame = 8 pixel frames in LTXV). Set to 2 or more to prevent the second frame from being blurry due to VAE block interpolation."),
                io.Float.Input("first_frame_strength", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="The direct latent injection strength of the first frame (0.0 = none, 1.0 = maximum). Controls how closely the starting frames resemble the input image."),
                io.Float.Input("first_frame_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="How strongly the generator pays attention to the first frame image during the entire video generation via cross-attention."),
                
                # Last Frame Inputs & Settings
                io.Image.Input("last_frame", optional=True, tooltip="The ending image of your video. Used to establish a specific end point for the motion sequence."),
                io.Boolean.Input("lock_last_frame", default=False, tooltip="If enabled, freezes the last frame's latents to force the video to end exactly on your last_frame image."),
                io.Float.Input("last_frame_strength", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="The direct latent injection strength of the last frame (0.0 = none, 1.0 = maximum). Controls how closely the final frames resemble the input image."),
                io.Float.Input("last_frame_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="How strongly the generator pays attention to the last frame image during the entire video generation via cross-attention."),
                
                # IC-LoRA General Settings
                io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=10.0, step=1.0, tooltip="The downscaling factor applied to the reference latents before injection. Set to 1.0 for normal, or 2.0 to match downscaled/compressed IC-LoRA resolutions."),
                io.Combo.Input("crop", options=["disabled", "center"], default="center", tooltip="How to resize images to the target resolution. 'disabled' stretches/squishes the image directly; 'center' crops to maintain the original aspect ratio."),
                io.Boolean.Input("use_tiled_encode", default=False, tooltip="Enables tiled VAE encoding. Strongly recommended for high resolutions to prevent running out of GPU memory (OOM)."),
                io.Int.Input("tile_size", default=256, min=64, max=512, step=32, tooltip="Size of individual pixel tiles used during tiled encoding. Smaller sizes use less VRAM but take slightly longer to process."),
                io.Int.Input("tile_overlap", default=64, min=16, max=256, step=16, tooltip="Overlap between adjacent tiles in pixels. Keeps transitions smooth and prevents visible seams or boundary lines in encoded images."),
                
                # Masks
                io.Mask.Input("msr_mask", optional=True, tooltip="An optional black and white mask to limit MSR IC-LoRA attention influence to specific spatial areas (e.g. only focusing on the subject)."),
                io.Mask.Input("first_frame_mask", optional=True, tooltip="An optional black and white mask to limit First Frame IC-LoRA attention influence to specific spatial areas."),
                io.Mask.Input("last_frame_mask", optional=True, tooltip="An optional black and white mask to limit Last Frame IC-LoRA attention influence to specific spatial areas."),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        msr_width,
        msr_height,
        msr_frame_count,
        msr_image_1=None,
        msr_image_2=None,
        msr_image_3=None,
        msr_image_4=None,
        msr_background=None,
        msr_strength=1.00,
        msr_attention_strength=1.00,
        first_frame=None,
        lock_first_frame=False,
        lock_first_frame_len=1,
        first_frame_strength=0.85,
        first_frame_attention_strength=1.00,
        last_frame=None,
        lock_last_frame=False,
        last_frame_strength=0.85,
        last_frame_attention_strength=1.00,
        latent_downscale_factor=1.0,
        crop="center",
        use_tiled_encode=False,
        tile_size=256,
        tile_overlap=64,
        msr_mask=None,
        first_frame_mask=None,
        last_frame_mask=None,
    ) -> io.NodeOutput:
        
        # Deferred import of LTX-Video helpers (avoids alphabet loading issues)
        iclora_attention = _get_ltx_module("iclora_attention")
        latents_mod = _get_ltx_module("latents")
        iclora_mod = _get_ltx_module("iclora")
        import comfy_extras.nodes_lt as nodes_lt

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        
        # Extract or initialize noise mask
        noise_mask = nodes_lt.get_noise_mask(latent)
        if noise_mask is None:
            b, _, l_len, l_h, l_w = latent_image.shape
            noise_mask = torch.ones((b, 1, l_len, l_h, l_w), dtype=torch.float32, device=latent_image.device)
        else:
            noise_mask = noise_mask.clone()

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = scale_factors[0]
        
        norm_msr_mask = iclora_attention.normalize_mask(msr_mask) if msr_mask is not None else None
        norm_ff_mask = iclora_attention.normalize_mask(first_frame_mask) if first_frame_mask is not None else None
        norm_lf_mask = iclora_attention.normalize_mask(last_frame_mask) if last_frame_mask is not None else None

        # --- Variables to store guide latents for inplace locking in Step 5 ---
        ff_guide_latent_ref = None
        lf_guide_latent_ref = None
        lf_latent_idx_ref = None
        lf_guide_len_ref = None

        # --- STEP 1: APPLY FIRST FRAME (FL) ---
        if first_frame is not None:
            causal_fix = True # Always True for index 0
            
            ff_image = first_frame

            # Use base encode from standard iclora
            ff_image, ff_guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                vae=vae, latent_width=latent_width, latent_height=latent_height, images=ff_image,
                scale_factors=scale_factors, latent_downscale_factor=latent_downscale_factor,
                crop=crop, use_tiled_encode=use_tiled_encode, tile_size=tile_size, tile_overlap=tile_overlap
            )

            ff_guide_orig_shape = list(ff_guide_latent.shape[2:])
            ff_guide_mask = None

            if latent_downscale_factor > 1:
                dilated = latents_mod.LTXVDilateLatent().dilate_latent(
                    {"samples": ff_guide_latent},
                    horizontal_scale=int(latent_downscale_factor),
                    vertical_scale=int(latent_downscale_factor),
                )[0]
                ff_guide_mask = dilated["noise_mask"]
                ff_guide_latent = dilated["samples"]

            ff_tokens_added = ff_guide_latent.shape[2] * ff_guide_latent.shape[3] * ff_guide_latent.shape[4]
            
            # Store reference for lock
            ff_guide_latent_ref = ff_guide_latent

            # Register standard keyframe
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, 0, latent_image, noise_mask, ff_guide_latent,
                first_frame_strength, scale_factors, guide_mask=ff_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
            )

            # Register cross-attention conditionings
            positive = iclora_attention.append_guide_attention_entry(
                positive, ff_tokens_added, ff_guide_orig_shape,
                attention_strength=first_frame_attention_strength, attention_mask=norm_ff_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, ff_tokens_added, ff_guide_orig_shape,
                attention_strength=first_frame_attention_strength, attention_mask=norm_ff_mask
            )

        # --- PREPARE MSR IMAGES ---
        msr_images = []
        for img in [msr_image_1, msr_image_2, msr_image_3, msr_image_4]:
            if img is not None:
                msr_images.append(img)
        if msr_background is not None:
            msr_images.append(msr_background)

        # --- STEP 2: MULTI-GUIDE LATENT INJECTION (MSR CHAIN 1) ---
        # Inject individual MSR images into latent at frame_idx=0 (similar to LTXVAddGuideMulti)
        if len(msr_images) > 0:
            for i, img in enumerate(msr_images):
                img_encoded, guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                    vae=vae, latent_width=latent_width, latent_height=latent_height, images=img,
                    scale_factors=scale_factors, latent_downscale_factor=latent_downscale_factor,
                    crop=crop, use_tiled_encode=use_tiled_encode, tile_size=tile_size, tile_overlap=tile_overlap
                )
                
                guide_mask = None
                if latent_downscale_factor > 1:
                    dilated = latents_mod.LTXVDilateLatent().dilate_latent(
                        {"samples": guide_latent},
                        horizontal_scale=int(latent_downscale_factor),
                        vertical_scale=int(latent_downscale_factor),
                    )[0]
                    guide_mask = dilated["noise_mask"]
                    guide_latent = dilated["samples"]

                # We call append_keyframe but DISCARD the returned positive/negative 
                # because we don't want these single frames adding attention metadata.
                # Only the video in Step 3 should add attention metadata.
                _, _, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                    positive.copy(), negative.copy(), 0, latent_image, noise_mask, guide_latent,
                    msr_strength, scale_factors, guide_mask=guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=True
                )

        # --- STEP 3: MSR VIDEO CONDITIONING (MSR CHAIN 2) ---
        # Package MSR images into a video sequence and add attention entries to conditioning
        if len(msr_images) > 0:
            fc = int(msr_frame_count)
            # 3.1 Resize using comfy.utils.common_upscale
            resized_images = []
            for img in msr_images:
                resized = comfy.utils.common_upscale(
                    img.movedim(-1, 1), msr_width, msr_height, "bilinear", crop=crop
                ).movedim(1, -1)
                resized_images.append(resized)

            # 3.2 Expand to MSR sequence
            base_count = fc // len(resized_images)
            remainder = fc % len(resized_images)
            msr_sequence_frames = []
            for index, img_tensor in enumerate(resized_images):
                repeats = base_count + (1 if index < remainder else 0)
                for _ in range(repeats):
                    msr_sequence_frames.append(img_tensor)
            
            msr_video = torch.cat(msr_sequence_frames, dim=0) # [fc, H, W, 3]

            num_frames_to_keep = ((msr_video.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
            msr_video = msr_video[:num_frames_to_keep]

            msr_image_out, msr_guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                vae=vae, latent_width=latent_width, latent_height=latent_height, images=msr_video,
                scale_factors=scale_factors, latent_downscale_factor=latent_downscale_factor,
                crop=crop, use_tiled_encode=use_tiled_encode, tile_size=tile_size, tile_overlap=tile_overlap
            )

            msr_guide_orig_shape = list(msr_guide_latent.shape[2:])
            msr_guide_mask = None

            if latent_downscale_factor > 1:
                dilated = latents_mod.LTXVDilateLatent().dilate_latent(
                    {"samples": msr_guide_latent},
                    horizontal_scale=int(latent_downscale_factor),
                    vertical_scale=int(latent_downscale_factor),
                )[0]
                msr_guide_mask = dilated["noise_mask"]
                msr_guide_latent = dilated["samples"]
                
            msr_tokens_added = msr_guide_latent.shape[2] * msr_guide_latent.shape[3] * msr_guide_latent.shape[4]

            # Injects keyframe_idxs into conditioning (we clone the latent/mask to prevent in-place modification of the whole video)
            positive, negative, _, _ = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, 0, latent_image.clone(), noise_mask.clone(), msr_guide_latent,
                0.0, scale_factors, guide_mask=msr_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=True
            )

            # Register MSR attention (but DO NOT modify the latent_image with this video)
            positive = iclora_attention.append_guide_attention_entry(
                positive, msr_tokens_added, msr_guide_orig_shape,
                attention_strength=msr_attention_strength, attention_mask=norm_msr_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, msr_tokens_added, msr_guide_orig_shape,
                attention_strength=msr_attention_strength, attention_mask=norm_msr_mask
            )

        # --- STEP 4: APPLY LAST FRAME (LF) ---
        if last_frame is not None:
            num_frames_to_keep = ((last_frame.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
            causal_fix = (num_frames_to_keep == 1)
            
            lf_image = last_frame
            if not causal_fix:
                lf_image = torch.cat([lf_image[:1], lf_image], dim=0)

            lf_image, lf_guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                vae=vae, latent_width=latent_width, latent_height=latent_height, images=lf_image,
                scale_factors=scale_factors, latent_downscale_factor=latent_downscale_factor,
                crop=crop, use_tiled_encode=use_tiled_encode, tile_size=tile_size, tile_overlap=tile_overlap
            )

            if not causal_fix:
                lf_guide_latent = lf_guide_latent[:, :, 1:, :, :]
                lf_image = lf_image[1:]

            lf_guide_orig_shape = list(lf_guide_latent.shape[2:])
            lf_guide_mask = None

            if latent_downscale_factor > 1:
                dilated = latents_mod.LTXVDilateLatent().dilate_latent(
                    {"samples": lf_guide_latent},
                    horizontal_scale=int(latent_downscale_factor),
                    vertical_scale=int(latent_downscale_factor),
                )[0]
                lf_guide_mask = dilated["noise_mask"]
                lf_guide_latent = dilated["samples"]

            lf_tokens_added = lf_guide_latent.shape[2] * lf_guide_latent.shape[3] * lf_guide_latent.shape[4]
            lf_guide_len = lf_guide_latent.shape[2]

            # Secure indexing formula from index calculation bugs:
            lf_latent_idx = latent_length - lf_guide_len
            lf_calculated_frame_idx = lf_latent_idx * time_scale_factor

            assert lf_latent_idx >= 0, "Last frame index is out of bounds."
            
            # Store reference for lock
            lf_guide_latent_ref = lf_guide_latent
            lf_latent_idx_ref = lf_latent_idx
            lf_guide_len_ref = lf_guide_len

            # Injects keyframe conditionings
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, lf_calculated_frame_idx, latent_image, noise_mask, lf_guide_latent,
                last_frame_strength, scale_factors, guide_mask=lf_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
            )

            # Register attention
            positive = iclora_attention.append_guide_attention_entry(
                positive, lf_tokens_added, lf_guide_orig_shape,
                attention_strength=last_frame_attention_strength, attention_mask=norm_lf_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, lf_tokens_added, lf_guide_orig_shape,
                attention_strength=last_frame_attention_strength, attention_mask=norm_lf_mask
            )

        # --- STEP 5: APPLY INPLACE LOCKS ---
        # This ensures MSR or any other modifications don't overwrite the locked First/Last frames
        if first_frame is not None and lock_first_frame and ff_guide_latent_ref is not None:
            for i in range(lock_first_frame_len):
                if i < latent_length:
                    g_frame = min(i, ff_guide_latent_ref.shape[2] - 1)
                    latent_image[:, :, i:i+1] = ff_guide_latent_ref[:, :, g_frame:g_frame+1]
                    noise_mask[:, :, i:i+1] = 0.0

        if last_frame is not None and lock_last_frame and lf_guide_latent_ref is not None:
            lf_end_idx = min(lf_latent_idx_ref + lf_guide_len_ref, latent_length)
            latent_image[:, :, lf_latent_idx_ref:lf_end_idx] = lf_guide_latent_ref[:, :, :lf_end_idx - lf_latent_idx_ref]
            noise_mask[:, :, lf_latent_idx_ref:lf_end_idx] = 0.0

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

