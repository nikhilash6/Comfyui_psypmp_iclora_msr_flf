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

    current_file_path = Path(__file__).resolve()
    custom_nodes_dir = current_file_path.parent.parent

    ltx_video_dir = None
    
    for p in custom_nodes_dir.iterdir():
        if p.is_dir() and p.name in ("ComfyUI-LTXVideo", "ComfyUI_LTXVideo"):
            ltx_video_dir = p
            break
            
    if ltx_video_dir is None:
        for p in custom_nodes_dir.iterdir():
            if p.is_dir() and ("ComfyUI-LTXVideo" in p.name or "ComfyUI_LTXVideo" in p.name):
                if "AVSplit" not in p.name:
                    ltx_video_dir = p
                    break

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

    custom_nodes_str = str(custom_nodes_dir)
    added_to_sys_path = False
    if custom_nodes_str not in sys.path:
        sys.path.insert(0, custom_nodes_str)
        added_to_sys_path = True

    try:
        full_module_name = f"{pkg_name}.{module_name}"
        if full_module_name in sys.modules:
            return sys.modules[full_module_name]

        return importlib.import_module(full_module_name)
    finally:
        if added_to_sys_path:
            sys.path.remove(custom_nodes_str)


# =====================================================================
# DYNAMIC CROPPING PATCH FOR LTXVCropGuides
# =====================================================================
def patch_crop_guides():
    try:
        import comfy_extras.nodes_lt as nodes_lt
        import node_helpers
        from comfy_api.latest import io

        if getattr(nodes_lt.LTXVCropGuides, "_patched_for_msr_flf", False):
            return

        @classmethod
        def patched_execute(cls, positive, negative, latent) -> io.NodeOutput:
            latent_image = latent["samples"].clone()
            noise_mask = nodes_lt.get_noise_mask(latent)

            num_appended = None
            for cond in (positive, negative):
                for t in cond:
                    if isinstance(t[1], dict) and "num_physically_appended_keyframes" in t[1]:
                        num_appended = t[1]["num_physically_appended_keyframes"]
                        break
                if num_appended is not None:
                    break

            if num_appended is not None:
                num_keyframes = num_appended
            else:
                _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, latent_image.shape)

            if num_keyframes == 0:
                return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

            latent_image = latent_image[:, :, :-num_keyframes] if num_keyframes > 0 else latent_image
            noise_mask = noise_mask[:, :, :-num_keyframes] if (noise_mask is not None and num_keyframes > 0) else noise_mask

            positive = node_helpers.conditioning_set_values(positive, {
                "keyframe_idxs": None,
                "guide_attention_entries": None,
            })
            negative = node_helpers.conditioning_set_values(negative, {
                "keyframe_idxs": None,
                "guide_attention_entries": None,
            })

            return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

        nodes_lt.LTXVCropGuides.execute = patched_execute
        nodes_lt.LTXVCropGuides._patched_for_msr_flf = True
    except Exception as e:
        print(f"Error applying LTXVCropGuides monkey patch: {e}")

try:
    patch_crop_guides()
except:
    pass


class LTXMSRICLoRAFLF_Experimental(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXMSRICLoRAFLF_Experimental",
            display_name="🅛🅣🅧 MSR IC LORA FLF (Test/Experimental)",
            category="Lightricks/IC-LoRA",
            description="Экспериментальная нода MSR IC-LoRA с индивидуальными масками и переключателями этапов.",
            inputs=[
                io.Conditioning.Input("positive", tooltip="Позитивный текстовый промпт. Сюда будут добавлены данные IC-LoRA внимания."),
                io.Conditioning.Input("negative", tooltip="Негативный текстовый промпт. Сюда будут добавлены данные IC-LoRA внимания."),
                io.Vae.Input("vae", tooltip="VAE модель для кодирования картинок в латентное пространство."),
                io.Latent.Input("latent", tooltip="Главный латент видео, который задает длину, ширину и высоту."),
                
                # Experimental Toggles
                io.Boolean.Input("enable_msr_latent_injection", default=True, tooltip="Включить внедрение MSR референсов напрямую в латент нулевого кадра. Выключите, чтобы проверить чистую работу внимания (Cross-Attention)."),
                io.Boolean.Input("enable_msr_attention", default=True, tooltip="Включить влияние MSR через механизм внимания (Cross-Attention)."),

                # MSR Global Settings
                io.Int.Input("msr_width", default=1920, min=32, max=8192, step=32, tooltip="Ширина ресайза для MSR изображений."),
                io.Int.Input("msr_height", default=1088, min=32, max=8192, step=32, tooltip="Высота ресайза для MSR изображений."),
                io.Combo.Input("msr_frame_count", options=["17", "25", "33", "41"], default="41", tooltip="Общее количество кадров MSR-секвенции. Изображения будут размножены для заполнения этой длины."),
                io.Float.Input("msr_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила влияния MSR секвенции на внимание (Cross-Attention) всей модели."),
                
                # MSR Image 1
                io.Image.Input("msr_image_1", optional=True, tooltip="MSR Референс 1."),
                io.Mask.Input("msr_mask_1", optional=True, tooltip="Маска для MSR Референса 1 (белое = объект)."),
                io.Float.Input("msr_strength_1", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта MSR Референса 1 в латент."),
                
                # MSR Image 2
                io.Image.Input("msr_image_2", optional=True, tooltip="MSR Референс 2."),
                io.Mask.Input("msr_mask_2", optional=True, tooltip="Маска для MSR Референса 2 (белое = объект)."),
                io.Float.Input("msr_strength_2", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта MSR Референса 2 в латент."),

                # MSR Image 3
                io.Image.Input("msr_image_3", optional=True, tooltip="MSR Референс 3."),
                io.Mask.Input("msr_mask_3", optional=True, tooltip="Маска для MSR Референса 3 (белое = объект)."),
                io.Float.Input("msr_strength_3", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта MSR Референса 3 в латент."),

                # MSR Image 4
                io.Image.Input("msr_image_4", optional=True, tooltip="MSR Референс 4."),
                io.Mask.Input("msr_mask_4", optional=True, tooltip="Маска для MSR Референса 4 (белое = объект)."),
                io.Float.Input("msr_strength_4", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта MSR Референса 4 в латент."),

                # MSR Background
                io.Image.Input("msr_background", optional=True, tooltip="Фоновый MSR Референс."),
                io.Mask.Input("msr_background_mask", optional=True, tooltip="Маска для Фонового MSR Референса."),
                io.Float.Input("msr_background_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта Фонового MSR Референса в латент."),
                
                # First Frame Inputs & Settings
                io.Image.Input("first_frame", optional=True, tooltip="Первый кадр (First Frame)."),
                io.Mask.Input("first_frame_mask", optional=True, tooltip="Маска внимания для первого кадра."),
                io.Boolean.Input("lock_first_frame", default=False, tooltip="Заморозить латенты первого кадра, чтобы видео начиналось строго с него."),
                io.Int.Input("lock_first_frame_len", default=1, min=1, max=16, step=1, tooltip="Сколько латентных кадров первого кадра заморозить."),
                io.Float.Input("first_frame_strength", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта первого кадра в латент."),
                io.Float.Input("first_frame_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила внимания на первый кадр (Cross-Attention)."),
                
                # Last Frame Inputs & Settings
                io.Image.Input("last_frame", optional=True, tooltip="Последний кадр (Last Frame)."),
                io.Mask.Input("last_frame_mask", optional=True, tooltip="Маска внимания для последнего кадра."),
                io.Boolean.Input("lock_last_frame", default=False, tooltip="Заморозить латенты последнего кадра."),
                io.Float.Input("last_frame_strength", default=0.85, min=0.0, max=1.0, step=0.01, tooltip="Сила инжекта последнего кадра в латент."),
                io.Float.Input("last_frame_attention_strength", default=1.00, min=0.0, max=1.0, step=0.01, tooltip="Сила внимания на последний кадр (Cross-Attention)."),
                
                # IC-LoRA General Settings
                io.Float.Input("latent_downscale_factor", default=1.0, min=1.0, max=10.0, step=1.0, tooltip="Фактор уменьшения латентов (downscale). 1.0 = обычный, 2.0 = сжатый IC-LoRA."),
                io.Combo.Input("crop", options=["disabled", "center"], default="center", tooltip="Тип кропа при ресайзе. center = сохраняет пропорции и обрезает края."),
                io.Boolean.Input("use_tiled_encode", default=False, tooltip="Использовать Tiled VAE (рекомендуется для предотвращения OOM)."),
                io.Int.Input("tile_size", default=256, min=64, max=512, step=32, tooltip="Размер тайла при Tiled Encode."),
                io.Int.Input("tile_overlap", default=64, min=16, max=256, step=16, tooltip="Перекрытие тайлов при Tiled Encode."),
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
        enable_msr_latent_injection,
        enable_msr_attention,
        msr_width,
        msr_height,
        msr_frame_count,
        msr_image_1=None, msr_mask_1=None, msr_strength_1=1.00,
        msr_image_2=None, msr_mask_2=None, msr_strength_2=1.00,
        msr_image_3=None, msr_mask_3=None, msr_strength_3=1.00,
        msr_image_4=None, msr_mask_4=None, msr_strength_4=1.00,
        msr_background=None, msr_background_mask=None, msr_background_strength=1.00,
        msr_attention_strength=1.00,
        first_frame=None, first_frame_mask=None,
        lock_first_frame=False, lock_first_frame_len=1,
        first_frame_strength=0.85, first_frame_attention_strength=1.00,
        last_frame=None, last_frame_mask=None,
        lock_last_frame=False,
        last_frame_strength=0.85, last_frame_attention_strength=1.00,
        latent_downscale_factor=1.0,
        crop="center",
        use_tiled_encode=False,
        tile_size=256,
        tile_overlap=64,
    ) -> io.NodeOutput:
        
        iclora_attention = _get_ltx_module("iclora_attention")
        latents_mod = _get_ltx_module("latents")
        iclora_mod = _get_ltx_module("iclora")
        import comfy_extras.nodes_lt as nodes_lt

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()
        
        noise_mask = nodes_lt.get_noise_mask(latent)
        if noise_mask is None:
            b, _, l_len, l_h, l_w = latent_image.shape
            noise_mask = torch.ones((b, 1, l_len, l_h, l_w), dtype=torch.float32, device=latent_image.device)
        else:
            noise_mask = noise_mask.clone()

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = scale_factors[0]
        num_appended = 0
        
        norm_ff_mask = iclora_attention.normalize_mask(first_frame_mask) if first_frame_mask is not None else None
        norm_lf_mask = iclora_attention.normalize_mask(last_frame_mask) if last_frame_mask is not None else None

        ff_guide_latent_ref = None
        lf_guide_latent_ref = None
        lf_latent_idx_ref = None
        lf_guide_len_ref = None

        # --- STEP 1: APPLY FIRST FRAME (FL) ---
        if first_frame is not None:
            causal_fix = True 
            ff_image = first_frame

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
            ff_guide_latent_ref = ff_guide_latent

            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, 0, latent_image, noise_mask, ff_guide_latent,
                first_frame_strength, scale_factors, guide_mask=ff_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
            )
            num_appended += ff_guide_latent.shape[2]

            positive = iclora_attention.append_guide_attention_entry(
                positive, ff_tokens_added, ff_guide_orig_shape,
                attention_strength=first_frame_attention_strength, attention_mask=norm_ff_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, ff_tokens_added, ff_guide_orig_shape,
                attention_strength=first_frame_attention_strength, attention_mask=norm_ff_mask
            )

        # --- PREPARE MSR IMAGES ---
        msr_data = []
        for img, mask, strength in [
            (msr_image_1, msr_mask_1, msr_strength_1),
            (msr_image_2, msr_mask_2, msr_strength_2),
            (msr_image_3, msr_mask_3, msr_strength_3),
            (msr_image_4, msr_mask_4, msr_strength_4),
            (msr_background, msr_background_mask, msr_background_strength),
        ]:
            if img is not None:
                msr_data.append({"image": img, "mask": mask, "strength": strength})

        # --- STEP 2: MULTI-GUIDE LATENT INJECTION (MSR CHAIN 1) ---
        if enable_msr_latent_injection and len(msr_data) > 0:
            for item in msr_data:
                img_encoded, guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                    vae=vae, latent_width=latent_width, latent_height=latent_height, images=item["image"],
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

                positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                    positive, negative, 0, latent_image, noise_mask, guide_latent,
                    item["strength"], scale_factors, guide_mask=guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=True
                )
                num_appended += guide_latent.shape[2]

        # --- STEP 3: MSR VIDEO CONDITIONING (MSR CHAIN 2) ---
        if enable_msr_attention and len(msr_data) > 0:
            fc = int(msr_frame_count)
            resized_images = []
            resized_masks = []
            
            for item in msr_data:
                img = item["image"]
                mask = item["mask"]
                
                # Resize image
                resized_img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), msr_width, msr_height, "bilinear", crop=crop
                ).movedim(1, -1)
                resized_images.append(resized_img)
                
                # Resize mask
                if mask is not None:
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0) # (1, H, W)
                    mask_tensor = mask.unsqueeze(0).unsqueeze(0) # (1, 1, 1, H, W)
                    
                    # common_upscale requires (B, C, H, W)
                    # We pass (1, 1, H, W)
                    mask_upscale_target = mask_tensor.squeeze(2) # (1, 1, H, W)
                    resized_m = comfy.utils.common_upscale(
                        mask_upscale_target, msr_width, msr_height, "bilinear", crop=crop
                    ) # (1, 1, H_new, W_new)
                    resized_m = resized_m.unsqueeze(2) # (1, 1, 1, H_new, W_new)
                    resized_masks.append(resized_m)
                else:
                    white_mask = torch.ones((1, 1, 1, msr_height, msr_width), dtype=torch.float32, device=img.device)
                    resized_masks.append(white_mask)

            base_count = fc // len(resized_images)
            remainder = fc % len(resized_images)
            msr_sequence_frames = []
            msr_sequence_masks = []
            
            for index, (img_tensor, mask_tensor) in enumerate(zip(resized_images, resized_masks)):
                repeats = base_count + (1 if index < remainder else 0)
                for _ in range(repeats):
                    msr_sequence_frames.append(img_tensor)
                    msr_sequence_masks.append(mask_tensor)
            
            msr_video = torch.cat(msr_sequence_frames, dim=0) # [fc, H, W, 3]
            msr_video_mask = torch.cat(msr_sequence_masks, dim=2) # [1, 1, fc, H, W]

            num_frames_to_keep = ((msr_video.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
            msr_video = msr_video[:num_frames_to_keep]
            msr_video_mask = msr_video_mask[:, :, :num_frames_to_keep, :, :]

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

            # Физически приклеиваем MSR-видео к латенту для сохранения баланса временных координат
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, 0, latent_image, noise_mask, msr_guide_latent,
                msr_strength_1, scale_factors, guide_mask=msr_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=True
            )
            num_appended += msr_guide_latent.shape[2]

            positive = iclora_attention.append_guide_attention_entry(
                positive, msr_tokens_added, msr_guide_orig_shape,
                attention_strength=msr_attention_strength, attention_mask=msr_video_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, msr_tokens_added, msr_guide_orig_shape,
                attention_strength=msr_attention_strength, attention_mask=msr_video_mask
            )

        # --- STEP 4: APPLY LAST FRAME (LF) ---
        if last_frame is not None:
            # We must encode the last frame as a "continuation" latent so it decodes correctly
            # at the end of the video. If we encode it natively, it becomes a causal start latent,
            # which causes severe glitches when injected at latent_idx > 0.
            # To fix this, we pad the beginning with `time_scale_factor` (8) frames.
            lf_image = last_frame
            pad_frames = time_scale_factor
            lf_image_padded = torch.cat([lf_image[0:1].repeat(pad_frames, 1, 1, 1), lf_image], dim=0)

            lf_image_encoded, lf_guide_latent = iclora_mod.LTXAddVideoICLoRAGuide.encode(
                vae=vae, latent_width=latent_width, latent_height=latent_height, images=lf_image_padded,
                scale_factors=scale_factors, latent_downscale_factor=latent_downscale_factor,
                crop=crop, use_tiled_encode=use_tiled_encode, tile_size=tile_size, tile_overlap=tile_overlap
            )

            # Drop the causal start latent (the first latent frame corresponds to the 8 pad frames)
            lf_guide_latent = lf_guide_latent[:, :, 1:, :, :]
            causal_fix = False

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

            lf_latent_idx = latent_length - lf_guide_len
            lf_calculated_frame_idx = lf_latent_idx * time_scale_factor

            assert lf_latent_idx >= 0, "Last frame index is out of bounds."
            
            lf_guide_latent_ref = lf_guide_latent
            lf_latent_idx_ref = lf_latent_idx
            lf_guide_len_ref = lf_guide_len

            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, lf_calculated_frame_idx, latent_image, noise_mask, lf_guide_latent,
                last_frame_strength, scale_factors, guide_mask=lf_guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
            )
            num_appended += lf_guide_latent.shape[2]

            positive = iclora_attention.append_guide_attention_entry(
                positive, lf_tokens_added, lf_guide_orig_shape,
                attention_strength=last_frame_attention_strength, attention_mask=norm_lf_mask
            )
            negative = iclora_attention.append_guide_attention_entry(
                negative, lf_tokens_added, lf_guide_orig_shape,
                attention_strength=last_frame_attention_strength, attention_mask=norm_lf_mask
            )

        # --- STEP 5: APPLY INPLACE LOCKS ---
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

        import node_helpers
        positive = node_helpers.conditioning_set_values(positive, {"num_physically_appended_keyframes": num_appended})
        negative = node_helpers.conditioning_set_values(negative, {"num_physically_appended_keyframes": num_appended})

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})
