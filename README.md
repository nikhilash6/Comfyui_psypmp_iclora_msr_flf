# 🅛🅣🅧 MSR IC-LoRA FLF (Standalone)

**Version:** 0.9.5

## 📝 Changelog
* **v0.9.5:** Fixed a critical bug in temporal coordinate alignment for the MSR video sequence. Previously, MSR video latents were dropped during conditioning, causing the temporal grid (RoPE) to shift backward into the main generated video. This resulted in broken motion at the end of the video and the failure of the `lock_last_frame` mechanism. The latents are now physically appended to maintain exact temporal synchronization, ensuring perfectly smooth motion and reliable frame locking up to the very last frame.
* **v0.9.4:** Complete architectural overhaul based on the dual-chain method demonstrated in [this video](https://www.youtube.com/watch?v=uirABckAK4o&). The node now fundamentally separates MSR processing into two isolated streams:
  1. **Cross-Attention Stream:** MSR images are assembled into a full video sequence (acting like `Licon-MSR`) and processed via IC-LoRA to extract ONLY conditioning metadata. This provides character/style consistency without overwriting physical frames.
  2. **Latent Injection Stream:** Individual MSR images, alongside the First and Last frames, are injected directly into the physical latent space at their precise indices (index `0` for MSR/First Frame, and the end index for Last Frame) using logic adapted from `LTXVAddGuideMulti`.

A highly optimized, monolithic ComfyUI custom node that integrates **Multi-Subject Reference (MSR)**, **First Frame (FL)**, and **Last Frame (LF)** into a single, high-performance module for LTX-Video.

By unifying these three features, this node permanently solves critical timeline indexing bugs, prevents memory crashes (`IndexError`), and provides perfect physical pixel preservation via hard-locking mechanisms (Inplace-locking) on the noise mask.

---

## 📖 Languages
* [Russian Version (Русская версия)](README_RU.md)

---

## 🛠️ The Problems It Solves

### 1. The Multi-Node Timeline Conflict (Index Shifting)
* **The Issue:** When chaining three separate `IC-LoRA Guide` nodes (for MSR, First Frame, and Last Frame) sequentially, early nodes (like MSR) inject reference tokens into the conditioning, artificially bloating the token count. Subsequent nodes (like Last Frame) try to calculate their timeline position based on this bloated length, causing the end frame to apply way too early (e.g., at the 12th second instead of the 18th).
* **The Crash:** Manually forcing the correct frame index on the UI often violates ComfyUI's internal assertions, triggering an `IndexError: list index out of range` inside the `prompt_worker` thread and crashing the entire model memory stack.
* **The Solution:** Our node performs **isolated timeline length tracking** strictly *before* token injection. We completely bypass the flawed automatic index calculation and use direct mathematical latent injection:
  $$\text{latent\_idx} = \text{original\_latent\_length} - \text{guide\_length}$$
  This ensures the Last Frame is anchored exactly at the final moment of the video, preventing overlaps, early-termination bugs, and memory crashes.

### 2. The "Soft" (Drifting) Frame Problem
* **The Issue:** Standard IC-LoRA acts as a guide (a conceptual magnet), not a hard physical lock. Because of this, the first and last frames can drift, deform, morph, or lose details during sampling as the model attempts to smooth out animations.
* **The Solution:** We introduced **Inplace Latent Locking** (`lock_first_frame` and `lock_last_frame`). When enabled, the node intercepts the noise mask for those specific frames and forces their noise level to `0.0`:
  ```python
  noise_mask[:, :, 0, :, :] = 0.0  # Physical freeze of the latent step
  ```
  This physically locks the starting and ending pixels in place, while IC-LoRA's cross-attention seamlessly blends the motion across the remaining timeline without visible seams.

### 3. The LTX-Video MSR Temporal Coordinate Alignment Bug (Fixed in v0.9.5)
* **The Issue:** In the previous implementation of MSR attention (Cross-Attention), a virtual sequence of frames (e.g., 41 frames) was generated and passed to `append_keyframe` on Step 3 to register its RoPE coordinates in the conditioning. However, the returned physical latent tensor and noise mask were discarded (using `positive, negative, _, _ = append_keyframe(...)`) to prevent "polluting" the main video with reference images.
* **The Core Physics of the Bug:** Inside the LTX-Video core model (`comfy/ldm/lightricks/model.py`), keyframe positioning relies on a strict offset from the end of the physical latent sequence:
  $$\text{Token Slice} = -K$$
  where $K$ is the total number of registered keyframes in the conditioning (including FF, MSR sequences, and LF). 
  If we register 13 keyframes (1 FF + 5 MSR-img + 6 MSR-vid + 1 LF) but physically append only 7, the sampler core overwrites the temporal coordinates of the last 13 tokens. This causes the model to invade the main generated video **6 frames deep** (pixels 313–360) and assign them a temporal coordinate of $t = 0$.
* **The Symptoms:** The motion completely collapses near the end of the video, moving backward, freezing, or degenerating into chaotic noise. The temporal connection to the locked Last Frame (at $t=360$) is severed, rendering `lock_last_frame` visually non-functional.
* **The Solution:** We resolved this by preserving the returned `latent_image` and `noise_mask` from the `append_keyframe` call in Step 3 and incrementing the `num_appended` counter accordingly. All 13 appended frames are now kept physically in sync with RoPE coordinates during sampling, and our patched `LTXVCropGuides` cleanly chops them off at the end of execution, yielding a perfect final frame lock and seamless end-of-video motion.

---

## 🌟 Key Features

* **Monolithic Node Design:** Consolidates MSR, First Frame, and Last Frame into one node. No more spaghetti wiring on your canvas.
* **Smart MSR Sequence Builder:** Accepts up to 4 subject images + 1 background image. It automatically resizes, duplicates, and balances them evenly into a high-quality MSR sequence (e.g., 17, 25, 33, 41 frames) to maintain consistent subject/style identity.
* **Hard Inplace Pixel Locking:** Zeroes out noise masks on first/last frames to guarantee 100% identical starting and ending frames on the video boundaries.
* **Tiled VAE Encoding:** Native support for tiled VAE encoding (`tile_size`, `tile_overlap`) to prevent GPU Out-of-Memory (OOM) crashes when processing heavy reference images (e.g., 1280x736 or higher).
* **Cross-Attention Masking:** Connect an optional black-and-white `attention_mask` to restrict IC-LoRA's guidance strictly to your subject, preventing background bleeding.
* **Latent Downscaling Support:** Built-in downscaling (`latent_downscale_factor` of 1.0 or 2.0) to match specific pre-trained compressed IC-LoRA models.

---

## 📐 Node Interface & Inputs

![LTX MSR IC-LoRA FLF Standalone Node](node.jpg)

| Input Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **positive** | CONDITIONING | *Required* | Positive text conditioning to append reference attention tokens to. |
| **negative** | CONDITIONING | *Required* | Negative text conditioning. |
| **vae** | VAE | *Required* | VAE model used to encode the reference images. |
| **latent** | LATENT | *Required* | The main video latent container (defines video dimensions and length). |
| **msr_width / msr_height** | INT | `1920 / 1088` | Dimensions to resize all MSR images to before VAE encoding. |
| **msr_frame_count** | COMBO | `41` | Total frames in the MSR sequence (17, 25, 33, 41). Higher = better identity but more VRAM. |
| **msr_image_1 to 4** | IMAGE | *Optional* | Reference images of your subject/character. |
| **msr_background** | IMAGE | *Optional* | Background reference image (placed at the end of the MSR sequence). |
| **msr_strength** | FLOAT | `0.90` | Injection strength of the MSR latents directly into latent space. |
| **msr_attention_strength**| FLOAT | `1.0` | Cross-attention influence of the MSR sequence on the generator. |
| **first_frame** | IMAGE | *Optional* | The starting image of the video. |
| **lock_first_frame** | BOOLEAN | `True` | Enables physical inplace locking for the first frame. |
| **lock_first_frame_len** | INT | `1` | Number of latent frames to freeze (prevents VAE block interpolation blur). |
| **first_frame_strength** | FLOAT | `0.85` | Direct latent injection strength for the first frame. |
| **first_frame_attn_strength**| FLOAT | `0.90` | Attention influence of the first frame. |
| **last_frame** | IMAGE | *Optional* | The ending image of the video. |
| **lock_last_frame** | BOOLEAN | `True` | Enables physical inplace locking for the last frame. |
| **last_frame_strength** | FLOAT | `0.85` | Direct latent injection strength for the last frame. |
| **last_frame_attn_strength**| FLOAT | `0.90` | Attention influence of the last frame. |
| **latent_downscale_factor**| FLOAT | `1.0` | Downscaling applied to references (e.g., 2.0 to match compressed IC-LoRA). |
| **crop** | COMBO | `center` | Image resize crop mode (`disabled` stretches, `center` crops aspect ratio). |
| **use_tiled_encode** | BOOLEAN | `False` | Enables tiled VAE encoding to prevent Out-Of-Memory errors. |
| **attention_mask** | MASK | *Optional* | Black & white spatial mask to restrict attention focus. |

---

## 🚀 Installation

1. Navigate to your ComfyUI directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Comfyui_psypmp_iclora_msr_flf.git
   ```

---

## 🤝 Credits & Acknowledgements

This custom node builds upon and merges the outstanding work of:
* **Lightricks / ComfyUI-LTXVideo** for the original IC-LoRA attention mechanics.
* **Licon-MSR** for the multi-subject sequence generation methodology.
