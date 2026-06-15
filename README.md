# 🅛🅣🅧 MSR IC-LoRA FLF (Standalone)

A highly optimized, monolithic ComfyUI custom node that integrates **Multi-Subject Reference (MSR)**, **First Frame (FL)**, and **Last Frame (LF)** conditioning into a single, high-performance module for LTX-Video. 

By unifying these three features, this node solves critical timeline indexing bugs, prevents memory crashes (`IndexError`), and provides perfect physical pixel preservation via hard-locking mechanisms.

---

## 📖 Languages
* [Russian Version (Русская версия)](README_RU.md)

---

## 🛠️ The Problems It Solves

### 1. The Multi-Node Timeline Conflict (Index Shifting)
* **The Issue:** When chaining three separate `IC-LoRA Guide` nodes (for MSR, First Frame, and Last Frame) sequentially, early nodes (like MSR) inject reference tokens into the conditioning, artificially bloating the token count. Subsequent nodes (like Last Frame) try to calculate their timeline position based on this bloated length, causing the end frame to apply way too early (e.g., at the 12th second instead of the 18th).
* **The Crash:** Manually forcing the correct frame index on the UI often violates ComfyUI's internal assertions, triggering an `IndexError: list index out of range` inside the `prompt_worker` thread and crashing the entire model memory stack.
* **The Solution:** Our node performs **isolated timeline length tracking** strictly *before* token injection. It bypasses the flawed index calculation and uses direct mathematical latent injection:
  $$\text{latent\_idx} = \text{original\_latent\_length} - \text{guide\_length}$$
  This ensures the Last Frame is anchored exactly at the final moment of the video, preventing overlaps, early-termination bugs, and memory crashes.

### 2. The "Soft" (Drifting) Frame Problem
* **The Issue:** Standard IC-LoRA acts as a guide (a conceptual magnet), not a hard physical lock. Because of this, the first and last frames can drift, deform, morph, or lose details during sampling as the model attempts to smooth out animations.
* **The Solution:** We introduced **Inplace Latent Locking** (`lock_first_frame` and `lock_last_frame`). When enabled, the node intercepts the noise mask for those specific frames and forces their noise level to `0.0`:
  ```python
  noise_mask[:, :, 0, :, :] = 0.0  # Physical freeze of the latent step
  ```
  This physically locks the starting and ending pixels in place, while IC-LoRA's cross-attention seamlessly blends the motion across the remaining timeline without visible seams.

### 3. Sibling Module Loading Issues
* **The Issue:** Many custom nodes fail if the parent folder name is changed (e.g., from `ComfyUI-LTXVideo` to `ComfyUI-LTXVideo-master`) or if competing fork packages (like AVSplit) are present.
* **The Solution:** Includes an **Ultra-Robust Sibling Module Resolver** that dynamically scans the custom nodes directory, prioritizes exact matches, avoids alternative modules, and safely handles imports on the fly.

---

## 🌟 Key Features

* **Monolithic Node Design:** Consolidates MSR, First Frame, and Last Frame into one node. Reduces spaghetti wiring in your workflows.
* **Smart MSR Sequence Builder:** Accept up to 4 subject images + 1 background image. It automatically resizes, duplicates, and balances them evenly into a high-quality MSR sequence (e.g., 17, 25, 33, 41 frames) to guide character/style identity.
* **Hard Inplace Pixel Locking:** Zeroes out noise masks on first/last frames to guarantee identical starting and ending frames.
* **Tiled VAE Encoding:** Native support for tiled VAE encoding (`tile_size`, `tile_overlap`) to prevent GPU Out-of-Memory (OOM) crashes when processing high-resolution reference images (e.g., 1280x736 or higher).
* **Cross-Attention Masking:** Connect an optional black-and-white `attention_mask` to restrict IC-LoRA's guidance strictly to your subject, preventing background bleeding.
* **Latent Downscaling Support:** Built-in downscaling (`latent_downscale_factor` of 1.0 or 2.0) to match specific pre-trained compressed IC-LoRA models.

---

## 📐 Node Interface & Inputs

### UI Screenshot Placeholder
> [!NOTE]
> *Insert a screenshot of the `🅛🅣🅧 MSR IC LORA FLF (Standalone)` node in your ComfyUI workspace here.*

| Input Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| **positive** | CONDITIONING | *Required* | Positive text conditioning to append reference attention tokens to. |
| **negative** | CONDITIONING | *Required* | Negative text conditioning. |
| **vae** | VAE | *Required* | VAE model used to encode the reference images. |
| **latent** | LATENT | *Required* | The main video latent container (defines video dimensions and length). |
| **msr_width / msr_height** | INT | `736 / 1280` | Dimensions to resize all MSR images to before VAE encoding. |
| **msr_frame_count** | COMBO | `17` | Total frames in the MSR sequence (17, 25, 33, 41). Higher = better identity but more VRAM. |
| **msr_image_1 to 4** | IMAGE | *Optional* | Reference images of your subject/character. |
| **msr_background** | IMAGE | *Optional* | Background reference image (placed at the end of the MSR sequence). |
| **msr_strength** | FLOAT | `1.0` | Injection strength of the MSR latents directly into latent space. |
| **msr_attention_strength**| FLOAT | `1.0` | Cross-attention influence of the MSR sequence on the generator. |
| **first_frame** | IMAGE | *Optional* | The starting image of the video. |
| **lock_first_frame** | BOOLEAN | `False` | Enables physical inplace locking for the first frame. |
| **lock_first_frame_len** | INT | `2` | Number of latent frames to freeze (prevents VAE block interpolation blur). |
| **first_frame_strength** | FLOAT | `1.0` | Direct latent injection strength for the first frame. |
| **first_frame_attn_strength**| FLOAT | `1.0` | Attention influence of the first frame. |
| **last_frame** | IMAGE | *Optional* | The ending image of the video. |
| **lock_last_frame** | BOOLEAN | `False` | Enables physical inplace locking for the last frame. |
| **last_frame_strength** | FLOAT | `1.0` | Direct latent injection strength for the last frame. |
| **last_frame_attn_strength**| FLOAT | `1.0` | Attention influence of the last frame. |
| **latent_downscale_factor**| FLOAT | `1.0` | Downscaling applied to references (e.g., 2.0 to match compressed IC-LoRA). |
| **crop** | COMBO | `disabled` | Image resize crop mode (`disabled` stretches, `center` crops aspect ratio). |
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
3. Restart ComfyUI. The node will appear under the category:
   `Lightricks/IC-LoRA` $\rightarrow$ `🅛🅣🅧 MSR IC LORA FLF (Standalone)`

---

## 💡 Recommended Workflow

1. Load your target **Latent** (e.g., empty latent 736x1280, 97 frames).
2. Connect your **Positive** and **Negative** conditionings from your prompt encoder.
3. Wire your **VAE** into the node's `vae` input.
4. Input your assets:
   * Load your starting image into `first_frame`. Enable `lock_first_frame` (set length to `2` to keep it sharp).
   * Load up to 4 angles of your character into `msr_image_1` to `4` to maintain consistent subject features.
   * Load your ending image into `last_frame` and enable `lock_last_frame` for a seamless loop or specific transition.
5. Connect the outputs (`positive`, `negative`, `latent`) directly into your sampler or guider.

---

## 🤝 Credits & Acknowledgements

This custom node builds upon and merges the outstanding work of:
* **Lightricks / ComfyUI-LTXVideo** for the original IC-LoRA attention mechanics.
* **Licon-MSR** for the multi-subject sequence generation methodology.
