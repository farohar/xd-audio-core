# RFC 0001: Native 4D Spatio-Temporal Lattice Space for Generative Audio Inference

**Author:** farohar <https://github.com/farohar>  
**Status:** Experimental / Request for Comments (RFC)  
**Created:** July 2026  
**License:** MIT

---

## 1. Objective & Abstract

This RFC proposes a fundamental architectural paradigm shift for generative audio models (Audio Diffusion, Flow-Matching, and Audio Transformers) to solve the chronic issue of **Latent Dimensional Collapse**. 

Current state-of-the-art models project multi-channel acoustic physical realities into a flat **2D Spectrogram (Time × Frequency)** canvas. This non-injective projection forces overlapping frequencies into the exact same pixel coordinates, leading to catastrophic masking, dynamic ducking, and destructive phase-cancellation during source separation or high-density music inference (e.g., dense heavy metal arrangements or orchestral mixes).

We propose **Project XD-Audio**: a native **4D Tensor Matrix Architecture (Time, Frequency, Panorama, Phase)** that treats spatial coordinates and wave phase relations as rigid, unyielding architectural axes rather than learned latent features. This ensures that overlapping frequencies occupy discrete memory addresses, rendering inference-based masking physically impossible.

---

## 2. The Root Cause: 2D Projection Failure

In high-density music production, multiple sonic objects share identical frequencies simultaneously but remain perfectly separated in the physical realm via spatial panning and phase offsets (e.g., double-tracked guitars panned hard left/right and vocals centered at 0.0). 

When squeezed into a legacy 2D Mel-Spectrogram, the attention layers suffer a spatial collision. During conditional inference, higher-priority semantic tokens (e.g., vocal crescendos) overwrite or smudge the underlying instrumentals. 
*   **The Artifact:** External source separation reveals severe gain-drops (ducking) and hollow, flanger-like digital phase ruins in the instrumentals because their frequency data was destructively compressed during the 2D generation step.

---

## 3. Technical Specification: The 4D Lattice Space

Project XD-Audio eliminates the 2D canvas. The generative model trains and executes inference inside a rigid **4D Tensor Grid**:

$$\text{Tensor}_{\text{Lattice}} = f(X, Y, Z, \Phi)$$

```text
                  [Y] Frequency (Spectral Bands)
                   ^ 
                   |   / [Z] Stereo-Panorama (Spatial Vector: -1.0 to +1.0)
                   |  /
                   | /
                   |/
                   +-----------------------> [X] Continuous Time
                  /
                 v
               [Φ] Phase Angle & Delays (Acoustic Depth / Hysteresis)
```

### 3.1 Dimension Definitions
1.  **Time ($X$):** The continuous temporal axis mapped via discrete hop sizes.
2.  **Frequency ($Y$):** The spectral height derived from a differentiable DSP front-end (FFT/Wavelet).
3.  **Stereo-Panorama ($Z$):** The interaural intensity and panning vector running from $-1.0$ (Hard Left) through $0.0$ (Center) to $+1.0$ (Hard Right).
4.  **Phase & Hysteresis ($\Phi$):** The physical wave phase angles and micro-time delays (early reflections, echoes) that synthesize psychoacoustic depth.

### 3.2 Hard Physical Constraints vs. Projected Textures
The network enforces strict separation between **Physics** and **Timbre**:
*   **The Physics Grid (Immutable):** The axes ($X, Y, Z, \Phi$) are mathematical boundaries. A data cluster at coordinate $Z = -1.0$ (Hard Left) can *never* overwrite or dilute energy from a cluster at $Z = 0.0$ (Center), regardless of shared frequency ($Y$) or time ($X$).
*   **The Projected Textures (Learned):** Cross-attention layers project learned characteristics (sound color, timbre, vocal traits, genres) *into* this 4D landscape. The model learns implicitly how a high-gain guitar wall or a deep baritone voice arranges its energy inside the 4D grid, without mixing their underlying physical foundations.

---

## 4. Scalability & VRAM Optimization Strategy

To bypass the quadratic memory complexity `O(N²)` of long monolith transformations, inference scales linearly `O(N)` using rolling **10-Second 4D Chunk Segments**:

*   **Grid Specs:** $1,722 \text{ frames (Time)} \times 1,025 \text{ bins (Freq)} \times 32 \text{ slices (Pan)} \times 16 \text{ bins (Phase)}$.
*   **VRAM Consumption:** Fixed at **~1.8 GB VRAM** in Float16 execution, allowing local inference on consumer-grade GPU hardware.
*   **Phase-Locked Seams:** Temporal boundaries (trailing $\Phi$-states and $Z$-states) of Chunk $N$ are fed as strict initialization constraints into Chunk $N+1$. This locks the sinewaves across cuts, completely eliminating temporal wobbling or transient shifting.

---

## 5. Implementation Roadmap (PoC Verification)

We invite the community to collaborate on the initial Proof of Concept (PoC) pipeline:

1.  **Differentiable DSP Encoder:** A Python-native processing layer (`scipy` / `librosa` wrappers) to decompose stereo WAV records into the 4D NumPy/PyTorch Tensor array via cross-channel phase and intensity deltas.
2.  **4D Diffusion Backbone:** A modified neural network executing diffusion iterations natively across the 4D tensor structure, guided by an lightweight temporal text-encoder.
3.  **Inversion Decoder:** An Inverse-STFT framework to map the spatial tensor directly back into phase-coherent PCM audio samples, exporting flawless stereo wave data.

---

## 6. The Inverse Pathway: Native 4D Audio Understanding & Deterministic Tagging

To solve the "Semantic Guesswork Dilemma" inherent in legacy 2D audio taggers, Project XD-Audio introduces a **Dual-Purpose 4D Encoder Architecture**. Because the 4D Lattice Grid ($X, Y, Z, \Phi$) mathematically segregates overlapping timbres by their spatial coordinates and phase alignments, we can extract flawless, deterministic feature tokens directly from the structure without relying on probabilistic 2D guesswork.

### 6.1 Spatial-Acoustic Attention Masking
Instead of feeding a flat image to an attention mechanism, the XD-Audio Understanding Model employs **Spatio-Temporal Attention Blocks**. 

- **Phase/Pan Gating:** The encoder isolates specific slices along the $Z$ (Stereo-Panorama) and $\Phi$ (Phase) axes. 
- **Example:** Double-tracked guitars sitting at $Z = -0.9$ and $Z = +0.9$ are processed in entirely separate neural pathways than a centered vocal track at $Z = 0.0$.
- **Result:** The model extracts the acoustic fingerprint of the guitar *independently* of the vocal or synth, preventing timbre-bleeding and mislabeling.

### 6.2 Deterministic Multi-Layer Feature Extraction
Instead of predicting generic text labels (e.g., "rock music"), the 4D Encoder maps the lattice directly to a structured, objective latent vector matrix:

1. **Physical Tokenizer (DSP-Driven):** Extracts absolute mathematical constants directly from the 4D grid coordinates:
   - **BPM / Micro-Timing:** Calculated deterministically via temporal $X$-axis energy spikes.
   - **Stereo Width / Center Mass:** Extracted directly from the $Z$-axis density distribution.
   - **Acoustic Environment / Reverb Depth:** Extracted from $\Phi$-axis hysteresis decay.

2. **Timbral Text Tokenizer (Cross-Modal Alignment):** A contrastive learning framework (similar to CLAP, but trained on 4D Latent Cubes) aligns the isolated spatial clusters with precise text tokens. Because the input features are physically clean (unmasked), a guitar is always labeled a guitar, and a synth is always labeled a synth.

---

## 7. Request for Comments (RFC)

We are actively seeking feedback from Digital Signal Processing (DSP) engineers and Machine Learning researchers on the following areas:
*   Optimal bin sizing for the continuous Phase ($\Phi$) axis to prevent quantization noise without exceeding the 1.8 GB VRAM buffer.
*   Cross-attention layer routing efficiency when mapping 1D text tokens to a 4D spatio-temporal lattice grid.

Please open an Issue or submit a Pull Request to discuss implementations.
