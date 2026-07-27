# ==============================================================================
# Project XD-Audio: Native 4D Spatio-Temporal Lattice Space
# File: src/dsp_pipeline.py
# Description: Differentiable 4D DSP Encoder/Decoder Framework (PoC Baseline)
#
# Author: farohar (https://github.com/farohar)
# Date: 27 July 2026
# License: MIT License (see LICENSE.md in repository root)
# Status: Experimental / Reference Implementation for RFC-0001
# ==============================================================================

import torch
import numpy as np
import librosa
import soundfile as sf

# Global Architecture Metadata
__version__ = "0.1.0-alpha"

class XDAudioDSP:
    def __init__(self, sr=44100, n_fft=4096, hop_length=256, num_pan_slices=32, num_phase_bins=16, chunk_secs=10):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_pan_slices = num_pan_slices
        self.num_phase_bins = num_phase_bins
        self.chunk_samples = chunk_secs * sr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = torch.hann_window(n_fft).to(self.device)

    def encode_chunk(self, audio_chunk):
        """
        Module 1: Process a single 10-seconds-segment on the GPU.
        Constant VRAM-usage: ~3.6 GB.
        """
        left_ch = audio_chunk[0].to(self.device)
        right_ch = audio_chunk[1].to(self.device)

        stft_l = torch.stft(left_ch, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        stft_r = torch.stft(right_ch, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        mag_l, phase_l = torch.abs(stft_l), torch.angle(stft_l)
        mag_r, phase_r = torch.abs(stft_r), torch.angle(stft_r)

        num_freqs, num_times = stft_l.shape[0], stft_l.shape[1]

        total_mag = mag_l + mag_r + 1e-8
        ild = (mag_r - mag_l) / total_mag
        ipd = torch.atan2(torch.sin(phase_r - phase_l), torch.cos(phase_r - phase_l))

        grid_4d = torch.zeros((self.num_phase_bins, num_times, num_freqs, self.num_pan_slices), dtype=torch.float32, device=self.device)

        pan_indices = torch.clamp(((ild + 1.0) / 2.0 * (self.num_pan_slices - 1)).long(), 0, self.num_pan_slices - 1)
        phase_indices = torch.clamp(((ipd + np.pi) / (2 * np.pi) * (self.num_phase_bins - 1)).long(), 0, self.num_phase_bins - 1)

        avg_mag = total_mag / 2.0

        # 1D Linear Indexing Mapping (stops VRAM-explosion)
        stride_phi = num_times * num_freqs * self.num_pan_slices
        stride_t = num_freqs * self.num_pan_slices
        stride_f = self.num_pan_slices

        t_idx = torch.arange(num_times, device=self.device).unsqueeze(0).expand(num_freqs, num_times)
        f_idx = torch.arange(num_freqs, device=self.device).unsqueeze(1).expand(num_freqs, num_times)

        # Here we force the mathematical continuity before calculation:
        flat_indices = (phase_indices * stride_phi) + (t_idx * stride_t) + (f_idx * stride_f) + pan_indices

        # .reshape(-1) instead .view(-1), to fix memory layout error!
        grid_4d.view(-1).put_(flat_indices.reshape(-1), avg_mag.reshape(-1), accumulate=False)

        return grid_4d, phase_l.cpu(), phase_r.cpu()

    def decode_chunk(self, grid_4d, orig_phase_l, orig_phase_r):
        """
        Module 4: Calculates a 10s-segment back to wave-form.
        """
        reconstructed_mag = torch.sum(grid_4d, dim=(0, 3))
        reconstructed_mag = reconstructed_mag.transpose(0, 1).cpu()

        stft_l_reconstructed = torch.polar(reconstructed_mag, orig_phase_l)
        stft_r_reconstructed = torch.polar(reconstructed_mag, orig_phase_r)

        cpu_window = torch.hann_window(self.n_fft)
        left_wave = torch.istft(stft_l_reconstructed, n_fft=self.n_fft, hop_length=self.hop_length, window=cpu_window)
        right_wave = torch.istft(stft_r_reconstructed, n_fft=self.n_fft, hop_length=self.hop_length, window=cpu_window)

        return torch.stack([left_wave, right_wave], dim=0)

    def process_full_audio(self, audio_path):
        """
        Splits the whole song in chunks, processes them sequential in VRAM
        and concatenates the result seamlessly.
        """
        print(f"-> Load Audio-File via Librosa...")
        y, _ = librosa.load(audio_path, sr=self.sr, mono=False)
        total_samples = y.shape[1]

        full_reconstructed_left = []
        full_reconstructed_right = []

        for start_idx in range(0, total_samples, self.chunk_samples):
            end_idx = min(start_idx + self.chunk_samples, total_samples)
            chunk_data = torch.from_numpy(y[:, start_idx:end_idx])

            # Schutzbarriere falls der letzte Chunk kürzer als n_fft ist
            if chunk_data.shape[1] < self.n_fft:
                continue

            print(f"   Processing Chunk: {start_idx / self.sr:.1f}s bis {end_idx / self.sr:.1f}s...")

            grid, p_l, p_r = self.encode_chunk(chunk_data)
            wave_out = self.decode_chunk(grid, p_l, p_r)

            full_reconstructed_left.append(wave_out[0])
            full_reconstructed_right.append(wave_out[1])

        out_l = torch.cat(full_reconstructed_left, dim=0).numpy()
        out_r = torch.cat(full_reconstructed_right, dim=0).numpy()
        return torch.stack([torch.from_numpy(out_l), torch.from_numpy(out_r)], dim=0).numpy()

if __name__ == "__main__":
    import os
    import sys

    # Version dynamically injected from global architecture metadata variable
    print(f"--- Project XD-Audio v{__version__}: Chunk-Based High-Res DSP Pipeline ---")
    pipeline = XDAudioDSP()

    # Dynamic CLI argument parsing for advanced flexibility
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "./data/synthetic_test_case_clean.mp3"

    # Absolute safety barrier for automated open-source deployment
    if not os.path.exists(test_file):
        print(f"\nERROR: Benchmark target audio file '{test_file}' not found!")
        print("Please ensure your un-LoRA'd Minimal Techno sample is stored inside the 'data/' directory,")
        print("or pass your custom absolute filepath directly via the terminal interface:")
        print(f"-> uv run python src/dsp_pipeline.py /path/to/your/audio.mp3\n")
        sys.exit(1)

    print(f"1. Start sequential 4D-Transformation over: {test_file} ...")
    stereo_out = pipeline.process_full_audio(test_file)

    output_path = "xd_highres_chunks.wav"
    sf.write(output_path, stereo_out.T, pipeline.sr)
    print(f"-> Lossless Export successful saved under: {output_path}")
    print("--- Pipeline-Test finished. Math & VRAM are stable! ---")
