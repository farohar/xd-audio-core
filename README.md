# XD-Audio Core

The native, high-performance reference implementation of the 4D Rigid Auditory Lattice Space framework. 

This repository contains the core digital signal processing (DSP) engines and tensor projection matrices optimized for phase-coherent audio synthesis.

## Architectural Specification
The full mathematical and theoretical foundation for this framework is formally documented as **RFC-0001** in the central specification archive:
👉 **[Read the Full Specification (RFC-0001)](https://github.com/farohar/farohar-rfcs/blob/main/RFC-0001/README.md)**

## Core Features Implemented Here
* **4D Tensor Grid Processing:** Low-level operations mapping $X, Y, Z,$ and $\Phi$ coordinate axes natively.
* **Differentiable DSP Encoder:** Phase angle and delay tracking matrices.
* **Spatial-Acoustic Attention Blocks:** Memory-optimized CUDA/C++ operations.

## Installation & Setup

Ensure you have [uv](https://github.com) installed. Then, clone the repository and install the dependencies natively:

```bash
git clone https://github.com/farohar/xd-audio-core.git
cd xd-audio-core

# Create a virtual environment and install all dependencies from pyproject.toml
uv pip install -e .
```

## License
This reference implementation is distributed under the MIT License.
