# LTX-2 Video Generation Experiments

Journal project for testing the LTX-2 video generation model on Apple Silicon (MacBook).

## Prerequisites

This project uses the [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) tool for video generation on Apple Silicon.

### Setup

Run the setup script to clone the required tools:

```bash
./setup.sh
```

This will clone the `ltx-2-mlx` repository at the tested commit.

### Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python environment with `uv` installed
- Sufficient RAM based on model choice:
  - bf16 models: 64GB+ RAM
  - int8 models: 32GB+ RAM
  - int4 models: 16GB+ RAM

## Usage

### Generate sample video

```bash
uv run ltx-2-mlx generate --prompt "A cat walking in a garden" --output test.mp4 --model dgrauet/ltx-2.3-mlx-q4 --frames 9 --height 480 --width 704
```

## Resources

- [ltx-2-mlx Repository](https://github.com/dgrauet/ltx-2-mlx)
- [Original LTX-2 Model](https://github.com/Lightricks/LTX-2)