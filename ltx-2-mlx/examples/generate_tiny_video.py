"""Generate a tiny test video using ltx-2-mlx Python API."""

import random
import sys
import time
from pathlib import Path

from rich.console import Console

from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

# Configuration
MODEL_NAME = "dgrauet/ltx-2.3-mlx-q4"
GEMMA_MODEL = "mlx-community/gemma-3-12b-it-4bit"
PROMPT = "A simple colorful abstract pattern"
OUTPUT_FILE = "../../data/generated_video/tiny_test.mp4"
FRAMES = 3
HEIGHT = 64
WIDTH = 64
SEED = -1


def main():
    console = Console()

    console.print("[bold cyan]Generating tiny test video[/bold cyan]")
    console.print(f"Resolution: {WIDTH}x{HEIGHT}")
    console.print(f"Frames: {FRAMES}")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Prompt: {PROMPT}")
    console.print()

    # Ensure output directory exists
    output_path = Path(__file__).parent / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve seed=-1 to a random value
    seed = SEED
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    console.print(f"Seed: {seed}")

    console.print("[yellow]Loading model...[/yellow]")
    t0 = time.time()

    try:
        # Create pipeline
        pipe = TextToVideoPipeline(
            model_dir=MODEL_NAME,
            gemma_model_id=GEMMA_MODEL,
        )

        # Generate and save
        console.print("[yellow]Generating video...[/yellow]")
        pipe.generate_and_save(
            prompt=PROMPT,
            output_path=str(output_path.absolute()),
            height=HEIGHT,
            width=WIDTH,
            num_frames=FRAMES,
            seed=seed,
            num_steps=None,
        )

        elapsed = time.time() - t0
        console.print(f"\n[bold green]Success![/bold green]")
        console.print(f"Saved to: [green]{output_path}[/green]")
        console.print(f"Time: {elapsed:.1f}s")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
