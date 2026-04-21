"""Generate a tiny test video using ltx-2-mlx."""

import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

# Configuration
MODEL_NAME = "dgrauet/ltx-2.3-mlx-q4"
PROMPT = "A simple colorful abstract pattern"
OUTPUT_FILE = "../data/generated_video/tiny_test.mp4"
FRAMES = 10
HEIGHT = 64
WIDTH = 64


def main():
    console = Console()

    console.print(f"[bold cyan]Generating tiny test video[/bold cyan]")
    console.print(f"Resolution: {WIDTH}x{HEIGHT}")
    console.print(f"Frames: {FRAMES}")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Prompt: {PROMPT}")
    console.print()

    # Ensure output directory exists
    project_root = Path(__file__).parent
    output_dir = project_root / "data" / "generated_video"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[yellow]Generating video (this may take a minute)...[/yellow]")
    t0 = time.time()

    # Call ltx-2-mlx generate using subprocess
    ltx_dir = project_root / "ltx-2-mlx"

    result = subprocess.run(
        [
            "uv", "run", "ltx-2-mlx", "generate",
            "--prompt", PROMPT,
            "--output", OUTPUT_FILE,
            "--model", MODEL_NAME,
            "--frames", str(FRAMES),
            "--height", str(HEIGHT),
            "--width", str(WIDTH),
        ],
        cwd=ltx_dir,
        capture_output=False,
    )

    elapsed = time.time() - t0

    if result.returncode == 0:
        console.print(f"\n[bold green]Success![/bold green]")
        console.print(f"Saved to: [green]{output_dir / 'tiny_test.mp4'}[/green]")
        console.print(f"Time: {elapsed:.1f}s")
        return 0
    else:
        console.print(f"\n[bold red]Generation failed with exit code {result.returncode}[/bold red]")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
