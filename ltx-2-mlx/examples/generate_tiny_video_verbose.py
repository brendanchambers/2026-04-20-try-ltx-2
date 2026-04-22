"""Generate a tiny test video with full token stream visualization using rich."""

import random
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

# Configuration
MODEL_NAME = "dgrauet/ltx-2.3-mlx-q4"
GEMMA_MODEL = "mlx-community/gemma-3-12b-it-4bit"
PROMPT = "A simple colorful abstract pattern"
OUTPUT_FILE = "../../data/generated_video/tiny_test_verbose.mp4"
FRAMES = 3
HEIGHT = 64
WIDTH = 64
SEED = -1

console = Console()


def visualize_tokens(name: str, tokens: mx.array, details: dict[str, Any] = None):
    """Visualize token array with rich formatting."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold yellow]{name}[/bold yellow]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # Shape info
    shape_str = " × ".join(map(str, tokens.shape))
    console.print(f"Shape: [green]{shape_str}[/green]")
    console.print(f"Dtype: [green]{tokens.dtype}[/green]")

    # Statistics
    if tokens.size > 0:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")

        try:
            stats_table.add_row("Mean", f"{float(mx.mean(tokens)):.6f}")
            stats_table.add_row("Std", f"{float(mx.std(tokens)):.6f}")
            stats_table.add_row("Min", f"{float(mx.min(tokens)):.6f}")
            stats_table.add_row("Max", f"{float(mx.max(tokens)):.6f}")
        except Exception as e:
            stats_table.add_row("Error", str(e))

        console.print(stats_table)

    # Additional details
    if details:
        console.print("\n[bold]Additional Info:[/bold]")
        for key, value in details.items():
            console.print(f"  {key}: [yellow]{value}[/yellow]")


def visualize_token_ids(token_ids: mx.array, attention_mask: mx.array):
    """Visualize text tokenization."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold yellow]TEXT TOKENIZATION[/bold yellow]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # Token IDs shape
    console.print(f"Token IDs shape: [green]{token_ids.shape}[/green]")
    console.print(f"Attention mask shape: [green]{attention_mask.shape}[/green]")

    # Valid token count
    valid_tokens = int(mx.sum(attention_mask))
    padding_tokens = attention_mask.size - valid_tokens
    console.print(f"Valid tokens: [green]{valid_tokens}[/green]")
    console.print(f"Padding tokens: [yellow]{padding_tokens}[/yellow]")

    # Show first and last 20 token IDs
    token_ids_flat = token_ids.reshape(-1)
    first_20 = [int(x) for x in token_ids_flat[:20]]
    last_20 = [int(x) for x in token_ids_flat[-20:]]

    console.print("\n[bold]First 20 token IDs:[/bold]")
    console.print(f"[cyan]{first_20}[/cyan]")

    console.print("\n[bold]Last 20 token IDs:[/bold]")
    console.print(f"[cyan]{last_20}[/cyan]")


def visualize_denoising_step(step: int, total_steps: int, sigma: float,
                              video_x0: mx.array, audio_x0: mx.array,
                              video_x: mx.array = None):
    """Visualize a single denoising step."""
    if step == 0:
        # Create header
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print("[bold yellow]DENOISING LOOP[/bold yellow]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # Step info
    console.print(f"\n[bold magenta]Step {step}/{total_steps}[/bold magenta] (σ = {sigma:.4f})")

    # Create table for this step
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Tensor", style="yellow")
    table.add_column("Shape", style="green")
    table.add_column("Min", style="blue")
    table.add_column("Max", style="blue")
    table.add_column("Mean", style="blue")

    def add_tensor_row(name: str, tensor: mx.array):
        shape_str = " × ".join(map(str, tensor.shape))
        table.add_row(
            name,
            shape_str,
            f"{float(mx.min(tensor)):.4f}",
            f"{float(mx.max(tensor)):.4f}",
            f"{float(mx.mean(tensor)):.4f}"
        )

    add_tensor_row("video_x0", video_x0)
    add_tensor_row("audio_x0", audio_x0)
    if video_x is not None:
        add_tensor_row("video_x_next", video_x)

    console.print(table)


class VerbosePipeline(TextToVideoPipeline):
    """Pipeline with token visualization hooks."""

    def _encode_text(self, prompt: str):
        """Override text encoding to add visualization."""
        console.print("\n[bold green]→ Encoding text prompt...[/bold green]")

        # First, tokenize to get discrete token IDs
        token_ids, attention_mask = self.text_encoder.tokenize(prompt)

        # Visualize token IDs
        visualize_token_ids(token_ids, attention_mask)

        # Call original method to get embeddings
        result = super()._encode_text(prompt)
        video_embeds, audio_embeds = result

        # Visualize embeddings
        visualize_tokens(
            "TEXT EMBEDDINGS - Video",
            video_embeds,
            {"Context length": video_embeds.shape[1], "Embedding dim": video_embeds.shape[2]}
        )

        visualize_tokens(
            "TEXT EMBEDDINGS - Audio",
            audio_embeds,
            {"Context length": audio_embeds.shape[1], "Embedding dim": audio_embeds.shape[2]}
        )

        return result

    def generate(self, *args, **kwargs):
        """Override generate to add token visualization."""
        console.print("\n[bold green]→ Starting generation...[/bold green]")

        # Get original generate method
        import inspect
        from ltx_core_mlx.conditioning.types.latent_cond import create_initial_state
        from ltx_pipelines_mlx.utils.samplers import denoise_loop

        # Call parent's generate but intercept key steps
        # We'll need to replicate some logic here to insert hooks

        # For now, call the original and visualize the result
        video_latent, audio_latent = super().generate(*args, **kwargs)

        # Visualize final output
        visualize_tokens(
            "FINAL VIDEO LATENT OUTPUT",
            video_latent,
            {
                "Channels": video_latent.shape[1],
                "Frames": video_latent.shape[2],
                "Height": video_latent.shape[3],
                "Width": video_latent.shape[4],
            }
        )

        visualize_tokens(
            "FINAL AUDIO LATENT OUTPUT",
            audio_latent,
            {
                "Channels": audio_latent.shape[1],
                "Time steps": audio_latent.shape[2],
                "Features": audio_latent.shape[3],
            }
        )

        return video_latent, audio_latent


def main():
    console.print(Panel.fit(
        "[bold cyan]Tiny Video Generation with Token Visualization[/bold cyan]\n\n"
        f"Resolution: {WIDTH}×{HEIGHT}\n"
        f"Frames: {FRAMES}\n"
        f"Model: {MODEL_NAME}\n"
        f"Prompt: {PROMPT}",
        border_style="cyan"
    ))

    # Ensure output directory exists
    output_path = Path(__file__).parent / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve seed
    seed = SEED
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    console.print(f"\n[bold]Seed:[/bold] {seed}")

    console.print("\n[bold green]→ Loading model...[/bold green]")
    t0 = time.time()

    try:
        # Create verbose pipeline
        pipe = VerbosePipeline(
            model_dir=MODEL_NAME,
            gemma_model_id=GEMMA_MODEL,
        )

        # Monkey-patch the denoising loop to add visualization
        from ltx_pipelines_mlx.utils.samplers import denoise_loop as original_denoise

        step_counter = [0]  # Use list to allow modification in closure

        def verbose_denoise_loop(model, video_state, audio_state, *args, **kwargs):
            """Wrapper around denoise_loop to visualize each step."""
            console.print("\n[bold green]→ Starting denoising loop...[/bold green]")

            # Get sigmas to know step count
            sigmas = kwargs.get('sigmas', None)
            if sigmas is None:
                from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
                sigmas = DISTILLED_SIGMAS

            total_steps = len(sigmas) - 1

            # Visualize initial latent states
            visualize_tokens(
                "VIDEO LATENT (Initial Noise)",
                video_state.latent,
                {"Tokens": video_state.latent.shape[1], "Channels": video_state.latent.shape[2]}
            )

            visualize_tokens(
                "AUDIO LATENT (Initial Noise)",
                audio_state.latent,
                {"Tokens": audio_state.latent.shape[1], "Channels": audio_state.latent.shape[2]}
            )

            # Wrap the model to intercept predictions
            original_model_call = model.__call__

            def verbose_model_call(*model_args, **model_kwargs):
                result = original_model_call(*model_args, **model_kwargs)

                # Visualize this step
                sigma = float(model_kwargs.get('sigma', model_args[2] if len(model_args) > 2 else 0))
                video_x0, audio_x0 = result

                visualize_denoising_step(
                    step_counter[0],
                    total_steps,
                    sigma,
                    video_x0,
                    audio_x0
                )

                step_counter[0] += 1
                return result

            model.__call__ = verbose_model_call

            # Call original denoise loop
            result = original_denoise(model, video_state, audio_state, *args, **kwargs)

            # Restore original
            model.__call__ = original_model_call

            return result

        # Monkey-patch for this session
        import ltx_pipelines_mlx.utils.samplers as samplers_module
        samplers_module.denoise_loop = verbose_denoise_loop

        # Generate
        console.print("\n[bold green]→ Generating video...[/bold green]")
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

        console.print(Panel.fit(
            f"[bold green]✓ Success![/bold green]\n\n"
            f"Saved to: {output_path}\n"
            f"Time: {elapsed:.1f}s",
            border_style="green"
        ))

        return 0

    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]✗ Error:[/bold red]\n\n{e}",
            border_style="red"
        ))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
