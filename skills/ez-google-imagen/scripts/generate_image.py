#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0.0",
#     "pillow>=10.0.0",
#     "google-auth>=2.0.0",
#     "httpx>=0.27.0",
#     "typer>=0.9.0",
# ]
# ///
"""Generate or edit images using Google Gemini and Imagen models via Vertex AI."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
DEFAULT_LOCATION = "global"

GEMINI_MODELS = ["gemini-3-pro-image-preview"]
IMAGEN_MODELS = ["imagen-4.0-generate-001"]
ALL_MODELS = GEMINI_MODELS + IMAGEN_MODELS
DEFAULT_MODEL = GEMINI_MODELS[0]

MODEL_ALIASES: dict[str, str] = {
    "nano-banana": "gemini-3-pro-image-preview",
    "gemini": "gemini-3-pro-image-preview",
    "imagen": "imagen-4.0-generate-001",
    "imagen4": "imagen-4.0-generate-001",
}

app = typer.Typer()


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def get_api_key(provided_key: str | None) -> str | None:
    if provided_key:
        return provided_key
    return os.environ.get("GEMINI_API_KEY")


def save_image(pil_image, output_path: Path):
    from PIL import Image as PILImage

    if pil_image.mode == "RGBA":
        rgb_image = PILImage.new("RGB", pil_image.size, (255, 255, 255))
        rgb_image.paste(pil_image, mask=pil_image.split()[3])
        rgb_image.save(str(output_path), "PNG")
    elif pil_image.mode == "RGB":
        pil_image.save(str(output_path), "PNG")
    else:
        pil_image.convert("RGB").save(str(output_path), "PNG")


async def generate_gemini(
    prompt: str,
    output_path: Path,
    input_image_path: str | None,
    resolution: str,
    model: str,
    api_key: str | None,
    project: str,
    location: str,
):
    from PIL import Image as PILImage

    if api_key:
        from google import genai

        typer.echo("Using Gemini API key authentication", err=True)
        client = genai.Client(api_key=api_key)
    else:
        try:
            import google.auth
            from google import genai
            from google.genai import types as genai_types

            _credentials, adc_project = google.auth.default()
            vertex_project = project
            if adc_project and not project:
                vertex_project = adc_project
            typer.echo(
                f"Using Vertex AI ADC (project: {vertex_project}, location: {location})",
                err=True,
            )
            client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=location,
                http_options=genai_types.HttpOptions(
                    api_version="v1beta1",
                    base_url="https://aiplatform.googleapis.com/",
                ),
            )
        except Exception as e:
            typer.echo(f"Error: Failed to initialize Vertex AI with ADC: {e}", err=True)
            typer.echo("Please either:", err=True)
            typer.echo("  1. Run: gcloud auth application-default login", err=True)
            typer.echo("  2. Provide --api-key argument", err=True)
            typer.echo("  3. Set GEMINI_API_KEY environment variable", err=True)
            sys.exit(1)

    from google.genai import types

    input_image = None
    output_resolution = resolution
    if input_image_path:
        try:
            input_image = PILImage.open(input_image_path)
            typer.echo(f"Loaded input image: {input_image_path}", err=True)
            if resolution == "1K":
                max_dim = max(input_image.size)
                if max_dim >= 3000:
                    output_resolution = "4K"
                elif max_dim >= 1500:
                    output_resolution = "2K"
                typer.echo(f"Auto-detected resolution: {output_resolution}", err=True)
        except Exception as e:
            typer.echo(f"Error loading input image: {e}", err=True)
            sys.exit(1)

    if input_image:
        typer.echo(f"Editing image with {model} at {output_resolution}...", err=True)
        gen_contents: list = [input_image, prompt]
    else:
        typer.echo(f"Generating image with {model} at {output_resolution}...", err=True)
        gen_contents = [prompt]

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=gen_contents,  # type: ignore[invalid-argument-type]
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(image_size=output_resolution),
            ),
        )

        image_saved = False
        for part in response.parts or []:
            if part.text is not None:
                typer.echo(f"Model response: {part.text}", err=True)
            elif part.inline_data is not None:
                import base64
                from io import BytesIO

                image_data = part.inline_data.data
                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)

                if image_data is None:
                    continue
                image = PILImage.open(BytesIO(image_data))
                save_image(image, output_path)
                image_saved = True

        if not image_saved:
            typer.echo("Error: No image was generated in the response.", err=True)
            sys.exit(1)

    except Exception as e:
        typer.echo(f"Error generating image: {e}", err=True)
        sys.exit(1)


async def generate_imagen(
    prompt: str,
    output_path: Path,
    project: str,
    location: str,
):
    import base64

    import google.auth
    import google.auth.transport.requests
    import httpx

    try:
        credentials, adc_project = google.auth.default()
        vertex_project = project
        if adc_project and not project:
            vertex_project = adc_project
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
    except Exception as e:
        typer.echo(f"Error: Failed to get ADC credentials: {e}", err=True)
        sys.exit(1)

    typer.echo(
        f"Using Vertex AI ADC (project: {vertex_project}, location: {location})",
        err=True,
    )
    typer.echo("Generating image with Imagen 4...", err=True)

    url = (
        f"https://aiplatform.googleapis.com/v1beta1/projects/{vertex_project}"
        f"/locations/{location}/publishers/google/models/"
        f"imagen-4.0-generate-001:predict"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {credentials.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "instances": [{"prompt": prompt}],
                    "parameters": {"sampleCount": 1},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            typer.echo(f"Error: Imagen 4 API failed: {resp.status_code}", err=True)
            typer.echo(f"Response: {resp.text}", err=True)
            sys.exit(1)

    predictions = resp.json().get("predictions", [])
    image_b64 = predictions[0].get("bytesBase64Encoded") if predictions else None
    if not image_b64:
        typer.echo("Error: No image data in response.", err=True)
        sys.exit(1)

    from io import BytesIO

    from PIL import Image as PILImage

    image = PILImage.open(BytesIO(base64.b64decode(image_b64)))
    save_image(image, output_path)


@app.command()
def main(
    prompt: Annotated[
        str,
        typer.Option(
            "-p", "--prompt", help="Image description or editing instructions"
        ),
    ],
    filename: Annotated[
        str,
        typer.Option("-f", "--filename", help="Output filename (e.g. sunset.png)"),
    ],
    model: Annotated[
        str, typer.Option("-m", "--model", help="Model to use")
    ] = DEFAULT_MODEL,
    input_image: Annotated[
        Optional[str],
        typer.Option(
            "-i",
            "--input-image",
            help="Input image path for editing (Gemini models only)",
        ),
    ] = None,
    resolution: Annotated[
        str,
        typer.Option("-r", "--resolution", help="Output resolution: 1K, 2K, or 4K"),
    ] = "1K",
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "-k", "--api-key", help="Gemini API key (overrides Vertex AI ADC)"
        ),
    ] = None,
    project: Annotated[
        str, typer.Option(help="Google Cloud project ID for Vertex AI (or set GOOGLE_CLOUD_PROJECT)")
    ] = DEFAULT_PROJECT,
    location: Annotated[
        str, typer.Option(help="Google Cloud region for Vertex AI")
    ] = DEFAULT_LOCATION,
):
    """Generate or edit images using Google Gemini and Imagen models."""
    model = resolve_model(model)
    if model not in ALL_MODELS:
        typer.echo(
            f"Error: Unknown model '{model}'. Choose from: {', '.join(ALL_MODELS)}",
            err=True,
        )
        typer.echo(
            f"Aliases: {', '.join(f'{k} -> {v}' for k, v in MODEL_ALIASES.items())}",
            err=True,
        )
        sys.exit(1)

    if resolution not in ("1K", "2K", "4K"):
        typer.echo(
            f"Error: Invalid resolution '{resolution}'. Choose from: 1K, 2K, 4K",
            err=True,
        )
        sys.exit(1)

    is_imagen = model in IMAGEN_MODELS

    if input_image and is_imagen:
        typer.echo(
            f"Error: Image editing is not supported with {model}. Use a Gemini model instead.",
            err=True,
        )
        sys.exit(1)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_imagen:
        asyncio.run(generate_imagen(prompt, output_path, project, location))
    else:
        asyncio.run(
            generate_gemini(
                prompt,
                output_path,
                input_image,
                resolution,
                model,
                get_api_key(api_key),
                project,
                location,
            )
        )

    typer.echo(f"\nImage saved: {output_path.resolve()}")


if __name__ == "__main__":
    app()
