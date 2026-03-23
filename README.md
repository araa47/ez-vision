# ez-image

Image generation and editing skills for AI agents.

Currently supports Google Imagen (Gemini) via Vertex AI or API key. More providers coming soon.

## Skills

| Skill | Description |
|-------|-------------|
| **ez-google-imagen** | Image generation and editing using Google Gemini and Imagen models. Supports text-to-image, image-to-image editing, and 1K/2K/4K resolution output. |

## Supported Models

| Model | Type | Editing | Notes |
|-------|------|---------|-------|
| `gemini-3-pro-image-preview` | Gemini (generate + edit) | Yes | Default. High quality generation and editing. |
| `imagen-4.0-generate-001` | Imagen 4 (generate only) | No | Best image quality. Generation only. |

All models support Vertex AI ADC and Gemini API key authentication.

Future releases will add support for additional image generation providers and models.

## Installation

```bash
npx skills add araa47/ez-image
```

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Google Cloud ADC (`gcloud auth application-default login`) **or** a Gemini API key

## Quick Start

```bash
# Generate an image (Vertex AI ADC)
uv run scripts/generate_image.py --prompt "A serene Japanese garden" --filename "garden.png"

# Generate with API key
uv run scripts/generate_image.py --prompt "A mountain landscape" --filename "mountain.png" --api-key "YOUR_KEY"

# Edit an existing image
uv run scripts/generate_image.py --prompt "make the sky dramatic" --filename "edited.png" --input-image "original.png"

# Use Imagen 4 model
uv run scripts/generate_image.py --prompt "A sunset" --filename "sunset.png" --model imagen-4.0-generate-001

# High resolution output
uv run scripts/generate_image.py --prompt "A sunset" --filename "sunset.png" --resolution 4K
```

See each skill's `SKILL.md` for full usage details.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
