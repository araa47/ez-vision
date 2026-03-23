import subprocess

import pytest

pytestmark = pytest.mark.e2e


def _run_generate(args):
    result = subprocess.run(
        ["uv", "run", "skills/ez-google-imagen/scripts/generate_image.py", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result


class TestGeminiGeneration:
    def test_generate_image(self, tmp_path):
        out = str(tmp_path / "test-gen.png")
        result = _run_generate(
            [
                "--prompt",
                "A simple red circle on a white background",
                "--filename",
                out,
                "--resolution",
                "1K",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Image saved" in result.stdout
        from pathlib import Path

        assert Path(out).exists()
        assert Path(out).stat().st_size > 100


class TestGeminiEditing:
    def test_edit_image(self, tmp_path):
        from PIL import Image as PILImage

        input_img = tmp_path / "input.png"
        PILImage.new("RGB", (256, 256), (0, 128, 255)).save(str(input_img))

        out = str(tmp_path / "test-edit.png")
        result = _run_generate(
            [
                "--prompt",
                "Add a small yellow star in the center",
                "--filename",
                out,
                "--input-image",
                str(input_img),
                "--resolution",
                "1K",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Image saved" in result.stdout
        from pathlib import Path

        assert Path(out).exists()
        assert Path(out).stat().st_size > 100
