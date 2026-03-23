import asyncio
import base64
from io import BytesIO
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from generate_image import (  # ty: ignore[unresolved-import]
    ALL_MODELS,
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    DEFAULT_PROJECT,
    GEMINI_MODELS,
    IMAGEN_MODELS,
    MODEL_ALIASES,
    app,
    generate_gemini,
    generate_imagen,
    get_api_key,
    resolve_model,
    save_image,
)
from typer.testing import CliRunner

runner = CliRunner()


class TestConstants:
    def test_gemini_models(self):
        assert GEMINI_MODELS == ["gemini-3-pro-image-preview"]

    def test_imagen_models(self):
        assert IMAGEN_MODELS == ["imagen-4.0-generate-001"]

    def test_all_models(self):
        assert ALL_MODELS == GEMINI_MODELS + IMAGEN_MODELS

    def test_default_model(self):
        assert DEFAULT_MODEL == GEMINI_MODELS[0]

    def test_default_project_from_env(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        from importlib import reload

        import generate_image  # ty: ignore[unresolved-import]

        reload(generate_image)
        assert generate_image.DEFAULT_PROJECT == ""

    def test_default_project_with_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-env-project")
        from importlib import reload

        import generate_image  # ty: ignore[unresolved-import]

        reload(generate_image)
        assert generate_image.DEFAULT_PROJECT == "my-env-project"

    def test_default_location(self):
        assert DEFAULT_LOCATION == "global"


class TestResolveModel:
    def test_alias_nano_banana(self):
        assert resolve_model("nano-banana") == "gemini-3-pro-image-preview"

    def test_alias_gemini(self):
        assert resolve_model("gemini") == "gemini-3-pro-image-preview"

    def test_alias_imagen(self):
        assert resolve_model("imagen") == "imagen-4.0-generate-001"

    def test_alias_imagen4(self):
        assert resolve_model("imagen4") == "imagen-4.0-generate-001"

    def test_full_model_name_passthrough(self):
        assert (
            resolve_model("gemini-3-pro-image-preview") == "gemini-3-pro-image-preview"
        )

    def test_unknown_name_passthrough(self):
        assert resolve_model("unknown-model") == "unknown-model"

    def test_all_aliases_resolve_to_known_models(self):
        for alias, model in MODEL_ALIASES.items():
            assert (
                model in ALL_MODELS
            ), f"Alias '{alias}' -> '{model}' not in ALL_MODELS"


class TestGetApiKey:
    def test_returns_provided_key(self):
        assert get_api_key("my-key") == "my-key"

    def test_returns_env_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        assert get_api_key(None) == "env-key"

    def test_returns_none_when_no_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert get_api_key(None) is None

    def test_provided_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        assert get_api_key("provided") == "provided"

    def test_empty_string_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        assert get_api_key("") == "env-key"


class TestSaveImage:
    def test_save_rgb_image(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (10, 10), (255, 0, 0))
        out = tmp_path / "test.png"
        save_image(img, out)
        assert out.exists()
        loaded = PILImage.open(out)
        assert loaded.mode == "RGB"

    def test_save_rgba_image(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("RGBA", (10, 10), (255, 0, 0, 128))
        out = tmp_path / "test.png"
        save_image(img, out)
        assert out.exists()
        loaded = PILImage.open(out)
        assert loaded.mode == "RGB"

    def test_save_l_mode_image(self, tmp_path):
        from PIL import Image as PILImage

        img = PILImage.new("L", (10, 10), 128)
        out = tmp_path / "test.png"
        save_image(img, out)
        assert out.exists()
        loaded = PILImage.open(out)
        assert loaded.mode == "RGB"


class TestMainCliValidation:
    def test_unknown_model_exits(self):
        result = runner.invoke(
            app,
            [
                "--prompt",
                "test",
                "--filename",
                "out.png",
                "--model",
                "bad-model",
            ],
        )
        assert result.exit_code != 0
        assert "Unknown model" in result.output

    def test_unknown_model_shows_aliases(self):
        result = runner.invoke(
            app,
            [
                "--prompt",
                "test",
                "--filename",
                "out.png",
                "--model",
                "bad-model",
            ],
        )
        assert "Aliases" in result.output
        assert "nano-banana" in result.output

    def test_alias_resolves_in_cli(self, tmp_path):
        output = tmp_path / "out.png"
        mock_response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with patch.dict("sys.modules", modules):
            result = runner.invoke(
                app,
                [
                    "--prompt",
                    "test",
                    "--filename",
                    str(output),
                    "--api-key",
                    "k",
                    "--model",
                    "nano-banana",
                ],
            )
        assert result.exit_code == 0
        assert "Image saved" in result.output

    def test_invalid_resolution_exits(self):
        result = runner.invoke(
            app,
            [
                "--prompt",
                "test",
                "--filename",
                "out.png",
                "--resolution",
                "8K",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid resolution" in result.output

    def test_imagen_with_input_image_exits(self):
        result = runner.invoke(
            app,
            [
                "--prompt",
                "test",
                "--filename",
                "out.png",
                "--model",
                "imagen-4.0-generate-001",
                "--input-image",
                "some.png",
            ],
        )
        assert result.exit_code != 0
        assert "not supported" in result.output


def _make_fake_png_bytes():
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (10, 10), (0, 128, 255))
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _make_mock_gemini_response(
    include_text=False, include_image=True, image_as_str=False, empty_data=False
):
    parts = []
    if include_text:
        text_part = MagicMock()
        text_part.text = "Here is your image"
        text_part.inline_data = None
        parts.append(text_part)
    if include_image:
        img_part = MagicMock()
        img_part.text = None
        inline = MagicMock()
        if empty_data:
            inline.data = None
        elif image_as_str:
            inline.data = base64.b64encode(_make_fake_png_bytes()).decode()
        else:
            inline.data = _make_fake_png_bytes()
        img_part.inline_data = inline
        parts.append(img_part)
    response = MagicMock()
    response.parts = parts
    return response


def _build_google_modules(mock_genai, mock_google_auth=None):
    google_mod = ModuleType("google")
    google_mod.genai = mock_genai  # type: ignore[attr-defined]
    modules: dict = {
        "google": google_mod,
        "google.genai": mock_genai,
        "google.genai.types": (
            mock_genai.types if hasattr(mock_genai, "types") else MagicMock()
        ),
    }
    if mock_google_auth is not None:
        google_mod.auth = mock_google_auth  # type: ignore[attr-defined]
        modules["google.auth"] = mock_google_auth
        transport_mod = MagicMock()
        modules["google.auth.transport"] = transport_mod
        modules["google.auth.transport.requests"] = transport_mod.requests
    return modules


class TestGenerateGemini:
    def _run(
        self,
        tmp_path,
        api_key="test-key",
        input_image_path=None,
        resolution="1K",
        response=None,
        mock_genai=None,
        mock_google_auth=None,
    ):
        output = tmp_path / "out.png"
        if response is None:
            response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)
        if mock_genai is None:
            mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        modules = _build_google_modules(mock_genai, mock_google_auth)

        with patch.dict("sys.modules", modules):
            asyncio.run(
                generate_gemini(
                    "a cat",
                    output,
                    input_image_path,
                    resolution,
                    "gemini-3-pro-image-preview",
                    api_key,
                    DEFAULT_PROJECT,
                    DEFAULT_LOCATION,
                )
            )
        return output, mock_genai, mock_client

    def test_generate_with_api_key(self, tmp_path):
        output, _, _ = self._run(tmp_path)
        assert output.exists()

    def test_generate_with_vertex_adc(self, tmp_path):
        mock_auth = MagicMock()
        mock_auth.default.return_value = (MagicMock(), "my-project")
        output, _, _ = self._run(tmp_path, api_key=None, mock_google_auth=mock_auth)
        assert output.exists()

    def test_vertex_adc_uses_adc_project_when_default(self, tmp_path):
        mock_auth = MagicMock()
        mock_auth.default.return_value = (MagicMock(), "adc-project")
        _, mock_genai, _ = self._run(tmp_path, api_key=None, mock_google_auth=mock_auth)
        call_kwargs = mock_genai.Client.call_args
        assert call_kwargs[1]["project"] == "adc-project"

    def test_vertex_adc_failure_exits(self, tmp_path):
        mock_auth = MagicMock()
        mock_auth.default.side_effect = Exception("no credentials")
        with pytest.raises(SystemExit):
            self._run(tmp_path, api_key=None, mock_google_auth=mock_auth)

    def test_generate_with_input_image(self, tmp_path):
        from PIL import Image as PILImage

        input_img = tmp_path / "input.png"
        PILImage.new("RGB", (100, 100)).save(str(input_img))
        output, _, _ = self._run(tmp_path, input_image_path=str(input_img))
        assert output.exists()

    def test_auto_resolution_large_image(self, tmp_path):
        from PIL import Image as PILImage

        input_img = tmp_path / "big.png"
        PILImage.new("RGB", (4000, 3000)).save(str(input_img))

        output = tmp_path / "out.png"
        mock_response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with (
            patch.dict("sys.modules", modules),
            patch("generate_image.typer") as mock_typer,
        ):
            asyncio.run(
                generate_gemini(
                    "enhance",
                    output,
                    str(input_img),
                    "1K",
                    "gemini-3-pro-image-preview",
                    "test-key",
                    DEFAULT_PROJECT,
                    DEFAULT_LOCATION,
                )
            )
        echo_calls = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("4K" in c for c in echo_calls)

    def test_auto_resolution_medium_image(self, tmp_path):
        from PIL import Image as PILImage

        input_img = tmp_path / "med.png"
        PILImage.new("RGB", (2000, 1500)).save(str(input_img))

        output = tmp_path / "out.png"
        mock_response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with (
            patch.dict("sys.modules", modules),
            patch("generate_image.typer") as mock_typer,
        ):
            asyncio.run(
                generate_gemini(
                    "enhance",
                    output,
                    str(input_img),
                    "1K",
                    "gemini-3-pro-image-preview",
                    "test-key",
                    DEFAULT_PROJECT,
                    DEFAULT_LOCATION,
                )
            )
        echo_calls = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("2K" in c for c in echo_calls)

    def test_bad_input_image_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, input_image_path="/nonexistent/bad.png")

    def test_response_with_text_and_image(self, tmp_path):
        resp = _make_mock_gemini_response(include_text=True, include_image=True)
        output, _, _ = self._run(tmp_path, response=resp)
        assert output.exists()

    def test_response_base64_string_image(self, tmp_path):
        resp = _make_mock_gemini_response(image_as_str=True)
        output, _, _ = self._run(tmp_path, response=resp)
        assert output.exists()

    def test_no_image_in_response_exits(self, tmp_path):
        resp = _make_mock_gemini_response(include_text=True, include_image=False)
        with pytest.raises(SystemExit):
            self._run(tmp_path, response=resp)

    def test_empty_parts_exits(self, tmp_path):
        resp = MagicMock()
        resp.parts = None
        with pytest.raises(SystemExit):
            self._run(tmp_path, response=resp)

    def test_inline_data_none_skipped(self, tmp_path):
        resp = _make_mock_gemini_response(empty_data=True)
        extra_part = MagicMock()
        extra_part.text = None
        extra_part.inline_data = MagicMock()
        extra_part.inline_data.data = _make_fake_png_bytes()
        resp.parts.append(extra_part)
        output, _, _ = self._run(tmp_path, response=resp)
        assert output.exists()

    def test_api_error_exits(self, tmp_path):
        output = tmp_path / "out.png"
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API error")
        )
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with patch.dict("sys.modules", modules), pytest.raises(SystemExit):
            asyncio.run(
                generate_gemini(
                    "a cat",
                    output,
                    None,
                    "1K",
                    "gemini-3-pro-image-preview",
                    "test-key",
                    DEFAULT_PROJECT,
                    DEFAULT_LOCATION,
                )
            )


def _build_imagen_mocks(response_json, auth_error=False, http_error=False):
    import httpx as real_httpx

    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_google_auth = MagicMock()

    if auth_error:
        mock_google_auth.default.side_effect = Exception("no creds")
    else:
        mock_google_auth.default.return_value = (mock_creds, "my-project")

    mock_resp = MagicMock()
    mock_resp.json.return_value = response_json
    if http_error:
        mock_resp.status_code = 403
        mock_resp.text = "forbidden"
        mock_resp.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_resp
        )
    else:
        mock_resp.raise_for_status = MagicMock()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.post = AsyncMock(return_value=mock_resp)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_http_client
    mock_httpx.HTTPStatusError = real_httpx.HTTPStatusError

    google_mod = ModuleType("google")
    google_mod.auth = mock_google_auth  # type: ignore[attr-defined]
    transport_mod = MagicMock()

    modules: dict = {
        "google": google_mod,
        "google.auth": mock_google_auth,
        "google.auth.transport": transport_mod,
        "google.auth.transport.requests": transport_mod.requests,
        "httpx": mock_httpx,
    }
    return modules, mock_http_client, mock_google_auth


class TestGenerateImagen:
    def _success_json(self):
        b64 = base64.b64encode(_make_fake_png_bytes()).decode()
        return {"predictions": [{"bytesBase64Encoded": b64}]}

    def _run(
        self,
        tmp_path,
        response_json=None,
        auth_error=False,
        http_error=False,
        mock_auth_override=None,
    ):
        output = tmp_path / "out.png"
        if response_json is None:
            response_json = self._success_json()
        modules, mock_http, mock_auth = _build_imagen_mocks(
            response_json, auth_error=auth_error, http_error=http_error
        )
        if mock_auth_override is not None:
            mock_auth = mock_auth_override
            modules["google.auth"] = mock_auth
            modules["google"].auth = mock_auth

        with patch.dict("sys.modules", modules):
            asyncio.run(
                generate_imagen("a sunset", output, DEFAULT_PROJECT, DEFAULT_LOCATION)
            )
        return output, mock_http, mock_auth

    def test_generate_success(self, tmp_path):
        output, _, _ = self._run(tmp_path)
        assert output.exists()

    def test_uses_adc_project_when_default(self, tmp_path):
        mock_auth = MagicMock()
        mock_auth.default.return_value = (MagicMock(token="t"), "adc-proj")
        _, mock_http, _ = self._run(tmp_path, mock_auth_override=mock_auth)
        url_arg = mock_http.post.call_args[0][0]
        assert "adc-proj" in url_arg

    def test_adc_failure_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, response_json={}, auth_error=True)

    def test_http_error_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, response_json={}, http_error=True)

    def test_no_predictions_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, response_json={"predictions": []})

    def test_no_image_data_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, response_json={"predictions": [{"other": "data"}]})


class TestMainCliIntegration:
    def test_gemini_generation_via_cli(self, tmp_path):
        output = tmp_path / "out.png"
        mock_response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with patch.dict("sys.modules", modules):
            result = runner.invoke(
                app,
                [
                    "--prompt",
                    "a dog",
                    "--filename",
                    str(output),
                    "--api-key",
                    "test-key",
                ],
            )
        assert result.exit_code == 0
        assert "Image saved" in result.output

    def test_imagen_generation_via_cli(self, tmp_path):
        output = tmp_path / "out.png"
        b64 = base64.b64encode(_make_fake_png_bytes()).decode()
        modules, mock_http, _ = _build_imagen_mocks(
            {"predictions": [{"bytesBase64Encoded": b64}]}
        )

        with patch.dict("sys.modules", modules):
            result = runner.invoke(
                app,
                [
                    "--prompt",
                    "a sunset",
                    "--filename",
                    str(output),
                    "--model",
                    "imagen-4.0-generate-001",
                ],
            )
        assert result.exit_code == 0
        assert "Image saved" in result.output

    def test_output_dir_created(self, tmp_path):
        output = tmp_path / "subdir" / "out.png"
        mock_response = _make_mock_gemini_response()
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        modules = _build_google_modules(mock_genai)

        with patch.dict("sys.modules", modules):
            result = runner.invoke(
                app,
                [
                    "--prompt",
                    "a dog",
                    "--filename",
                    str(output),
                    "--api-key",
                    "test-key",
                ],
            )
        assert result.exit_code == 0
        assert output.parent.exists()
