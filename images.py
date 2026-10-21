"""
Image input: recognize image files, normalize them into the form the model accepts, and
read images from the system clipboard.
"""
import asyncio
import io
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic_ai import BinaryContent

from core.paths import get_session_dir

# Images are recognized by extension; the content is validated when the file is read
IMAGE_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Upper bounds applied before an image reaches the model: large payloads cost tokens and
# can be rejected by the API, so anything above these is downscaled or refused
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2000
JPEG_QUALITIES = (85, 70, 55)

_IMAGE_PLACEHOLDER = re.compile(r"\[Image #(\d+)\]")


class ImageError(Exception):
    """
    An image could not be used. The message is safe to show to the user or the model.
    """


def is_image(path: str) -> bool:
    """
    Whether this path looks like an image we can send to the model.
    """
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def load_image(path: str) -> BinaryContent:
    """
    Read an image file as BinaryContent, the form the model layer accepts.

    The file is validated with Pillow (so a mislabelled extension cannot slip through) and
    downscaled or re-encoded when it exceeds MAX_IMAGE_BYTES or MAX_IMAGE_DIMENSION.

    Raises:
        ImageError: the file cannot be read, is not a real image, or stays too large.
    """
    media_type = IMAGE_EXTENSIONS[os.path.splitext(path)[1].lower()]
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ImageError(f"Cannot read image {path}: {e}") from e
    return _normalize(path, data, media_type)


def _normalize(name: str, data: bytes, media_type: str) -> BinaryContent:
    """
    Return the image unchanged when it is already small enough, otherwise downscale it.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            too_big = max(img.size) > MAX_IMAGE_DIMENSION
            if not too_big and len(data) <= MAX_IMAGE_BYTES:
                return BinaryContent(data=data, media_type=media_type)
            return _shrink(name, img)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise ImageError(f"{name} is not a readable image: {e}") from e


def _shrink(name: str, img: Image.Image) -> BinaryContent:
    """
    Downscale to MAX_IMAGE_DIMENSION and re-encode, lowering quality until it fits.
    """
    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
    # PNG preserves transparency, but a photo re-encoded as PNG is needlessly large
    keep_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
    if keep_alpha:
        buffer = io.BytesIO()
        img.convert("RGBA").save(buffer, format="PNG", optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            return BinaryContent(data=encoded, media_type="image/png")

    rgb = img.convert("RGB")
    for quality in JPEG_QUALITIES:
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            return BinaryContent(data=encoded, media_type="image/jpeg")

    raise ImageError(
        f"{name} is still larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB after downscaling"
    )


def build_user_content(
    text: str, attachments: list[BinaryContent]
) -> str | list[str | BinaryContent]:
    """
    Replace each [Image #N] placeholder with its image, keeping the position inside the text.

    A placeholder repeated in the text only contributes its image once; any attachment
    without a placeholder is appended in order at the end.
    """
    if not attachments:
        return text

    content: list[str | BinaryContent] = []
    cursor = 0
    inserted = set()

    for match in _IMAGE_PLACEHOLDER.finditer(text):
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(attachments):
            continue
        before = text[cursor:match.start()]
        if before:
            content.append(before)
        if index not in inserted:
            content.append(attachments[index])
            inserted.add(index)
        cursor = match.end()

    after = text[cursor:]
    if after:
        content.append(after)

    content.extend(image for index, image in enumerate(attachments) if index not in inserted)
    return content


def describe(content):
    """
    Summarize an image for display; any other content is returned unchanged.

    Printing raw image bytes would only fill the terminal and the log with noise.
    """
    if isinstance(content, BinaryContent):
        return f"[{content.media_type}, {len(content.data) / 1024:.0f} KB]"
    return content


# Per-platform command that reports whether the clipboard holds an image
_CHECK_COMMANDS = {
    "Darwin": ["osascript", "-e", "clipboard info for «class PNGf»"],
    "Linux": ["sh", "-c", "xclip -selection clipboard -t TARGETS -o | grep -q image/"],
    "Windows": ["powershell", "-NoProfile", "-Command", "(Get-Clipboard -Format Image) -ne $null"],
}


def _save_command(path: str) -> list[str]:
    """
    Per-platform command that writes the clipboard image to `path`.
    """
    return {
        "Darwin": [
            "osascript",
            "-e", "set png_data to (the clipboard as «class PNGf»)",
            "-e", f'set fp to open for access POSIX file "{path}" with write permission',
            "-e", "write png_data to fp",
            "-e", "close access fp",
        ],
        "Linux": ["sh", "-c", f'xclip -selection clipboard -t image/png -o > "{path}"'],
        "Windows": [
            "powershell", "-NoProfile", "-Command",
            f"$img = Get-Clipboard -Format Image; if ($img) {{ $img.Save('{path}') }}",
        ],
    }[platform.system()]


def _clipboard_dir(session_id: str) -> Path:
    # Keep pasted images inside the session bundle so they are cleaned up with the session
    return get_session_dir(session_id) / "images"


def read_clipboard_image(session_id: str) -> BinaryContent | None:
    """
    Return the image currently on the system clipboard, or None when it holds no image.

    The terminal only ever hands the program text, so the image is saved to a file by the
    platform clipboard tool and read back from there.

    Raises:
        ImageError: the clipboard held an image that could not be used.
    """
    system = platform.system()
    check = _CHECK_COMMANDS.get(system)
    if check is None:
        return None

    try:
        probe = subprocess.run(check, capture_output=True, timeout=5)
        if probe.returncode != 0:
            return None
        # macOS and Windows exit 0 even for an empty clipboard, so the output decides
        if system == "Darwin" and not probe.stdout.strip():
            return None
        if system == "Windows" and b"True" not in probe.stdout:
            return None

        # Create the directory only once an image is really there, so a plain paste with
        # no image leaves no empty directory behind
        target_dir = _clipboard_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.png"
        saved = subprocess.run(_save_command(str(path)), capture_output=True, timeout=10)
        # The save command reports failure through its exit code rather than an exception
        if saved.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None

    if not path.is_file() or path.stat().st_size == 0:
        return None
    return load_image(str(path))


async def read_clipboard_image_async(session_id: str) -> BinaryContent | None:
    """
    Off-loop wrapper for read_clipboard_image: the clipboard tools are blocking processes
    and would otherwise freeze the REPL for seconds.
    """
    return await asyncio.to_thread(read_clipboard_image, session_id)
