"""Vision extraction module: image to structured JSON.

Takes image_bytes and a multimodal model (resolved from the 'vision' role) and
returns structured JSON with description, visible_text, and named entities.
The description and visible_text are concatenated into a synthetic text chunk
suitable for embedding downstream.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from worker.models.base import CompletionRequest, CompletionResponse
from worker.models.resolver import ModelRegistry, ResolvedModel
from worker.pipeline.chunker import Chunk

logger = logging.getLogger(__name__)

VISION_ROLE = "vision"

ENTITY_CONFIDENCE = 0.80
LLM_TIMEOUT = 120.0  # seconds
MAX_ENTITY_NAME_LEN = 512
MAX_DESCRIPTION_LEN = 5000
MAX_VISIBLE_TEXT_LEN = 10000

SYSTEM_PROMPT = """\
You are a vision analysis system. Given an image, extract:

1. **description**: A concise description of what the image shows.
2. **visible_text**: Any text visible in the image (OCR). Empty string if none.
3. **entities**: Named entities visible or referenced in the image. Each entity \
has a "name" and a "type" (one of: Person, Technology, Project, Organization, Concept).

Respond with a JSON object exactly matching this schema:
{
  "description": "...",
  "visible_text": "...",
  "entities": [{"name": "...", "type": "..."}]
}

Respond ONLY with the JSON object. No markdown fences, no commentary."""


@dataclass
class VisionResult:
    """Structured output from vision extraction."""

    description: str = ""
    visible_text: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)


def extract_image(
    image_bytes: bytes,
    model: ResolvedModel,
    mime_type: str = "image/png",
) -> VisionResult:
    """Extract structured information from an image using a multimodal model.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.
    model:
        A resolved multimodal model (from the 'vision' role).
    mime_type:
        MIME type of the image (e.g. "image/png", "image/jpeg").

    Returns
    -------
    VisionResult
        Extracted description, visible text, and entities.
        Returns empty result on unrecoverable failure.
    """
    b64 = base64.b64encode(image_bytes).decode("ascii")

    req = CompletionRequest(
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this image and extract the requested information as JSON.",
                    },
                ],
            }
        ],
        temperature=0.0,
        timeout=LLM_TIMEOUT,
    )

    try:
        resp = model.complete(req, task="vision")
        return _parse_response(resp)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Vision extraction failed: %s", exc)
        return VisionResult()


def extract_image_from_registry(
    image_bytes: bytes,
    registry: ModelRegistry,
    mime_type: str = "image/png",
) -> VisionResult:
    """Convenience wrapper that resolves the vision model from a registry.

    Parameters
    ----------
    image_bytes:
        Raw image bytes.
    registry:
        Model registry for resolving the 'vision' role.
    mime_type:
        MIME type of the image.

    Returns
    -------
    VisionResult
        Extracted description, visible text, and entities.
    """
    model = registry.for_role(VISION_ROLE)
    return extract_image(image_bytes, model, mime_type)


def vision_result_to_chunk(result: VisionResult) -> Chunk | None:
    """Concatenate description and visible_text into a synthetic text chunk.

    Returns None if both fields are empty (nothing to embed).
    """
    parts: list[str] = []
    if result.description:
        parts.append(result.description)
    if result.visible_text:
        parts.append(result.visible_text)

    if not parts:
        return None

    return Chunk(text="\n\n".join(parts), index=0)


def vision_result_to_entities(result: VisionResult) -> list[dict[str, Any]]:
    """Convert VisionResult entities to the standard entity dict format.

    Each entity gets a fixed confidence of ENTITY_CONFIDENCE (0.80).
    """
    return [
        {
            "name": ent["name"],
            "type": ent.get("type", "Concept"),
            "confidence": ENTITY_CONFIDENCE,
        }
        for ent in result.entities
        if "name" in ent
    ]


def _parse_response(resp: CompletionResponse) -> VisionResult:
    """Parse a model response into a VisionResult.

    Raises
    ------
    json.JSONDecodeError
        If the response text cannot be parsed as JSON.
    ValueError
        If the parsed JSON is not a dict or has invalid structure.
    """
    text = resp.text.strip() if resp.text else ""

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        text = "\n".join(lines).strip()

    if not text:
        return VisionResult()

    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    raw_desc = data.get("description", "")
    description = raw_desc if isinstance(raw_desc, str) else ""
    if not isinstance(raw_desc, str):
        logger.warning("Vision description is %s, expected str — discarding", type(raw_desc).__name__)
    description = description[:MAX_DESCRIPTION_LEN]

    raw_vt = data.get("visible_text", "")
    visible_text = raw_vt if isinstance(raw_vt, str) else ""
    if not isinstance(raw_vt, str):
        logger.warning("Vision visible_text is %s, expected str — discarding", type(raw_vt).__name__)
    visible_text = visible_text[:MAX_VISIBLE_TEXT_LEN]

    entities_raw = data.get("entities", [])
    if not isinstance(entities_raw, list):
        entities_raw = []

    entities: list[dict[str, Any]] = []
    for ent in entities_raw:
        if not isinstance(ent, dict) or "name" not in ent:
            continue
        name = ent["name"]
        if not isinstance(name, str) or not name.strip():
            logger.warning("Vision entity name is not a valid string, skipping")
            continue
        if len(name) > MAX_ENTITY_NAME_LEN:
            logger.warning(
                "Vision entity name too long (%d chars, limit %d), skipping",
                len(name), MAX_ENTITY_NAME_LEN,
            )
            continue
        entities.append({
            "name": name,
            "type": ent.get("type", "Concept") if isinstance(ent.get("type"), str) else "Concept",
        })

    return VisionResult(
        description=description,
        visible_text=visible_text,
        entities=entities,
    )
