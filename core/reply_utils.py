from __future__ import annotations

from core.models import ConversationState


def should_show_quick_replies(*, state: ConversationState, has_open_task: bool) -> bool:
    if state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
        return False
    return has_open_task or state in {ConversationState.NORMAL, ConversationState.INTENSE}


def split_reply_text(text: str, *, max_len: int = 380) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return [""]
    if len(cleaned) <= max_len:
        return [cleaned]

    paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
    if not paragraphs:
        return [cleaned[:max_len]]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_len:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_len:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start : start + max_len])
            start += max_len
        current = ""

    if current:
        chunks.append(current)
    return chunks or [cleaned[:max_len]]