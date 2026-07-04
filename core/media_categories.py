from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class VideoCategoryIndex:
    folder_counts: dict[str, int] = field(default_factory=dict)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        videos_path: Path,
        video_paths: list[Path],
        aliases_csv: str = "",
    ) -> VideoCategoryIndex:
        aliases = cls._parse_aliases_csv(aliases_csv)
        counts: Counter[str] = Counter()

        for path in video_paths:
            try:
                relative = path.relative_to(videos_path)
            except ValueError:
                continue
            if len(relative.parts) < 2:
                continue
            counts[relative.parts[0].casefold()] += 1

        normalized_aliases: dict[str, tuple[str, ...]] = {}
        for folder, alias_values in aliases.items():
            normalized_aliases[folder.casefold()] = tuple(
                alias.strip().casefold() for alias in alias_values if alias.strip()
            )

        return cls(folder_counts=dict(counts), aliases=normalized_aliases)

    @staticmethod
    def _parse_aliases_csv(value: str) -> dict[str, tuple[str, ...]]:
        parsed: dict[str, tuple[str, ...]] = {}
        for chunk in value.split(";"):
            piece = chunk.strip()
            if not piece or "=" not in piece:
                continue
            folder, raw_aliases = piece.split("=", 1)
            folder_name = folder.strip()
            if not folder_name:
                continue
            aliases = tuple(alias.strip() for alias in raw_aliases.split(",") if alias.strip())
            if aliases:
                parsed[folder_name] = aliases
        return parsed

    def match_categories_in_text(self, text: str) -> list[str]:
        normalized = text.strip().casefold()
        if not normalized:
            return []

        matched: list[str] = []
        for folder, count in self.folder_counts.items():
            if count <= 0:
                continue
            folder_key = folder.casefold()
            if folder_key in normalized:
                matched.append(folder_key)
                continue
            for alias in self.aliases.get(folder_key, ()):
                if alias in normalized:
                    matched.append(folder_key)
                    break
        return matched

    def format_for_context(self) -> str:
        if not self.folder_counts:
            return "none"

        parts: list[str] = []
        for folder in sorted(self.folder_counts):
            count = self.folder_counts[folder]
            alias_labels = self.aliases.get(folder, ())
            if alias_labels:
                alias_text = "/".join(alias_labels)
                parts.append(f"{folder}({count})[{alias_text}]")
            else:
                parts.append(f"{folder}({count})")
        return ", ".join(parts)

    def primary_category_for_path(self, *, videos_path: Path, asset_path: Path) -> str | None:
        try:
            relative = asset_path.relative_to(videos_path)
        except ValueError:
            return None
        if len(relative.parts) < 2:
            return None

        top_folder = relative.parts[0].casefold()

        # 自定义规则（满足用户需求）：
        # - 子文件夹名字里出现“套路” → pov
        # - 保留标准的 pov / sm 文件夹名
        # - 其他所有子文件夹一律归为 sm
        if "套路" in top_folder:
            return "pov"
        if top_folder in ("pov", "sm"):
            return top_folder
        return "sm"