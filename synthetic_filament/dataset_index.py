from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

MASK_RE = re.compile(r"(?P<base>.+)_f(?P<frame>\d{4})_(?P<kind>mask(?:_\d+)?|cellmask(?:_\d+)?)$", re.IGNORECASE)


@dataclass
class FrameRecord:
    raw_path: Path
    raw_stem: str
    frame_index: int
    frame_tag: str
    filament_mask_path: Path | None = None
    cell_mask_path: Path | None = None

    @property
    def sample_id(self) -> str:
        return f"{self.raw_stem}_{self.frame_tag}"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["sample_id"] = self.sample_id
        data["raw_path"] = str(self.raw_path)
        data["filament_mask_path"] = str(self.filament_mask_path) if self.filament_mask_path else None
        data["cell_mask_path"] = str(self.cell_mask_path) if self.cell_mask_path else None
        return data


def _is_mask_stem(stem: str) -> bool:
    return MASK_RE.match(stem) is not None


def _parse_mask(path: Path) -> tuple[str, int, str] | None:
    match = MASK_RE.match(path.stem)
    if not match:
        return None
    base = match.group("base")
    frame_index = int(match.group("frame"))
    kind = match.group("kind").lower()
    return base, frame_index, kind


def build_dataset_index(root: Path) -> pd.DataFrame:
    tif_paths = sorted(root.rglob("*.tif"))
    raw_by_stem = {path.stem: path for path in tif_paths if not _is_mask_stem(path.stem)}

    records: dict[tuple[str, int], FrameRecord] = {}
    for stem, raw_path in raw_by_stem.items():
        records[(stem, 0)] = FrameRecord(
            raw_path=raw_path,
            raw_stem=stem,
            frame_index=0,
            frame_tag="f0000",
        )

    for path in tif_paths:
        parsed = _parse_mask(path)
        if parsed is None:
            continue
        base, frame_index, kind = parsed
        raw_path = raw_by_stem.get(base)
        if raw_path is None:
            continue
        key = (base, frame_index)
        record = records.get(key)
        if record is None:
            record = FrameRecord(
                raw_path=raw_path,
                raw_stem=base,
                frame_index=frame_index,
                frame_tag=f"f{frame_index:04d}",
            )
            records[key] = record
        if kind == "mask" or kind.startswith("mask_"):
            record.filament_mask_path = path
        elif kind.startswith("cellmask"):
            if record.cell_mask_path is None or path.name < record.cell_mask_path.name:
                record.cell_mask_path = path

    frame_records = [record.to_dict() for _, record in sorted(records.items(), key=lambda item: (item[0][0], item[0][1]))]
    index = pd.DataFrame(frame_records)
    if index.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "raw_path",
                "raw_stem",
                "frame_index",
                "frame_tag",
                "filament_mask_path",
                "cell_mask_path",
            ]
        )

    index["has_filament_mask"] = index["filament_mask_path"].notna()
    index["has_cell_mask"] = index["cell_mask_path"].notna()
    index["has_full_annotation"] = index["has_filament_mask"] & index["has_cell_mask"]
    return index


def save_dataset_index(index: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
