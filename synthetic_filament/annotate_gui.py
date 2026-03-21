from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import re

import numpy as np
from PIL import Image, ImageTk
from tifffile import imread, imwrite


def list_raw_tifs(data_dir: Path) -> list[Path]:
    tif_paths = sorted(data_dir.glob("*.tif"))
    mask_re = re.compile(r".+_f\d{4}_(mask|cellmask(?:_\d+)?)$", re.IGNORECASE)
    return [p for p in tif_paths if mask_re.match(p.stem) is None]


def normalize_frame(frame: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo = np.percentile(frame, lo_pct)
    hi = np.percentile(frame, hi_pct)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def mask_path_for(raw_path: Path, frame_index: int, object_kind: str, object_index: int) -> Path:
    if object_kind == "filament":
        suffix = f"_mask_{object_index}.tif"
    else:
        suffix = f"_cellmask_{object_index}.tif"
    return raw_path.with_name(f"{raw_path.stem}_f{frame_index:04d}{suffix}")


def find_existing_masks(raw_path: Path, frame_index: int, object_kind: str) -> list[Path]:
    base = f"{raw_path.stem}_f{frame_index:04d}"
    if object_kind == "filament":
        pattern = f"{base}_mask*.tif"
    else:
        pattern = f"{base}_cellmask_*.tif"
    return sorted(raw_path.parent.glob(pattern))


@dataclass
class PromptState:
    positive: list[tuple[int, int]]
    negative: list[tuple[int, int]]
    mask: np.ndarray | None = None


class SAM2Backend:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.error: str | None = None
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.error = f"PyTorch is required for SAM2 GPU inference: {exc}"
            return
        if not torch.cuda.is_available():
            self.error = "SAM2 annotator is configured for GPU use, but CUDA is not available."
            return
        try:
            from ultralytics import SAM  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.error = f"Failed to import ultralytics.SAM: {exc}"
            return
        try:
            self.model = SAM(model_path)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
        except Exception as exc:  # pragma: no cover
            self.error = f"Failed to load SAM/SAM2 model '{model_path}': {exc}"

    def available(self) -> bool:
        return self.model is not None

    def predict_mask(self, image_rgb: np.ndarray, positive: list[tuple[int, int]], negative: list[tuple[int, int]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(self.error or "SAM2 backend is unavailable")

        all_points = positive + negative
        if not all_points:
            raise ValueError("Add at least one point before segmenting.")
        labels = [1] * len(positive) + [0] * len(negative)

        call_attempts = [
            {"points": all_points, "labels": labels, "device": self.device, "verbose": False},
            {"point_coords": all_points, "point_labels": labels, "device": self.device, "verbose": False},
        ]
        last_error: Exception | None = None
        for kwargs in call_attempts:
            try:
                results = self.model(image_rgb, **kwargs)
                mask = self._extract_mask(results, image_rgb.shape[:2])
                if mask is not None:
                    return mask
            except Exception as exc:  # pragma: no cover
                last_error = exc
        try:
            results = self.model.predict(image_rgb, points=all_points, labels=labels, device=self.device, verbose=False)
            mask = self._extract_mask(results, image_rgb.shape[:2])
            if mask is not None:
                return mask
        except Exception as exc:  # pragma: no cover
            last_error = exc
        raise RuntimeError(f"SAM2 prediction failed. Last error: {last_error}")

    @staticmethod
    def _extract_mask(results: object, image_shape: tuple[int, int]) -> np.ndarray | None:
        if results is None:
            return None
        if not isinstance(results, (list, tuple)):
            results = [results]
        best_mask = None
        best_area = -1
        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None:
                continue
            data = getattr(masks, "data", None)
            if data is None:
                continue
            if hasattr(data, "cpu"):
                arr = data.cpu().numpy()
            else:
                arr = np.asarray(data)
            if arr.ndim == 2:
                arr = arr[None, ...]
            for candidate in arr:
                candidate = np.asarray(candidate) > 0
                if candidate.shape != image_shape:
                    continue
                area = int(candidate.sum())
                if area > best_area:
                    best_area = area
                    best_mask = candidate
        return best_mask


class AnnotatorApp:
    def __init__(self, root: tk.Tk, data_dir: Path, model_path: str, device: str):
        self.root = root
        self.data_dir = data_dir
        self.backend = SAM2Backend(model_path, device=device)
        self.raw_paths = list_raw_tifs(data_dir)
        self.raw_index = 0
        self.frame_index = 0
        self.display_scale = 4
        self.current_stack: np.ndarray | None = None
        self.current_raw_path: Path | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.object_kind = tk.StringVar(value="filament")
        self.click_mode = tk.StringVar(value="positive")
        self.status = tk.StringVar(value="")
        self.low_pct = tk.DoubleVar(value=1.0)
        self.high_pct = tk.DoubleVar(value=99.5)
        self.object_index = tk.IntVar(value=1)
        self.prompt_cache: dict[tuple[str, int, str], PromptState] = {}

        self._build_ui()
        if not self.raw_paths:
            self.status.set(f"No raw TIFF stacks found in {data_dir}")
        else:
            self.load_raw(0)

    def _build_ui(self) -> None:
        self.root.title("SAM2 TIFF Annotator")
        self.root.geometry("980x780")

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(controls)
        row1.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(row1, text="Open Dir", command=self.choose_directory).pack(side=tk.LEFT)
        ttk.Button(row1, text="Prev TIFF", command=self.prev_raw).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Next TIFF", command=self.next_raw).pack(side=tk.LEFT)
        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(row1, text="Prev Frame", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(row1, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=4)
        self.frame_label = ttk.Label(row1, text="frame")
        self.frame_label.pack(side=tk.LEFT, padx=8)

        row2 = ttk.Frame(controls)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Radiobutton(row2, text="Filament", variable=self.object_kind, value="filament", command=self.on_object_change).pack(side=tk.LEFT)
        ttk.Radiobutton(row2, text="Cell", variable=self.object_kind, value="cell", command=self.on_object_change).pack(side=tk.LEFT, padx=4)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(row2, text="Object #").pack(side=tk.LEFT)
        self.object_index_spin = tk.Spinbox(
            row2,
            from_=1,
            to=99,
            width=4,
            textvariable=self.object_index,
            command=self.on_object_change,
        )
        self.object_index_spin.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(row2, text="Prev Obj", command=self.prev_object_index).pack(side=tk.LEFT)
        ttk.Button(row2, text="Next Obj", command=self.next_object_index).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="New Obj", command=self.new_object_index).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Radiobutton(row2, text="Positive Click", variable=self.click_mode, value="positive").pack(side=tk.LEFT)
        ttk.Radiobutton(row2, text="Negative Click", variable=self.click_mode, value="negative").pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(controls)
        row3.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(row3, text="Segment", command=self.run_segmentation).pack(side=tk.LEFT)
        ttk.Button(row3, text="Clear Points", command=self.clear_points).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="Reload Mask", command=self.reload_existing_mask).pack(side=tk.LEFT)
        ttk.Button(row3, text="Save Mask", command=self.save_current_mask).pack(side=tk.LEFT, padx=4)
        ttk.Label(row3, text="Low %").pack(side=tk.LEFT, padx=(12, 0))
        low_slider = tk.Scale(
            row3,
            from_=0.0,
            to=40.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.low_pct,
            command=self.on_contrast_change,
            length=180,
        )
        low_slider.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(row3, text="High %").pack(side=tk.LEFT)
        high_slider = tk.Scale(
            row3,
            from_=60.0,
            to=100.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.high_pct,
            command=self.on_contrast_change,
            length=180,
        )
        high_slider.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(row3, text="Reset Contrast", command=self.reset_contrast).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        footer = ttk.Frame(self.root, padding=8)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(footer, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.root.bind("<Left>", lambda _event: self.prev_frame())
        self.root.bind("<Right>", lambda _event: self.next_frame())
        self.root.bind("s", lambda _event: self.run_segmentation())
        self.root.bind("w", lambda _event: self.save_current_mask())

    def choose_directory(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self.data_dir))
        if not selected:
            return
        self.data_dir = Path(selected)
        self.raw_paths = list_raw_tifs(self.data_dir)
        self.raw_index = 0
        self.frame_index = 0
        if not self.raw_paths:
            self.current_stack = None
            self.current_raw_path = None
            self.status.set(f"No raw TIFF stacks found in {self.data_dir}")
            self.refresh_view()
            return
        self.load_raw(0)

    def _current_key(self) -> tuple[str, int, str]:
        assert self.current_raw_path is not None
        return (str(self.current_raw_path), self.frame_index, f"{self.object_kind.get()}_{self.object_index.get()}")

    def _state(self) -> PromptState:
        key = self._current_key()
        if key not in self.prompt_cache:
            self.prompt_cache[key] = PromptState(positive=[], negative=[], mask=None)
        return self.prompt_cache[key]

    def on_object_change(self) -> None:
        try:
            value = int(self.object_index_spin.get())
        except ValueError:
            value = 1
        self.object_index.set(max(1, value))
        self.reload_existing_mask()
        self.refresh_view()

    def prev_object_index(self) -> None:
        self.object_index.set(max(1, self.object_index.get() - 1))
        self.on_object_change()

    def next_object_index(self) -> None:
        self.object_index.set(self.object_index.get() + 1)
        self.on_object_change()

    def new_object_index(self) -> None:
        if self.current_raw_path is None:
            return
        existing = find_existing_masks(self.current_raw_path, self.frame_index, self.object_kind.get())
        next_index = max([self._object_index_from_path(p, self.object_kind.get()) for p in existing] + [0]) + 1
        self.object_index.set(next_index)
        self.clear_points()
        self.status.set(f"Started new {self.object_kind.get()} object #{next_index}")
        self.refresh_view()

    def on_contrast_change(self, _value: str | None = None) -> None:
        if self.low_pct.get() >= self.high_pct.get():
            if _value is not None:
                if float(_value) == self.low_pct.get():
                    self.high_pct.set(min(100.0, self.low_pct.get() + 0.5))
                else:
                    self.low_pct.set(max(0.0, self.high_pct.get() - 0.5))
        self.refresh_view()

    def reset_contrast(self) -> None:
        self.low_pct.set(1.0)
        self.high_pct.set(99.5)
        self.refresh_view()

    def load_raw(self, index: int) -> None:
        self.raw_index = int(np.clip(index, 0, max(len(self.raw_paths) - 1, 0)))
        self.current_raw_path = self.raw_paths[self.raw_index]
        stack = imread(self.current_raw_path)
        self.current_stack = stack if stack.ndim == 3 else stack[None, ...]
        self.frame_index = int(np.clip(self.frame_index, 0, self.current_stack.shape[0] - 1))
        self.reload_existing_mask()
        self.refresh_view()

    def prev_raw(self) -> None:
        if self.raw_paths:
            self.load_raw(max(0, self.raw_index - 1))

    def next_raw(self) -> None:
        if self.raw_paths:
            self.load_raw(min(len(self.raw_paths) - 1, self.raw_index + 1))

    def prev_frame(self) -> None:
        if self.current_stack is None:
            return
        self.frame_index = max(0, self.frame_index - 1)
        self.reload_existing_mask()
        self.refresh_view()

    def next_frame(self) -> None:
        if self.current_stack is None:
            return
        self.frame_index = min(self.current_stack.shape[0] - 1, self.frame_index + 1)
        self.reload_existing_mask()
        self.refresh_view()

    def on_left_click(self, event: tk.Event) -> None:
        self._handle_click(event, force_mode="positive")

    def on_right_click(self, event: tk.Event) -> None:
        self._handle_click(event, force_mode="negative")

    def _handle_click(self, event: tk.Event, force_mode: str | None = None) -> None:
        if self.current_stack is None:
            return
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        frame = self.current_stack[self.frame_index]
        if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            return
        mode = force_mode or self.click_mode.get()
        state = self._state()
        if mode == "positive":
            state.positive.append((x, y))
        else:
            state.negative.append((x, y))
        self.status.set(f"Added {mode} point at ({x}, {y})")
        self.refresh_view()

    def clear_points(self) -> None:
        state = self._state()
        state.positive.clear()
        state.negative.clear()
        state.mask = None
        self.status.set("Cleared points and preview mask for current object/frame")
        self.refresh_view()

    def reload_existing_mask(self) -> None:
        if self.current_raw_path is None:
            return
        state = self._state()
        path = mask_path_for(self.current_raw_path, self.frame_index, self.object_kind.get(), self.object_index.get())
        if not path.exists() and self.object_kind.get() == "filament" and self.object_index.get() == 1:
            legacy_path = self.current_raw_path.with_name(f"{self.current_raw_path.stem}_f{self.frame_index:04d}_mask.tif")
            path = legacy_path if legacy_path.exists() else path
        if path.exists():
            state.mask = imread(path).astype(bool)
            self.status.set(f"Loaded existing {self.object_kind.get()} mask #{self.object_index.get()}: {path.name}")
        else:
            state.mask = None

    def run_segmentation(self) -> None:
        if self.current_stack is None:
            return
        if not self.backend.available():
            messagebox.showerror("SAM2 unavailable", self.backend.error or "ultralytics SAM2 backend not available")
            return
        frame = self.current_stack[self.frame_index]
        preview = normalize_frame(frame, lo_pct=self.low_pct.get(), hi_pct=self.high_pct.get())
        image_rgb = np.repeat(preview[..., None], 3, axis=2)
        state = self._state()
        try:
            state.mask = self.backend.predict_mask(image_rgb, state.positive, state.negative)
        except Exception as exc:
            messagebox.showerror("Segmentation failed", str(exc))
            return
        self.status.set(f"Updated {self.object_kind.get()} mask preview")
        self.refresh_view()

    def save_current_mask(self) -> None:
        if self.current_raw_path is None:
            return
        state = self._state()
        if state.mask is None:
            messagebox.showwarning("No mask", "Run segmentation or load an existing mask first.")
            return
        out_path = mask_path_for(self.current_raw_path, self.frame_index, self.object_kind.get(), self.object_index.get())
        imwrite(out_path, state.mask.astype(np.uint8))
        self.status.set(f"Saved mask to {out_path.name}")

    def _compose_overlay(self) -> np.ndarray:
        if self.current_stack is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        frame = self.current_stack[self.frame_index]
        image = normalize_frame(frame, lo_pct=self.low_pct.get(), hi_pct=self.high_pct.get())
        rgb = np.repeat(image[..., None], 3, axis=2)
        state = self._state()
        if state.mask is not None and state.mask.shape == frame.shape:
            color = np.array([255, 80, 80], dtype=np.uint8) if self.object_kind.get() == "filament" else np.array([80, 220, 120], dtype=np.uint8)
            alpha = 0.35
            rgb[state.mask] = (rgb[state.mask] * (1 - alpha) + color * alpha).astype(np.uint8)

        for x, y in state.positive:
            self._draw_point(rgb, x, y, (0, 255, 0))
        for x, y in state.negative:
            self._draw_point(rgb, x, y, (255, 0, 0))
        return rgb

    @staticmethod
    def _draw_point(rgb: np.ndarray, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
        y0 = max(0, y - radius)
        y1 = min(rgb.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(rgb.shape[1], x + radius + 1)
        rgb[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)

    def refresh_view(self) -> None:
        overlay = self._compose_overlay()
        display = Image.fromarray(overlay).resize(
            (overlay.shape[1] * self.display_scale, overlay.shape[0] * self.display_scale),
            resample=Image.Resampling.NEAREST,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.config(width=display.width, height=display.height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.current_raw_path is None or self.current_stack is None:
            self.frame_label.config(text="No TIFF loaded")
            return
        self.frame_label.config(
            text=f"{self.current_raw_path.name} | frame {self.frame_index + 1}/{self.current_stack.shape[0]} | object={self.object_kind.get()} #{self.object_index.get()}"
        )

    @staticmethod
    def _object_index_from_path(path: Path, object_kind: str) -> int:
        stem = path.stem
        if object_kind == "filament":
            match = re.search(r"_mask(?:_(\d+))?$", stem)
        else:
            match = re.search(r"_cellmask_(\d+)$", stem)
        if match is None:
            return 1
        group = match.group(1)
        return int(group) if group is not None else 1


def launch_annotator(data_dir: Path, model_path: str, device: str = "cuda:0") -> None:
    root = tk.Tk()
    app = AnnotatorApp(root, data_dir=data_dir, model_path=model_path, device=device)
    root.mainloop()
