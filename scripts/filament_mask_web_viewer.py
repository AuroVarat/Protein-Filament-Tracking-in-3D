#!/usr/bin/env python3
"""
Standalone local web viewer for labeled filament mask TIFF stacks.

This is intentionally independent of the older Gradio viewers.
It serves a small HTML/JS app backed by the saved mask TIFFs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import tifffile


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Filament Mask Viewer</title>
  <style>
    :root {
      --bg: #0d1316;
      --panel: #141d21;
      --panel-2: #19252b;
      --text: #ecf1f3;
      --muted: #8aa0ab;
      --line: rgba(255,255,255,0.08);
      --accent: #31d0aa;
      --accent-2: #ff7a59;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #0c1215 0%, #11191d 100%);
    }
    .app {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.18);
    }
    .controls {
      padding: 18px;
      position: sticky;
      top: 20px;
      height: fit-content;
    }
    h1 {
      font-size: 22px;
      margin: 0 0 8px;
    }
    .subtle {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 18px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin: 14px 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    select, input[type="range"], button {
      width: 100%;
    }
    select, button {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
    }
    input[type="range"] {
      accent-color: var(--accent);
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .button-row button {
      cursor: pointer;
    }
    .status {
      margin-top: 14px;
      padding: 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      min-height: 72px;
      white-space: pre-wrap;
    }
    .id-list {
      margin-top: 8px;
      max-height: 240px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
      background: rgba(255,255,255,0.02);
    }
    .id-chip {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 4px;
      font-size: 13px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex: 0 0 auto;
    }
    .viewer {
      display: grid;
      grid-template-rows: auto auto;
      gap: 18px;
    }
    .main-canvas-wrap {
      padding: 16px;
    }
    .main-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .main-head h2, .proj-head h2 {
      margin: 0;
      font-size: 16px;
    }
    .small-note {
      color: var(--muted);
      font-size: 12px;
    }
    canvas {
      width: 100%;
      display: block;
      background: #0f171a;
      border-radius: 14px;
      border: 1px solid var(--line);
    }
    #main-canvas {
      aspect-ratio: 1.35 / 1;
      cursor: grab;
    }
    #main-canvas.dragging {
      cursor: grabbing;
    }
    .projections {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
    }
    .proj-card {
      padding: 14px;
    }
    .proj-head {
      display: flex;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .proj-canvas {
      aspect-ratio: 1 / 1;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .controls { position: static; }
      .projections { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel controls">
      <h1>Filament Mask Viewer</h1>
      <div class="subtle">Local viewer for labeled `(T, Z, Y, X)` mask TIFFs. Drag the 3D canvas to rotate. This is mask-only; it does not rerun inference.</div>

      <label for="file-select">Mask TIFF</label>
      <select id="file-select"></select>

      <label for="frame-slider">Frame</label>
      <input id="frame-slider" type="range" min="0" max="0" value="0">
      <div class="small-note"><span id="frame-label">Frame 0</span></div>

      <label for="point-size">Point Size</label>
      <input id="point-size" type="range" min="1" max="6" value="3" step="0.5">

      <div class="button-row">
        <button id="play-btn" type="button">Play</button>
        <button id="reset-btn" type="button">Reset View</button>
      </div>

      <label style="margin-top:14px; text-transform:none; letter-spacing:0;">
        <input id="show-cell-outline" type="checkbox" checked style="width:auto; margin-right:8px;">
        Show cell outline
      </label>

      <label>Visible IDs</label>
      <div id="id-list" class="id-list"></div>

      <div id="status" class="status">Loading file list...</div>
    </aside>

    <main class="viewer">
      <section class="panel main-canvas-wrap">
        <div class="main-head">
          <h2 id="main-title">3D Mask View</h2>
          <div class="small-note">Drag to rotate. Scroll on page normally; no wheel zoom here.</div>
        </div>
        <canvas id="main-canvas" width="980" height="720"></canvas>
      </section>

      <section class="projections">
        <article class="panel proj-card">
          <div class="proj-head"><h2>XY Projection</h2><span class="small-note">top</span></div>
          <canvas id="xy-canvas" class="proj-canvas" width="360" height="360"></canvas>
        </article>
        <article class="panel proj-card">
          <div class="proj-head"><h2>XZ Projection</h2><span class="small-note">side</span></div>
          <canvas id="xz-canvas" class="proj-canvas" width="360" height="360"></canvas>
        </article>
        <article class="panel proj-card">
          <div class="proj-head"><h2>YZ Projection</h2><span class="small-note">front</span></div>
          <canvas id="yz-canvas" class="proj-canvas" width="360" height="360"></canvas>
        </article>
      </section>
    </main>
  </div>

  <script>
    const COLORS = [
      "#39ff14", "#ff4fd8", "#ffe600", "#00e5ff", "#ff7a00",
      "#b266ff", "#ff1744", "#00ffa6", "#8bc34a", "#00b0ff"
    ];

    const state = {
      files: [],
      file: null,
      meta: null,
      frame: null,
      frameIndex: 0,
      selectedIds: new Set(),
      pointSize: 3,
      showCellOutline: true,
      playing: false,
      playHandle: null,
      rotationAzimuth: -0.9,
      rotationElevation: 0.55,
      dragging: false,
      dragStart: null,
    };

    const el = {
      fileSelect: document.getElementById("file-select"),
      frameSlider: document.getElementById("frame-slider"),
      frameLabel: document.getElementById("frame-label"),
      pointSize: document.getElementById("point-size"),
      playBtn: document.getElementById("play-btn"),
      resetBtn: document.getElementById("reset-btn"),
      showCellOutline: document.getElementById("show-cell-outline"),
      idList: document.getElementById("id-list"),
      status: document.getElementById("status"),
      title: document.getElementById("main-title"),
      mainCanvas: document.getElementById("main-canvas"),
      xyCanvas: document.getElementById("xy-canvas"),
      xzCanvas: document.getElementById("xz-canvas"),
      yzCanvas: document.getElementById("yz-canvas"),
    };

    function colorForId(id) {
      return COLORS[(id - 1) % COLORS.length];
    }

    async function fetchJson(path) {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    function setStatus(message) {
      el.status.textContent = message;
    }

    async function loadFiles() {
      const payload = await fetchJson("/api/files");
      state.files = payload.files;
      el.fileSelect.innerHTML = payload.files
        .map((item) => `<option value="${item.path}">${item.name}</option>`)
        .join("");
      if (payload.files.length) {
        await selectFile(payload.files[0].path);
      } else {
        setStatus("No mask TIFFs found in the configured mask directory.");
      }
    }

    async function selectFile(path) {
      state.file = path;
      const q = encodeURIComponent(path);
      state.meta = await fetchJson(`/api/meta?path=${q}`);
      state.frameIndex = 0;
      state.selectedIds = new Set(state.meta.labels);
      el.frameSlider.max = String(Math.max(0, state.meta.frames - 1));
      el.frameSlider.value = "0";
      renderIdList();
      await loadFrame(0);
    }

    async function loadFrame(index) {
      if (!state.file) return;
      const q = encodeURIComponent(state.file);
      state.frameIndex = index;
      state.frame = await fetchJson(`/api/frame?path=${q}&frame=${index}`);
      el.frameLabel.textContent = `Frame ${index}`;
      el.title.textContent = `${state.meta.name} | frame ${index}`;
      setStatus(
        `frames=${state.meta.frames}, z=${state.meta.z_planes}, size=${state.meta.height}x${state.meta.width}\\n` +
        `frame=${index}, active_ids=${state.frame.active_labels.length}, voxels=${state.frame.voxel_count}, cell_outline=${state.meta.has_cell_mask ? "yes" : "no"}`
      );
      drawAll();
    }

    function renderIdList() {
      el.idList.innerHTML = "";
      for (const id of state.meta.labels) {
        const row = document.createElement("label");
        row.className = "id-chip";
        const checked = state.selectedIds.has(id) ? "checked" : "";
        row.innerHTML = `
          <input type="checkbox" data-id="${id}" ${checked}>
          <span class="swatch" style="background:${colorForId(id)}"></span>
          <span>Filament ${id}</span>
        `;
        el.idList.appendChild(row);
      }
      el.idList.querySelectorAll("input[type=checkbox]").forEach((input) => {
        input.addEventListener("change", () => {
          const id = Number(input.dataset.id);
          if (input.checked) state.selectedIds.add(id);
          else state.selectedIds.delete(id);
          drawAll();
        });
      });
    }

    function visiblePoints() {
      if (!state.frame) return [];
      return state.frame.points.filter((p) => state.selectedIds.has(p.id));
    }

    function project3d(point, dims, width, height) {
      const zStretch = Math.max(8, Math.min(14, (dims.height / Math.max(dims.z, 1)) * 0.35));
      const centered = [
        point.x - dims.width / 2,
        point.y - dims.height / 2,
        point.z * zStretch - (dims.z * zStretch) / 2,
      ];

      const cosAz = Math.cos(state.rotationAzimuth);
      const sinAz = Math.sin(state.rotationAzimuth);
      const rotY = [
        cosAz * centered[0] + sinAz * centered[2],
        centered[1],
        -sinAz * centered[0] + cosAz * centered[2],
      ];

      const cosEl = Math.cos(state.rotationElevation);
      const sinEl = Math.sin(state.rotationElevation);
      const rotated = [
        rotY[0],
        cosEl * rotY[1] - sinEl * rotY[2],
        sinEl * rotY[1] + cosEl * rotY[2],
      ];

      const baseScale = Math.min(
        width / (dims.width * 1.3),
        height / ((dims.height + dims.z * zStretch) * 0.95)
      );
      const cameraDistance = Math.max(dims.width, dims.height) * 2.6;
      const perspective = cameraDistance / (cameraDistance + rotated[2] + 80);

      return {
        x: width / 2 + rotated[0] * baseScale * perspective,
        y: height / 2 - rotated[1] * baseScale * perspective,
        depth: rotated[2],
        perspective,
      };
    }

    function drawMainCanvas() {
      const canvas = el.mainCanvas;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0f171a";
      ctx.fillRect(0, 0, width, height);

      if (!state.meta || !state.frame) return;

      const dims = { width: state.meta.width, height: state.meta.height, z: state.meta.z_planes };
      drawBox(ctx, dims, width, height);

      const projected = visiblePoints()
        .map((p) => ({ p, v: project3d(p, dims, width, height) }))
        .sort((a, b) => a.v.depth - b.v.depth);

      if (state.showCellOutline && state.frame.cell_points) {
        const cellProjected = state.frame.cell_points
          .filter((p) => !state.selectedIds.size || state.selectedIds.has(p.id))
          .map((p) => ({ p, v: project3d(p, dims, width, height) }))
          .sort((a, b) => a.v.depth - b.v.depth);
        for (const item of cellProjected) {
          ctx.fillStyle = "rgba(255,255,255,0.20)";
          ctx.beginPath();
          ctx.arc(item.v.x, item.v.y, Math.max(0.9, state.pointSize * 0.35 * item.v.perspective), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      for (const item of projected) {
        const alpha = Math.max(0.15, Math.min(0.95, 0.25 + (item.p.z / Math.max(dims.z, 1)) * 0.55));
        ctx.fillStyle = hexToRgba(colorForId(item.p.id), alpha);
        ctx.beginPath();
        ctx.arc(item.v.x, item.v.y, state.pointSize * item.v.perspective, 0, Math.PI * 2);
        ctx.fill();
      }

      drawAxisGizmo(ctx, width, height);
    }

    function drawBox(ctx, dims, width, height) {
      const zStretch = Math.max(8, Math.min(14, (dims.height / Math.max(dims.z, 1)) * 0.35));
      const corners = [
        { x: 0, y: 0, z: 0 }, { x: dims.width, y: 0, z: 0 },
        { x: dims.width, y: dims.height, z: 0 }, { x: 0, y: dims.height, z: 0 },
        { x: 0, y: 0, z: dims.z }, { x: dims.width, y: 0, z: dims.z },
        { x: dims.width, y: dims.height, z: dims.z }, { x: 0, y: dims.height, z: dims.z },
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1.2;
      for (const [a, b] of edges) {
        const pa = project3d(corners[a], dims, width, height);
        const pb = project3d(corners[b], dims, width, height);
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
    }

    function drawAxisGizmo(ctx, width, height) {
      const anchor = { x: 90, y: height - 80 };
      const length = 34;
      const basis = [
        { label: "x", color: "#ef7b45", vector: [1, 0, 0] },
        { label: "y", color: "#7fb8ad", vector: [0, 1, 0] },
        { label: "z", color: "#f4d35e", vector: [0, 0, 1] },
      ];
      for (const axis of basis) {
        const rotated = rotateVector(axis.vector);
        ctx.strokeStyle = axis.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(anchor.x, anchor.y);
        ctx.lineTo(anchor.x + rotated[0] * length, anchor.y - rotated[1] * length);
        ctx.stroke();
        ctx.fillStyle = axis.color;
        ctx.font = "600 12px Inter, sans-serif";
        ctx.fillText(axis.label, anchor.x + rotated[0] * (length + 8), anchor.y - rotated[1] * (length + 8));
      }
    }

    function rotateVector(vector) {
      const cosAz = Math.cos(state.rotationAzimuth);
      const sinAz = Math.sin(state.rotationAzimuth);
      const rotY = [
        cosAz * vector[0] + sinAz * vector[2],
        vector[1],
        -sinAz * vector[0] + cosAz * vector[2],
      ];
      const cosEl = Math.cos(state.rotationElevation);
      const sinEl = Math.sin(state.rotationElevation);
      return [
        rotY[0],
        cosEl * rotY[1] - sinEl * rotY[2],
        sinEl * rotY[1] + cosEl * rotY[2],
      ];
    }

    function drawProjection(canvas, keyX, keyY, rangeX, rangeY) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0f171a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);

      if (!state.frame) return;

      if (state.showCellOutline && state.frame.cell_points) {
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        for (const point of state.frame.cell_points) {
          if (state.selectedIds.size && !state.selectedIds.has(point.id)) continue;
          const x = (point[keyX] / Math.max(rangeX, 1)) * canvas.width;
          const y = (point[keyY] / Math.max(rangeY, 1)) * canvas.height;
          ctx.fillRect(x, y, 1.2, 1.2);
        }
      }

      for (const point of visiblePoints()) {
        const x = (point[keyX] / Math.max(rangeX, 1)) * canvas.width;
        const y = (point[keyY] / Math.max(rangeY, 1)) * canvas.height;
        ctx.fillStyle = colorForId(point.id);
        ctx.beginPath();
        ctx.arc(x, y, 2.1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function drawAll() {
      drawMainCanvas();
      if (!state.meta) return;
      drawProjection(el.xyCanvas, "x", "y", state.meta.width, state.meta.height);
      drawProjection(el.xzCanvas, "x", "zScaled", state.meta.width, state.meta.z_planes - 1 || 1);
      drawProjection(el.yzCanvas, "y", "zScaled", state.meta.height, state.meta.z_planes - 1 || 1);
    }

    function hexToRgba(hex, alpha) {
      const clean = hex.replace("#", "");
      const r = parseInt(clean.slice(0, 2), 16);
      const g = parseInt(clean.slice(2, 4), 16);
      const b = parseInt(clean.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function togglePlay() {
      state.playing = !state.playing;
      el.playBtn.textContent = state.playing ? "Pause" : "Play";
      if (state.playing) {
        state.playHandle = setInterval(async () => {
          if (!state.meta) return;
          const next = (state.frameIndex + 1) % state.meta.frames;
          el.frameSlider.value = String(next);
          await loadFrame(next);
        }, 250);
      } else if (state.playHandle) {
        clearInterval(state.playHandle);
      }
    }

    function resetView() {
      state.rotationAzimuth = -0.9;
      state.rotationElevation = 0.55;
      drawAll();
    }

    function bindEvents() {
      el.fileSelect.addEventListener("change", async () => {
        await selectFile(el.fileSelect.value);
      });
      el.frameSlider.addEventListener("input", async () => {
        await loadFrame(Number(el.frameSlider.value));
      });
      el.pointSize.addEventListener("input", () => {
        state.pointSize = Number(el.pointSize.value);
        drawAll();
      });
      el.showCellOutline.addEventListener("change", () => {
        state.showCellOutline = el.showCellOutline.checked;
        drawAll();
      });
      el.playBtn.addEventListener("click", togglePlay);
      el.resetBtn.addEventListener("click", resetView);

      el.mainCanvas.addEventListener("mousedown", (event) => {
        state.dragging = true;
        state.dragStart = { x: event.clientX, y: event.clientY };
        el.mainCanvas.classList.add("dragging");
      });
      window.addEventListener("mouseup", () => {
        state.dragging = false;
        state.dragStart = null;
        el.mainCanvas.classList.remove("dragging");
      });
      window.addEventListener("mousemove", (event) => {
        if (!state.dragging || !state.dragStart) return;
        const dx = event.clientX - state.dragStart.x;
        const dy = event.clientY - state.dragStart.y;
        state.dragStart = { x: event.clientX, y: event.clientY };
        state.rotationAzimuth += dx * 0.01;
        state.rotationElevation = Math.max(-1.2, Math.min(1.2, state.rotationElevation + dy * 0.01));
        drawAll();
      });
    }

    async function init() {
      bindEvents();
      await loadFiles();
    }

    init().catch((error) => {
      setStatus(`Failed to initialize viewer: ${error.message}`);
      console.error(error);
    });
  </script>
</body>
</html>
"""


class MaskStore:
    def __init__(self, mask_dir: Path, cell_mask_dir: Path):
        self.mask_dir = mask_dir
        self.cell_mask_dir = cell_mask_dir
        self._cache: dict[str, np.ndarray] = {}
        self._cell_cache: dict[str, np.ndarray] = {}

    def list_files(self) -> list[dict[str, str]]:
        files = sorted(self.mask_dir.glob("*.tif"))
        return [{"name": path.name, "path": str(path.resolve())} for path in files]

    def load(self, path: str) -> np.ndarray:
        resolved = str(Path(path).resolve())
        if resolved not in self._cache:
            arr = tifffile.imread(resolved)
            if arr.ndim != 4:
                raise ValueError(f"Expected (T, Z, Y, X) mask TIFF, got {arr.shape}")
            self._cache[resolved] = arr
        return self._cache[resolved]

    def cell_path_for(self, path: str) -> Path:
        return (self.cell_mask_dir / Path(path).name).resolve()

    def load_cell(self, path: str) -> np.ndarray | None:
        cell_path = self.cell_path_for(path)
        if not cell_path.exists():
            return None
        resolved = str(cell_path)
        if resolved not in self._cell_cache:
            arr = tifffile.imread(resolved)
            if arr.ndim != 4:
                raise ValueError(f"Expected (T, Z, Y, X) cell mask TIFF, got {arr.shape}")
            self._cell_cache[resolved] = arr
        return self._cell_cache[resolved]

    def _slice_boundaries(self, labels_2d: np.ndarray) -> np.ndarray:
        boundary = np.zeros_like(labels_2d, dtype=bool)
        center = labels_2d
        boundary[1:, :] |= center[1:, :] != center[:-1, :]
        boundary[:-1, :] |= center[:-1, :] != center[1:, :]
        boundary[:, 1:] |= center[:, 1:] != center[:, :-1]
        boundary[:, :-1] |= center[:, :-1] != center[:, 1:]
        boundary &= center > 0
        return boundary

    def cell_boundary_points(self, path: str, frame_idx: int) -> list[dict[str, int]]:
        cell = self.load_cell(path)
        if cell is None:
            return []
        frame = cell[frame_idx]
        points: list[dict[str, int]] = []
        for z_idx in range(frame.shape[0]):
            boundary = self._slice_boundaries(frame[z_idx])
            y, x = np.nonzero(boundary)
            labels = frame[z_idx, y, x].astype(int)
            points.extend(
                {
                    "x": int(px),
                    "y": int(py),
                    "z": int(z_idx),
                    "zScaled": int(z_idx),
                    "id": int(fid),
                }
                for py, px, fid in zip(y, x, labels, strict=False)
            )
        return points

    def meta(self, path: str) -> dict[str, object]:
        arr = self.load(path)
        labels = sorted(int(v) for v in np.unique(arr) if v > 0)
        return {
            "name": Path(path).name,
            "path": str(Path(path).resolve()),
            "frames": int(arr.shape[0]),
            "z_planes": int(arr.shape[1]),
            "height": int(arr.shape[2]),
            "width": int(arr.shape[3]),
            "labels": labels,
            "has_cell_mask": self.cell_path_for(path).exists(),
        }

    def frame(self, path: str, frame_idx: int) -> dict[str, object]:
        arr = self.load(path)
        frame_idx = int(np.clip(frame_idx, 0, arr.shape[0] - 1))
        frame = arr[frame_idx]
        z, y, x = np.nonzero(frame)
        labels = frame[z, y, x].astype(int)
        points = [
            {"x": int(px), "y": int(py), "z": int(pz), "zScaled": int(pz), "id": int(fid)}
            for pz, py, px, fid in zip(z, y, x, labels, strict=False)
        ]
        active_labels = sorted({int(v) for v in labels})
        return {
            "frame": frame_idx,
            "voxel_count": int(len(points)),
            "active_labels": active_labels,
            "points": points,
            "cell_points": self.cell_boundary_points(path, frame_idx),
        }


def make_handler(store: MaskStore):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)
            try:
                if path == "/":
                    self._send_html(HTML)
                    return
                if path == "/api/files":
                    self._send_json({"files": store.list_files()})
                    return
                if path == "/api/meta":
                    tif_path = self._require_param(params, "path")
                    self._send_json(store.meta(tif_path))
                    return
                if path == "/api/frame":
                    tif_path = self._require_param(params, "path")
                    frame_idx = int(self._require_param(params, "frame"))
                    self._send_json(store.frame(tif_path, frame_idx))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _require_param(self, params: dict[str, list[str]], key: str) -> str:
            values = params.get(key)
            if not values:
                raise ValueError(f"Missing query parameter: {key}")
            return values[0]

        def _send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web viewer for labeled filament mask TIFFs.")
    parser.add_argument("--mask-dir", default="results/masks", help="Directory containing labeled mask TIFFs.")
    parser.add_argument("--cell-mask-dir", default="results/cell_masks", help="Directory containing labeled cell mask TIFFs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local web server to.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the local web server to. If omitted, a free port is chosen automatically.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def choose_port(host: str, requested_port: int | None) -> int:
    if requested_port is not None:
        return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    args = parse_args()
    store = MaskStore(Path(args.mask_dir), Path(args.cell_mask_dir))
    handler = make_handler(store)
    port = choose_port(args.host, args.port)
    with socketserver.ThreadingTCPServer((args.host, port), handler) as httpd:
        url = f"http://{args.host}:{port}"
        print(f"Viewer running at {url}")
        print(f"Reading mask TIFFs from {Path(args.mask_dir).resolve()}")
        print(f"Reading cell mask TIFFs from {Path(args.cell_mask_dir).resolve()}")
        if not args.no_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
