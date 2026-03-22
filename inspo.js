const METRIC_OPTIONS_3D = [
  { key: "filament_length", label: "Filament length", color: "#0d8b8b" },
  { key: "n_recovered_points", label: "Recovered points", color: "#ef7b45" },
  { key: "chosen_linearity", label: "Linearity", color: "#355c7d" },
  { key: "chosen_linear_density", label: "Linear density", color: "#5a7d2b" },
  { key: "chosen_score", label: "Dense-line score", color: "#8d5a97" },
  { key: "tortuosity", label: "Tortuosity", color: "#d1495b" },
];

const state = {
  data: null,
  currentTrackIndex: 0,
  currentFrameIndex: 0,
  metricKey: "filament_length",
  playing: false,
  playHandle: null,
  renderToken: 0,
  showSampled: true,
  showRecovered: true,
  showBackbone: true,
  showAxis: true,
  rotationAzimuth: -0.85,
  rotationElevation: 0.5,
  dragging: false,
  lastPointer: null,
};

const elements = {
  summaryCards: document.querySelector("#summary-cards-3d"),
  volumeTitle: document.querySelector("#volume-title"),
  volumeStatus: document.querySelector("#volume-status"),
  volumeScore: document.querySelector("#volume-score"),
  trackSelect: document.querySelector("#track-select"),
  playToggle: document.querySelector("#play-toggle-3d"),
  resetView: document.querySelector("#reset-view"),
  volumeSlider: document.querySelector("#volume-slider"),
  volumeCanvas: document.querySelector("#volume-canvas"),
  toggleSampled: document.querySelector("#toggle-sampled-3d"),
  toggleRecovered: document.querySelector("#toggle-recovered-3d"),
  toggleBackbone: document.querySelector("#toggle-backbone-3d"),
  toggleAxis: document.querySelector("#toggle-axis-3d"),
  volumeMetrics: document.querySelector("#volume-metrics"),
  xyCanvas: document.querySelector("#xy-canvas"),
  xzCanvas: document.querySelector("#xz-canvas"),
  yzCanvas: document.querySelector("#yz-canvas"),
  metricSelect: document.querySelector("#metric-select-3d"),
  timelineChart: document.querySelector("#timeline-chart-3d"),
  componentsChart: document.querySelector("#components-chart-3d"),
  scoreChart: document.querySelector("#score-chart-3d"),
  orientationChart: document.querySelector("#orientation-chart-3d"),
  msdChart: document.querySelector("#msd-chart-3d"),
  methodPoints: document.querySelector("#method-points-3d"),
  parameterList: document.querySelector("#parameter-list-3d"),
  notableVolumes: document.querySelector("#notable-volumes"),
};

const imageCache = new Map();

async function loadData() {
  const response = await fetch("./data/tda_3d_volumes.json");
  if (!response.ok) {
    throw new Error(`Failed to load analysis JSON: ${response.status}`);
  }
  return response.json();
}

function currentTrack() {
  return state.data.tracks[state.currentTrackIndex];
}

function currentFrame() {
  return currentTrack().frames[state.currentFrameIndex];
}

function formatCompact(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Not detected";
  }
  const number = Number(value);
  if (Math.abs(number) >= 1000) {
    return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return number.toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Not detected";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function summaryCard(label, value, footnote) {
  return `
    <article class="summary-card">
      <p class="metric-label">${label}</p>
      <strong class="summary-value">${value}</strong>
      <span class="summary-footnote">${footnote}</span>
    </article>
  `;
}

function metricCard(label, value, footnote) {
  return `
    <article class="metric-card">
      <p class="metric-label">${label}</p>
      <strong class="metric-value">${value}</strong>
      <span class="metric-footnote">${footnote}</span>
    </article>
  `;
}

function renderSummary() {
  const globalSummary = state.data.summary;
  const trackSummary = currentTrack().summary;

  elements.summaryCards.innerHTML = [
    summaryCard("Tracks", `${globalSummary.tracks_total}`, "File and channel combinations"),
    summaryCard("Volumes with filament", `${globalSummary.volumes_with_filament} / ${globalSummary.volumes_total}`, "Across the exported dataset"),
    summaryCard("Selected track detection", `${trackSummary.n_frames_with_filament} / ${trackSummary.n_frames_total}`, "Detected volumes in the active track"),
    summaryCard("Mean length", formatCompact(trackSummary.mean_length), "Recovered 3D backbone length"),
  ].join("");
}

function renderMethod() {
  elements.methodPoints.innerHTML = state.data.metadata.notebook_mapping
    .map((text) => `<div class="finding-item">${text}</div>`)
    .join("");

  elements.parameterList.innerHTML = Object.entries(state.data.metadata.parameters)
    .map(([key, value]) => {
      const renderedValue = Array.isArray(value) ? value.join(", ") : value;
      return `<dt>${key.replaceAll("_", " ")}</dt><dd>${renderedValue}</dd>`;
    })
    .join("");
}

function renderTrackSelect() {
  elements.trackSelect.innerHTML = state.data.tracks
    .map(
      (track, index) =>
        `<option value="${index}">${track.track_label}</option>`
    )
    .join("");
  elements.trackSelect.value = String(state.currentTrackIndex);
}

function renderMetricOptions() {
  elements.metricSelect.innerHTML = METRIC_OPTIONS_3D.map(
    (metric) => `<option value="${metric.key}">${metric.label}</option>`
  ).join("");
  elements.metricSelect.value = state.metricKey;
}

function loadImage(src) {
  if (imageCache.has(src)) {
    return imageCache.get(src);
  }

  const image = new Image();
  image.decoding = "async";
  image.src = src;
  const promise = new Promise((resolve, reject) => {
    image.onload = () => resolve(image);
    image.onerror = reject;
  });
  imageCache.set(src, promise);
  return promise;
}

function visualZStretch(volumeShape) {
  return Math.max(8, Math.min(14, (volumeShape.height / zExtent(volumeShape)) * 0.35));
}

function zExtent(volumeShape) {
  const zScale = state.data?.metadata?.parameters?.z_scale ?? 1;
  return Math.max(volumeShape.depth * zScale, 1);
}

function visualizePoint(point, volumeShape) {
  return [
    point[0],
    -point[1],
    point[2] * visualZStretch(volumeShape),
  ];
}

function projectVisualPoint(point, volumeShape, width, height) {
  const zStretch = visualZStretch(volumeShape);
  const centered = [
    point[0] - volumeShape.width / 2,
    point[1] - volumeShape.height / 2,
    point[2] - (zExtent(volumeShape) * zStretch) / 2,
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
    width / (volumeShape.width * 1.3),
    height / ((volumeShape.height + volumeShape.depth * zStretch) * 0.95)
  );
  const cameraDistance = Math.max(volumeShape.width, volumeShape.height) * 2.6;
  const perspective = cameraDistance / (cameraDistance + rotated[2] + 80);

  return {
    x: width / 2 + rotated[0] * baseScale * perspective,
    y: height / 2 - rotated[1] * baseScale * perspective,
    depth: rotated[2],
    perspective,
  };
}

function drawLine3D(ctx, points, volumeShape, color, lineWidth, width, height) {
  if (!points?.length) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach((point, index) => {
    const projected = projectVisualPoint(visualizePoint(point, volumeShape), volumeShape, width, height);
    if (index === 0) {
      ctx.moveTo(projected.x, projected.y);
    } else {
      ctx.lineTo(projected.x, projected.y);
    }
  });
  ctx.stroke();
}

function drawBox(ctx, volumeShape, width, height) {
  const zStretch = visualZStretch(volumeShape);
  const corners = [
    [0, 0, 0],
    [volumeShape.width, 0, 0],
    [volumeShape.width, volumeShape.height, 0],
    [0, volumeShape.height, 0],
    [0, 0, zExtent(volumeShape) * zStretch],
    [volumeShape.width, 0, zExtent(volumeShape) * zStretch],
    [volumeShape.width, volumeShape.height, zExtent(volumeShape) * zStretch],
    [0, volumeShape.height, zExtent(volumeShape) * zStretch],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];

  ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
  ctx.lineWidth = 1.2;
  for (const [start, end] of edges) {
    const a = projectVisualPoint(corners[start], volumeShape, width, height);
    const b = projectVisualPoint(corners[end], volumeShape, width, height);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawAxisGizmo(ctx, volumeShape, width, height) {
  const anchor = { x: 90, y: height - 84 };
  const length = 36;
  const basis = [
    { label: "x", color: "#ef7b45", vector: [1, 0, 0] },
    { label: "y", color: "#7fb8ad", vector: [0, 1, 0] },
    { label: "z", color: "#f4d35e", vector: [0, 0, 1] },
  ];

  const rotateVector = (vector) => {
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
  };

  for (const axis of basis) {
    const rotated = rotateVector(axis.vector);
    ctx.strokeStyle = axis.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(anchor.x, anchor.y);
    ctx.lineTo(anchor.x + rotated[0] * length, anchor.y - rotated[1] * length);
    ctx.stroke();
    ctx.fillStyle = axis.color;
    ctx.font = "600 12px Avenir Next, Segoe UI, sans-serif";
    ctx.fillText(axis.label, anchor.x + rotated[0] * (length + 8), anchor.y - rotated[1] * (length + 8));
  }
}

function drawPointCloud(ctx, points, volumeShape, width, height, color, radius, alphaScale = 1) {
  if (!points?.length) {
    return;
  }
  const projectedPoints = points
    .map((point) => {
      const projected = projectVisualPoint(visualizePoint(point, volumeShape), volumeShape, width, height);
      return { point, projected };
    })
    .sort((a, b) => a.projected.depth - b.projected.depth);

  for (const item of projectedPoints) {
    const depthFraction = item.point[2] / zExtent(volumeShape);
    const alpha = Math.max(0.08, Math.min(0.95, (0.25 + depthFraction * 0.45) * alphaScale));
    ctx.fillStyle = color.replace("ALPHA", alpha.toFixed(3));
    ctx.beginPath();
    ctx.arc(item.projected.x, item.projected.y, radius * item.projected.perspective, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderVolumeCanvas(frame) {
  const canvas = elements.volumeCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const volumeShape = frame.volume_shape;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101816";
  ctx.fillRect(0, 0, width, height);

  drawBox(ctx, volumeShape, width, height);

  if (state.showSampled) {
    drawPointCloud(ctx, frame.point_cloud_preview, volumeShape, width, height, "rgba(255,255,255,ALPHA)", 2.2, 0.38);
  }
  if (state.showRecovered) {
    drawPointCloud(ctx, frame.recovered_points, volumeShape, width, height, "rgba(13,139,139,ALPHA)", 3.2, 0.95);
  }
  if (state.showBackbone) {
    drawLine3D(ctx, frame.backbone_path, volumeShape, "rgba(239, 123, 69, 0.95)", 4, width, height);
  }
  if (state.showAxis) {
    drawLine3D(ctx, frame.principal_axis, volumeShape, "rgba(244, 211, 94, 0.92)", 3, width, height);
  }

  drawAxisGizmo(ctx, volumeShape, width, height);

  ctx.fillStyle = "rgba(255, 248, 235, 0.92)";
  ctx.font = "600 13px Avenir Next, Segoe UI, sans-serif";
  ctx.fillText(`z display stretch x${visualZStretch(volumeShape).toFixed(1)}`, 24, 28);

  if (!frame.has_filament) {
    ctx.fillStyle = "rgba(13, 20, 18, 0.72)";
    ctx.fillRect(0, height - 88, width, 88);
    ctx.fillStyle = "rgba(255, 248, 235, 0.95)";
    ctx.font = "600 22px Avenir Next, Segoe UI, sans-serif";
    ctx.fillText("No dense 3D filament passed the thresholds in this volume.", 24, height - 34);
  }
}

function projectionCoordinates(point, view, volumeShape, canvas) {
  const yImg = -point[1];
  if (view === "xy") {
    return {
      x: (point[0] / volumeShape.width) * canvas.width,
      y: (yImg / volumeShape.height) * canvas.height,
    };
  }
  if (view === "xz") {
    return {
      x: (point[0] / volumeShape.width) * canvas.width,
      y: (point[2] / zExtent(volumeShape)) * canvas.height,
    };
  }
  return {
    x: (yImg / volumeShape.height) * canvas.width,
    y: (point[2] / zExtent(volumeShape)) * canvas.height,
  };
}

function drawProjectionPoints(ctx, points, view, volumeShape, canvas, color, radius) {
  if (!points?.length) {
    return;
  }
  ctx.fillStyle = color;
  for (const point of points) {
    const coords = projectionCoordinates(point, view, volumeShape, canvas);
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawProjectionPolyline(ctx, points, view, volumeShape, canvas, color, lineWidth) {
  if (!points?.length) {
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach((point, index) => {
    const coords = projectionCoordinates(point, view, volumeShape, canvas);
    if (index === 0) {
      ctx.moveTo(coords.x, coords.y);
    } else {
      ctx.lineTo(coords.x, coords.y);
    }
  });
  ctx.stroke();
}

async function renderProjectionCanvas(canvas, imagePath, frame, view) {
  const ctx = canvas.getContext("2d");
  const token = state.renderToken;
  const image = await loadImage(imagePath);
  if (token !== state.renderToken) {
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  if (state.showSampled) {
    drawProjectionPoints(ctx, frame.point_cloud_preview, view, frame.volume_shape, canvas, "rgba(255,255,255,0.2)", 1.3);
  }
  if (state.showRecovered) {
    drawProjectionPoints(ctx, frame.recovered_points, view, frame.volume_shape, canvas, "rgba(13,139,139,0.8)", 1.8);
  }
  if (state.showBackbone) {
    drawProjectionPolyline(ctx, frame.backbone_path, view, frame.volume_shape, canvas, "rgba(239,123,69,0.95)", 3);
  }
  if (state.showAxis) {
    drawProjectionPolyline(ctx, frame.principal_axis, view, frame.volume_shape, canvas, "rgba(244,211,94,0.92)", 2.5);
  }
}

function renderProjections(frame) {
  state.renderToken += 1;
  renderProjectionCanvas(elements.xyCanvas, frame.image_paths.xy, frame, "xy");
  renderProjectionCanvas(elements.xzCanvas, frame.image_paths.xz, frame, "xz");
  renderProjectionCanvas(elements.yzCanvas, frame.image_paths.yz, frame, "yz");
}

function renderVolumeMetrics(frame) {
  const metrics = frame.metrics;
  const orientation = metrics.yaw_deg === null
    ? "Not detected"
    : `${formatCompact(metrics.yaw_deg)}° / ${formatCompact(metrics.pitch_deg)}° / ${formatCompact(metrics.roll_deg)}°`;

  elements.volumeMetrics.innerHTML = [
    metricCard("Recovered points", `${metrics.n_recovered_points}`, "Weighted samples retained"),
    metricCard("Filament length", formatCompact(metrics.filament_length), "Recovered backbone"),
    metricCard("Linearity", formatCompact(metrics.chosen_linearity), "Major span over transverse span"),
    metricCard("Density", formatCompact(metrics.chosen_linear_density), "Points per unit span"),
    metricCard("Tortuosity", formatCompact(metrics.tortuosity), "Length divided by end-to-end distance"),
    metricCard("Yaw / Pitch / Roll", orientation, "PCA-based orientation proxy"),
  ].join("");
}

function updateHeader(frame) {
  const track = currentTrack();
  elements.volumeTitle.textContent = `${track.track_label} | t=${frame.time_index}`;
  elements.volumeStatus.textContent = frame.has_filament ? "Filament detected" : "No filament";
  elements.volumeStatus.style.background = frame.has_filament ? "rgba(13, 139, 139, 0.14)" : "rgba(97, 113, 108, 0.14)";
  elements.volumeStatus.style.color = frame.has_filament ? "#0d8b8b" : "#61716c";
  elements.volumeScore.textContent = frame.metrics.chosen_score === null
    ? "Dense-line score unavailable"
    : `Dense-line score ${formatCompact(frame.metrics.chosen_score)}`;
}

function validPointsFromSeries(xValues, yValues) {
  return xValues
    .map((x, index) => ({ x, y: yValues[index] }))
    .filter((point) => point.y !== null && point.y !== undefined && !Number.isNaN(Number(point.y)));
}

function linePath(points, xScale, yScale) {
  if (!points.length) {
    return "";
  }
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`)
    .join(" ");
}

function buildLineChart({
  svg,
  title,
  xValues,
  yValues,
  stroke,
  selectedX = null,
  selectedY = null,
  onClick = null,
  showArea = false,
  xLabel = "Time",
  yLabel = "",
  referenceX = null,
}) {
  const width = 720;
  const height = svg.viewBox.baseVal.height || 260;
  const margin = { top: 26, right: 20, bottom: 44, left: 62 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const validPoints = validPointsFromSeries(xValues, yValues);
  if (!validPoints.length) {
    svg.innerHTML = `
      <text x="${margin.left}" y="16" fill="#61716c" font-size="12" letter-spacing="1.8">${title}</text>
      <text x="${width / 2}" y="${height / 2}" fill="#61716c" font-size="15" text-anchor="middle">No finite values available for this chart.</text>
    `;
    return;
  }

  const validY = validPoints.map((point) => Number(point.y));
  const minY = Math.min(...validY);
  const maxY = Math.max(...validY);
  const yPadding = maxY === minY ? 1 : (maxY - minY) * 0.12;
  const domainMinY = minY - yPadding;
  const domainMaxY = maxY + yPadding;

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const xScale = (value) => margin.left + ((value - minX) / Math.max(maxX - minX, 1e-6)) * innerWidth;
  const yScale = (value) =>
    margin.top + innerHeight - ((value - domainMinY) / Math.max(domainMaxY - domainMinY, 1e-6)) * innerHeight;

  const path = linePath(validPoints, xScale, yScale);
  const areaPath = showArea && path
    ? `${path} L ${xScale(validPoints[validPoints.length - 1].x).toFixed(2)} ${(margin.top + innerHeight).toFixed(2)} L ${xScale(validPoints[0].x).toFixed(2)} ${(margin.top + innerHeight).toFixed(2)} Z`
    : "";

  const tickValuesY = Array.from({ length: 5 }, (_, index) => domainMinY + ((domainMaxY - domainMinY) / 4) * index);
  const tickValuesX = Array.from({ length: 6 }, (_, index) => minX + ((maxX - minX) / 5) * index);

  svg.innerHTML = `
    <text x="${margin.left}" y="16" fill="#61716c" font-size="12" letter-spacing="1.8">${title}</text>
    ${tickValuesY
      .map((value) => {
        const y = yScale(value);
        return `
          <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(31, 49, 44, 0.10)" />
          <text x="${margin.left - 10}" y="${y + 4}" fill="#61716c" font-size="11" text-anchor="end">${formatCompact(value, 1)}</text>
        `;
      })
      .join("")}
    ${tickValuesX
      .map((value) => {
        const x = xScale(value);
        return `
          <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + innerHeight}" stroke="rgba(31, 49, 44, 0.06)" />
          <text x="${x}" y="${height - 16}" fill="#61716c" font-size="11" text-anchor="middle">${Math.round(value)}</text>
        `;
      })
      .join("")}
    <line x1="${margin.left}" y1="${margin.top + innerHeight}" x2="${width - margin.right}" y2="${margin.top + innerHeight}" stroke="rgba(31, 49, 44, 0.18)" />
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerHeight}" stroke="rgba(31, 49, 44, 0.18)" />
    ${areaPath ? `<path d="${areaPath}" fill="url(#fill-${svg.id})" opacity="0.18"></path>` : ""}
    <path d="${path}" fill="none" stroke="${stroke}" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"></path>
    ${
      referenceX !== null
        ? `<line x1="${xScale(referenceX)}" y1="${margin.top}" x2="${xScale(referenceX)}" y2="${margin.top + innerHeight}" stroke="rgba(239, 123, 69, 0.9)" stroke-dasharray="6 6"></line>`
        : ""
    }
    ${
      selectedX !== null && selectedY !== null && !Number.isNaN(Number(selectedY))
        ? `
          <line x1="${xScale(selectedX)}" y1="${margin.top}" x2="${xScale(selectedX)}" y2="${margin.top + innerHeight}" stroke="rgba(31,49,44,0.24)" stroke-dasharray="4 6"></line>
          <circle cx="${xScale(selectedX)}" cy="${yScale(selectedY)}" r="4.5" fill="${stroke}" stroke="#fff8eb" stroke-width="2"></circle>
        `
        : ""
    }
    <text x="${width - margin.right}" y="16" fill="#61716c" font-size="11" text-anchor="end">${yLabel}</text>
    <text x="${width - margin.right}" y="${height - 16}" fill="#61716c" font-size="11" text-anchor="end">${xLabel}</text>
    <defs>
      <linearGradient id="fill-${svg.id}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${stroke}"></stop>
        <stop offset="100%" stop-color="${stroke}" stop-opacity="0"></stop>
      </linearGradient>
    </defs>
    ${onClick ? `<rect x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}" fill="transparent" data-hit="true"></rect>` : ""}
  `;

  if (onClick) {
    svg.querySelector("[data-hit='true']").addEventListener("click", (event) => {
      const bounds = svg.getBoundingClientRect();
      const relativeX = event.clientX - bounds.left;
      const value = minX + ((relativeX - margin.left) / innerWidth) * (maxX - minX);
      onClick(value);
    });
  }
}

function buildMultiSeriesChart({
  svg,
  title,
  xValues,
  series,
  selectedX = null,
  xLabel = "Time",
  yLabel = "Angle (deg)",
}) {
  const width = 720;
  const height = svg.viewBox.baseVal.height || 260;
  const margin = { top: 26, right: 20, bottom: 44, left: 62 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const allValid = series.flatMap((entry) => validPointsFromSeries(xValues, entry.values).map((point) => Number(point.y)));
  if (!allValid.length) {
    svg.innerHTML = `
      <text x="${margin.left}" y="16" fill="#61716c" font-size="12" letter-spacing="1.8">${title}</text>
      <text x="${width / 2}" y="${height / 2}" fill="#61716c" font-size="15" text-anchor="middle">No finite values available for this chart.</text>
    `;
    return;
  }

  const minY = Math.min(...allValid);
  const maxY = Math.max(...allValid);
  const yPadding = maxY === minY ? 1 : (maxY - minY) * 0.12;
  const domainMinY = minY - yPadding;
  const domainMaxY = maxY + yPadding;
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);

  const xScale = (value) => margin.left + ((value - minX) / Math.max(maxX - minX, 1e-6)) * innerWidth;
  const yScale = (value) =>
    margin.top + innerHeight - ((value - domainMinY) / Math.max(domainMaxY - domainMinY, 1e-6)) * innerHeight;

  const tickValuesY = Array.from({ length: 5 }, (_, index) => domainMinY + ((domainMaxY - domainMinY) / 4) * index);
  const tickValuesX = Array.from({ length: 6 }, (_, index) => minX + ((maxX - minX) / 5) * index);

  svg.innerHTML = `
    <text x="${margin.left}" y="16" fill="#61716c" font-size="12" letter-spacing="1.8">${title}</text>
    ${tickValuesY
      .map((value) => {
        const y = yScale(value);
        return `
          <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(31,49,44,0.10)" />
          <text x="${margin.left - 10}" y="${y + 4}" fill="#61716c" font-size="11" text-anchor="end">${formatCompact(value, 1)}</text>
        `;
      })
      .join("")}
    ${tickValuesX
      .map((value) => {
        const x = xScale(value);
        return `
          <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + innerHeight}" stroke="rgba(31,49,44,0.06)" />
          <text x="${x}" y="${height - 16}" fill="#61716c" font-size="11" text-anchor="middle">${Math.round(value)}</text>
        `;
      })
      .join("")}
    <line x1="${margin.left}" y1="${margin.top + innerHeight}" x2="${width - margin.right}" y2="${margin.top + innerHeight}" stroke="rgba(31,49,44,0.18)" />
    <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerHeight}" stroke="rgba(31,49,44,0.18)" />
    ${series
      .map((entry) => {
        const points = validPointsFromSeries(xValues, entry.values);
        return `<path d="${linePath(points, xScale, yScale)}" fill="none" stroke="${entry.stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>`;
      })
      .join("")}
    ${
      selectedX !== null
        ? `<line x1="${xScale(selectedX)}" y1="${margin.top}" x2="${xScale(selectedX)}" y2="${margin.top + innerHeight}" stroke="rgba(31,49,44,0.24)" stroke-dasharray="4 6"></line>`
        : ""
    }
    ${series
      .map((entry) => {
        const currentValue = entry.values[xValues.findIndex((value) => value === selectedX)];
        if (currentValue === null || currentValue === undefined || Number.isNaN(Number(currentValue))) {
          return "";
        }
        return `<circle cx="${xScale(selectedX)}" cy="${yScale(currentValue)}" r="4.2" fill="${entry.stroke}" stroke="#fff8eb" stroke-width="2"></circle>`;
      })
      .join("")}
    <text x="${width - margin.right}" y="16" fill="#61716c" font-size="11" text-anchor="end">${yLabel}</text>
    <text x="${width - margin.right}" y="${height - 16}" fill="#61716c" font-size="11" text-anchor="end">${xLabel}</text>
    <g transform="translate(${margin.left}, ${height - 10})">
      ${series
        .map(
          (entry, index) => `
            <rect x="${index * 130}" y="-9" width="12" height="12" rx="4" fill="${entry.stroke}"></rect>
            <text x="${index * 130 + 18}" y="1" fill="#61716c" font-size="11">${entry.label}</text>
          `
        )
        .join("")}
    </g>
  `;
}

function renderTimeline() {
  const track = currentTrack();
  const metric = METRIC_OPTIONS_3D.find((item) => item.key === state.metricKey);
  const xValues = track.time_series.map((row) => row.time_index);
  const yValues = track.time_series.map((row) => row[state.metricKey]);
  const frame = currentFrame();

  buildLineChart({
    svg: elements.timelineChart,
    title: metric.label,
    xValues,
    yValues,
    stroke: metric.color,
    selectedX: frame.time_index,
    selectedY: frame.metrics[state.metricKey],
    showArea: true,
    yLabel: metric.label,
    onClick: (value) => selectFrameByTimeIndex(Math.round(value)),
  });
}

function renderOrientationChart() {
  const track = currentTrack();
  const xValues = track.time_series.map((row) => row.time_index);
  buildMultiSeriesChart({
    svg: elements.orientationChart,
    title: "Orientation through time",
    xValues,
    series: [
      { label: "Yaw", stroke: "#0d8b8b", values: track.time_series.map((row) => row.yaw_deg) },
      { label: "Pitch", stroke: "#ef7b45", values: track.time_series.map((row) => row.pitch_deg) },
      { label: "Roll", stroke: "#355c7d", values: track.time_series.map((row) => row.roll_deg) },
    ],
    selectedX: currentFrame().time_index,
  });
}

function renderDiagnostics(frame) {
  const diagnostics = frame.diagnostics;
  const chosenIndex = diagnostics.eps_values.findIndex((value) => value === diagnostics.chosen_eps);

  buildLineChart({
    svg: elements.componentsChart,
    title: "Connected components vs epsilon",
    xValues: diagnostics.eps_values,
    yValues: diagnostics.component_counts,
    stroke: "#355c7d",
    selectedX: diagnostics.chosen_eps,
    selectedY: chosenIndex >= 0 ? diagnostics.component_counts[chosenIndex] : null,
    yLabel: "Components",
    xLabel: "Epsilon",
    referenceX: diagnostics.chosen_eps,
  });

  buildLineChart({
    svg: elements.scoreChart,
    title: "Dense-line score vs epsilon",
    xValues: diagnostics.eps_values,
    yValues: diagnostics.dense_line_scores,
    stroke: "#ef7b45",
    selectedX: diagnostics.chosen_eps,
    selectedY: chosenIndex >= 0 ? diagnostics.dense_line_scores[chosenIndex] : null,
    yLabel: "Score",
    xLabel: "Epsilon",
    referenceX: diagnostics.chosen_eps,
  });
}

function renderMsd() {
  const track = currentTrack();
  if (!track.msd.length) {
    elements.msdChart.innerHTML = `
      <text x="60" y="40" fill="#61716c" font-size="12" letter-spacing="1.8">MSD</text>
      <text x="360" y="150" fill="#61716c" font-size="15" text-anchor="middle">Need at least two detected volumes for the MSD curve.</text>
    `;
    return;
  }

  buildLineChart({
    svg: elements.msdChart,
    title: "Centroid MSD",
    xValues: track.msd.map((row) => row.lag_time),
    yValues: track.msd.map((row) => row.msd),
    stroke: "#0d8b8b",
    showArea: true,
    yLabel: "MSD",
    xLabel: "Lag",
  });
}

function renderNotables() {
  const track = currentTrack();
  const notableFrames = [...track.frames]
    .filter((frame) => frame.has_filament)
    .sort((a, b) => (b.metrics.chosen_score ?? -Infinity) - (a.metrics.chosen_score ?? -Infinity))
    .slice(0, 6);

  elements.notableVolumes.innerHTML = notableFrames
    .map(
      (frame) => `
        <button class="notable-card" data-time-index="${frame.time_index}" type="button">
          <img src="${frame.image_paths.xy}" alt="Volume ${frame.time_index}" />
          <div class="notable-copy">
            <strong>t = ${frame.time_index}</strong>
            <span>Score ${formatCompact(frame.metrics.chosen_score)} | Length ${formatCompact(frame.metrics.filament_length)}</span>
            <span>Linearity ${formatCompact(frame.metrics.chosen_linearity)} | MSD-ready centroid track</span>
          </div>
        </button>
      `
    )
    .join("");

  elements.notableVolumes.querySelectorAll("[data-time-index]").forEach((button) => {
    button.addEventListener("click", () => selectFrameByTimeIndex(Number(button.dataset.timeIndex)));
  });
}

function stopPlayback() {
  if (state.playHandle) {
    window.clearInterval(state.playHandle);
    state.playHandle = null;
  }
  state.playing = false;
  elements.playToggle.textContent = "Play";
}

function startPlayback() {
  stopPlayback();
  state.playing = true;
  elements.playToggle.textContent = "Pause";
  state.playHandle = window.setInterval(() => {
    const nextIndex = (state.currentFrameIndex + 1) % currentTrack().frames.length;
    selectFrame(nextIndex, { keepPlayback: true });
  }, 260);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function resetView() {
  state.rotationAzimuth = -0.85;
  state.rotationElevation = 0.5;
  renderVolumeCanvas(currentFrame());
}

function selectTrack(index) {
  state.currentTrackIndex = Math.max(0, Math.min(index, state.data.tracks.length - 1));
  state.currentFrameIndex = 0;
  elements.trackSelect.value = String(state.currentTrackIndex);
  elements.volumeSlider.max = String(currentTrack().frames.length - 1);
  elements.volumeSlider.value = "0";
  renderSummary();
  renderMsd();
  renderNotables();
  renderCurrentFrame();
  stopPlayback();
}

function selectFrameByTimeIndex(timeIndex, options = {}) {
  const frames = currentTrack().frames;
  const bestIndex = frames.reduce((best, frame, index) => {
    const bestDelta = Math.abs(frames[best].time_index - timeIndex);
    const candidateDelta = Math.abs(frame.time_index - timeIndex);
    return candidateDelta < bestDelta ? index : best;
  }, 0);
  selectFrame(bestIndex, options);
}

function selectFrame(index, { keepPlayback = false } = {}) {
  const clamped = Math.max(0, Math.min(index, currentTrack().frames.length - 1));
  state.currentFrameIndex = clamped;
  elements.volumeSlider.value = String(clamped);
  renderCurrentFrame();
  if (!keepPlayback) {
    stopPlayback();
  }
}

function renderCurrentFrame() {
  const frame = currentFrame();
  updateHeader(frame);
  renderVolumeMetrics(frame);
  renderTimeline();
  renderOrientationChart();
  renderDiagnostics(frame);
  renderVolumeCanvas(frame);
  renderProjections(frame);
}

function attach3DInteraction() {
  const canvas = elements.volumeCanvas;
  canvas.style.touchAction = "none";

  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.lastPointer = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging || !state.lastPointer) {
      return;
    }
    const deltaX = event.clientX - state.lastPointer.x;
    const deltaY = event.clientY - state.lastPointer.y;
    state.rotationAzimuth += deltaX * 0.012;
    state.rotationElevation = Math.max(-1.2, Math.min(1.2, state.rotationElevation + deltaY * 0.012));
    state.lastPointer = { x: event.clientX, y: event.clientY };
    renderVolumeCanvas(currentFrame());
  });

  const release = () => {
    state.dragging = false;
    state.lastPointer = null;
  };

  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("pointerleave", release);
}

function attachEvents() {
  elements.trackSelect.addEventListener("change", (event) => {
    selectTrack(Number(event.target.value));
  });
  elements.metricSelect.addEventListener("change", (event) => {
    state.metricKey = event.target.value;
    renderTimeline();
  });
  elements.playToggle.addEventListener("click", togglePlayback);
  elements.resetView.addEventListener("click", resetView);
  elements.volumeSlider.addEventListener("input", (event) => {
    selectFrame(Number(event.target.value));
  });
  elements.toggleSampled.addEventListener("change", (event) => {
    state.showSampled = event.target.checked;
    renderCurrentFrame();
  });
  elements.toggleRecovered.addEventListener("change", (event) => {
    state.showRecovered = event.target.checked;
    renderCurrentFrame();
  });
  elements.toggleBackbone.addEventListener("change", (event) => {
    state.showBackbone = event.target.checked;
    renderCurrentFrame();
  });
  elements.toggleAxis.addEventListener("change", (event) => {
    state.showAxis = event.target.checked;
    renderCurrentFrame();
  });
  window.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopPlayback();
    }
  });
  attach3DInteraction();
}

function initializeUi() {
  renderMethod();
  renderTrackSelect();
  renderMetricOptions();
  elements.volumeSlider.max = String(currentTrack().frames.length - 1);
  attachEvents();
  renderSummary();
  renderMsd();
  renderNotables();
  renderCurrentFrame();
}

async function main() {
  try {
    state.data = await loadData();
    initializeUi();
  } catch (error) {
    elements.volumeTitle.textContent = "Failed to load analysis";
    elements.volumeStatus.textContent = "Error";
    elements.volumeScore.textContent = error.message;
    console.error(error);
  }
}

main();