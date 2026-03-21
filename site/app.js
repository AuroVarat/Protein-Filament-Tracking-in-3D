const METRIC_OPTIONS = [
  { key: "filament_length", label: "Filament length", color: "#0d8b8b" },
  { key: "n_recovered_points", label: "Recovered points", color: "#ef7b45" },
  { key: "chosen_linearity", label: "Linearity", color: "#355c7d" },
  { key: "chosen_linear_density", label: "Linear density", color: "#5a7d2b" },
  { key: "chosen_eps", label: "Chosen epsilon", color: "#8d5a97" },
  { key: "tortuosity", label: "Tortuosity", color: "#d1495b" },
  { key: "orientation_deg", label: "Orientation", color: "#2a9d8f" },
];

const state = {
  data: null,
  currentFrameIndex: 0,
  metricKey: "filament_length",
  playing: false,
  showPoints: true,
  showBackbone: true,
  showAxis: true,
  playHandle: null,
  renderToken: 0,
};

const elements = {
  summaryCards: document.querySelector("#summary-cards"),
  frameTitle: document.querySelector("#frame-title"),
  frameStatus: document.querySelector("#frame-status"),
  frameScore: document.querySelector("#frame-score"),
  frameCanvas: document.querySelector("#frame-canvas"),
  frameSlider: document.querySelector("#frame-slider"),
  playToggle: document.querySelector("#play-toggle"),
  togglePoints: document.querySelector("#toggle-points"),
  toggleBackbone: document.querySelector("#toggle-backbone"),
  toggleAxis: document.querySelector("#toggle-axis"),
  frameMetrics: document.querySelector("#frame-metrics"),
  metricSelect: document.querySelector("#metric-select"),
  timelineChart: document.querySelector("#timeline-chart"),
  componentsChart: document.querySelector("#components-chart"),
  scoreChart: document.querySelector("#score-chart"),
  msdChart: document.querySelector("#msd-chart"),
  methodPoints: document.querySelector("#method-points"),
  parameterList: document.querySelector("#parameter-list"),
  notableFrames: document.querySelector("#notable-frames"),
};

const imageCache = new Map();

async function loadData() {
  const response = await fetch("./data/tda_timeline.json");
  if (!response.ok) {
    throw new Error(`Failed to load analysis JSON: ${response.status}`);
  }
  return response.json();
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Not detected";
  }
  return Number(value).toFixed(digits);
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

function metricLabel(key) {
  return METRIC_OPTIONS.find((item) => item.key === key)?.label ?? key;
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

function frameDimensions() {
  return state.data?.metadata?.frame_dimensions ?? { width: 128, height: 128 };
}

function drawPointSet(ctx, points, width, height, color, radius) {
  if (!points?.length) {
    return;
  }
  const dims = frameDimensions();
  ctx.fillStyle = color;
  for (const [x, y] of points) {
    ctx.beginPath();
    ctx.arc((x / dims.width) * width, (y / dims.height) * height, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawPolyline(ctx, points, width, height, color, lineWidth) {
  if (!points?.length) {
    return;
  }
  const dims = frameDimensions();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    const scaledX = (x / dims.width) * width;
    const scaledY = (y / dims.height) * height;
    if (index === 0) {
      ctx.moveTo(scaledX, scaledY);
    } else {
      ctx.lineTo(scaledX, scaledY);
    }
  });
  ctx.stroke();
}

async function renderFrameCanvas(frame) {
  const token = ++state.renderToken;
  const canvas = elements.frameCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const image = await loadImage(frame.image_path);
  if (token !== state.renderToken) {
    return;
  }
  ctx.drawImage(image, 0, 0, width, height);

  if (state.showPoints) {
    drawPointSet(ctx, frame.filament_points, width, height, "rgba(13, 139, 139, 0.88)", 3.6);
  }
  if (state.showBackbone) {
    drawPolyline(ctx, frame.backbone_path, width, height, "rgba(239, 123, 69, 0.95)", 4);
  }
  if (state.showAxis) {
    drawPolyline(ctx, frame.principal_axis, width, height, "rgba(244, 211, 94, 0.92)", 3);
  }

  if (!frame.has_filament) {
    ctx.fillStyle = "rgba(13, 20, 18, 0.7)";
    ctx.fillRect(0, height - 92, width, 92);
    ctx.fillStyle = "rgba(255, 250, 241, 0.95)";
    ctx.font = "600 22px Avenir Next, Segoe UI, sans-serif";
    ctx.fillText("No dense filament passed the thresholds in this frame.", 24, height - 38);
  }
}

function renderSummary() {
  const summary = state.data.summary;
  elements.summaryCards.innerHTML = [
    summaryCard("Frames with filament", `${summary.frames_with_filament} / ${summary.frames_total}`, "Dense elongated component detected"),
    summaryCard("Detection rate", `${formatNumber(summary.detection_rate * 100, 1)}%`, "Across the full timespan"),
    summaryCard("Mean length", formatCompact(summary.mean_length), "Recovered backbone length"),
    summaryCard("Max MSD", formatCompact(summary.max_msd), "Centroid displacement scale"),
  ].join("");
}

function renderFrameMetrics(frame) {
  const metrics = frame.metrics;
  elements.frameMetrics.innerHTML = [
    metricCard("Recovered points", `${metrics.n_recovered_points}`, "Weighted samples retained"),
    metricCard("Filament length", formatCompact(metrics.filament_length), "Backbone estimate"),
    metricCard("Linearity", formatCompact(metrics.chosen_linearity), "Major span over minor span"),
    metricCard("Density", formatCompact(metrics.chosen_linear_density), "Points per unit span"),
    metricCard("Chosen epsilon", formatCompact(metrics.chosen_eps), "Selected from the sweep"),
    metricCard("Orientation", metrics.orientation_deg === null ? "Not detected" : `${formatCompact(metrics.orientation_deg)}°`, "Principal axis angle"),
  ].join("");
}

function renderMethod() {
  elements.methodPoints.innerHTML = state.data.metadata.notebook_mapping
    .map((text) => `<div class="finding-item">${text}</div>`)
    .join("");

  elements.parameterList.innerHTML = Object.entries(state.data.metadata.parameters)
    .map(([key, value]) => `<dt>${key.replaceAll("_", " ")}</dt><dd>${value}</dd>`)
    .join("");
}

function renderMetricOptions() {
  elements.metricSelect.innerHTML = METRIC_OPTIONS.map(
    (item) => `<option value="${item.key}">${item.label}</option>`
  ).join("");
  elements.metricSelect.value = state.metricKey;
}

function toPath(points, xScale, yScale) {
  const validPoints = points.filter((point) => point.y !== null && point.y !== undefined && !Number.isNaN(point.y));
  if (!validPoints.length) {
    return "";
  }

  return validPoints
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`)
    .join(" ");
}

function buildChart({
  svg,
  title,
  xValues,
  yValues,
  stroke,
  selectedX = null,
  selectedY = null,
  onClick = null,
  showArea = false,
  xLabel = "Frame",
  yLabel = "",
  markerRadius = 4.5,
  referenceX = null,
}) {
  const width = 720;
  const height = svg.viewBox.baseVal.height || 260;
  const margin = { top: 26, right: 20, bottom: 44, left: 62 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const validY = yValues.filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
  const minY = validY.length ? Math.min(...validY) : 0;
  const maxY = validY.length ? Math.max(...validY) : 1;
  const yPadding = maxY === minY ? 1 : (maxY - minY) * 0.12;
  const domainMinY = minY - yPadding;
  const domainMaxY = maxY + yPadding;

  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);

  const xScale = (value) => margin.left + ((value - minX) / Math.max(maxX - minX, 1e-6)) * innerWidth;
  const yScale = (value) =>
    margin.top + innerHeight - ((value - domainMinY) / Math.max(domainMaxY - domainMinY, 1e-6)) * innerHeight;

  const points = xValues.map((x, index) => ({ x, y: yValues[index] }));
  const path = toPath(points, xScale, yScale);

  const ticksY = 4;
  const tickValuesY = Array.from({ length: ticksY + 1 }, (_, index) => domainMinY + ((domainMaxY - domainMinY) / ticksY) * index);
  const tickValuesX = Array.from({ length: 6 }, (_, index) => minX + ((maxX - minX) / 5) * index);

  const areaPath = showArea && path
    ? `${path} L ${xScale(points[points.length - 1].x).toFixed(2)} ${(margin.top + innerHeight).toFixed(2)} L ${xScale(points[0].x).toFixed(2)} ${(margin.top + innerHeight).toFixed(2)} Z`
    : "";

  svg.innerHTML = `
    <text x="${margin.left}" y="16" fill="#61716c" font-size="12" letter-spacing="1.8" text-transform="uppercase">${title}</text>
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
    ${areaPath ? `<path d="${areaPath}" fill="url(#chart-fill-${svg.id})" opacity="0.18"></path>` : ""}
    <path d="${path}" fill="none" stroke="${stroke}" stroke-width="3.4" stroke-linecap="round" stroke-linejoin="round"></path>
    ${
      referenceX !== null
        ? `<line x1="${xScale(referenceX)}" y1="${margin.top}" x2="${xScale(referenceX)}" y2="${margin.top + innerHeight}" stroke="rgba(239, 123, 69, 0.9)" stroke-dasharray="6 6"></line>`
        : ""
    }
    ${
      selectedX !== null && selectedY !== null && !Number.isNaN(selectedY)
        ? `
          <line x1="${xScale(selectedX)}" y1="${margin.top}" x2="${xScale(selectedX)}" y2="${margin.top + innerHeight}" stroke="rgba(31, 49, 44, 0.24)" stroke-dasharray="4 6"></line>
          <circle cx="${xScale(selectedX)}" cy="${yScale(selectedY)}" r="${markerRadius}" fill="${stroke}" stroke="#fff8eb" stroke-width="2"></circle>
        `
        : ""
    }
    <text x="${width - margin.right}" y="16" fill="#61716c" font-size="11" text-anchor="end">${yLabel}</text>
    <text x="${width - margin.right}" y="${height - 16}" fill="#61716c" font-size="11" text-anchor="end">${xLabel}</text>
    <defs>
      <linearGradient id="chart-fill-${svg.id}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${stroke}"></stop>
        <stop offset="100%" stop-color="${stroke}" stop-opacity="0"></stop>
      </linearGradient>
    </defs>
    ${
      onClick
        ? `<rect x="${margin.left}" y="${margin.top}" width="${innerWidth}" height="${innerHeight}" fill="transparent" data-chart-hit="true"></rect>`
        : ""
    }
  `;

  if (onClick) {
    const hitRect = svg.querySelector("[data-chart-hit='true']");
    hitRect.addEventListener("click", (event) => {
      const bounds = svg.getBoundingClientRect();
      const relativeX = event.clientX - bounds.left;
      const value = minX + ((relativeX - margin.left) / innerWidth) * (maxX - minX);
      onClick(value);
    });
  }
}

function renderTimeline() {
  const metric = METRIC_OPTIONS.find((item) => item.key === state.metricKey);
  const xValues = state.data.time_series.map((row) => row.frame_index);
  const yValues = state.data.time_series.map((row) => row[state.metricKey]);
  const current = state.data.time_series[state.currentFrameIndex];

  buildChart({
    svg: elements.timelineChart,
    title: metric.label,
    xValues,
    yValues,
    stroke: metric.color,
    selectedX: current.frame_index,
    selectedY: current[state.metricKey],
    showArea: true,
    yLabel: metric.label,
    onClick: (value) => selectFrame(Math.round(value)),
  });
}

function renderDiagnostics(frame) {
  const diagnostics = frame.diagnostics;

  buildChart({
    svg: elements.componentsChart,
    title: "Connected components vs epsilon",
    xValues: diagnostics.eps_values,
    yValues: diagnostics.component_counts,
    stroke: "#355c7d",
    selectedX: diagnostics.chosen_eps,
    selectedY:
      diagnostics.chosen_eps === null
        ? null
        : diagnostics.component_counts[diagnostics.eps_values.findIndex((value) => value === diagnostics.chosen_eps)],
    yLabel: "Components",
    xLabel: "Epsilon",
    referenceX: diagnostics.chosen_eps,
  });

  buildChart({
    svg: elements.scoreChart,
    title: "Dense-line score vs epsilon",
    xValues: diagnostics.eps_values,
    yValues: diagnostics.dense_line_scores,
    stroke: "#ef7b45",
    selectedX: diagnostics.chosen_eps,
    selectedY:
      diagnostics.chosen_eps === null
        ? null
        : diagnostics.dense_line_scores[diagnostics.eps_values.findIndex((value) => value === diagnostics.chosen_eps)],
    yLabel: "Score",
    xLabel: "Epsilon",
    referenceX: diagnostics.chosen_eps,
  });
}

function renderMsd() {
  if (!state.data.msd.length) {
    elements.msdChart.innerHTML = `
      <text x="60" y="40" fill="#61716c" font-size="12" letter-spacing="1.8">MSD</text>
      <text x="360" y="150" fill="#61716c" font-size="15" text-anchor="middle">Need at least two detected frames to build the MSD curve.</text>
    `;
    return;
  }

  buildChart({
    svg: elements.msdChart,
    title: "Centroid MSD",
    xValues: state.data.msd.map((row) => row.lag_time),
    yValues: state.data.msd.map((row) => row.msd),
    stroke: "#0d8b8b",
    selectedX: null,
    selectedY: null,
    showArea: true,
    yLabel: "MSD",
    xLabel: "Lag",
  });
}

function renderNotableFrames() {
  const notableFrames = [...state.data.frames]
    .filter((frame) => frame.has_filament)
    .sort((a, b) => (b.metrics.chosen_score ?? -Infinity) - (a.metrics.chosen_score ?? -Infinity))
    .slice(0, 6);

  elements.notableFrames.innerHTML = notableFrames
    .map(
      (frame) => `
        <button class="notable-card" data-frame-index="${frame.frame_index}" type="button">
          <img src="${frame.image_path}" alt="Frame ${frame.frame_index}" />
          <div class="notable-copy">
            <strong>Frame ${frame.frame_index}</strong>
            <span>Score ${formatCompact(frame.metrics.chosen_score)} | Length ${formatCompact(frame.metrics.filament_length)}</span>
            <span>Linearity ${formatCompact(frame.metrics.chosen_linearity)} | Epsilon ${formatCompact(frame.metrics.chosen_eps)}</span>
          </div>
        </button>
      `
    )
    .join("");

  elements.notableFrames.querySelectorAll("[data-frame-index]").forEach((button) => {
    button.addEventListener("click", () => selectFrame(Number(button.dataset.frameIndex)));
  });
}

function updateFrameHeader(frame) {
  elements.frameTitle.textContent = `Frame ${frame.frame_index} of ${state.data.frames.length - 1}`;
  elements.frameStatus.textContent = frame.has_filament ? "Filament detected" : "No filament";
  elements.frameStatus.style.background = frame.has_filament ? "rgba(13, 139, 139, 0.14)" : "rgba(97, 113, 108, 0.14)";
  elements.frameStatus.style.color = frame.has_filament ? "#0d8b8b" : "#61716c";
  elements.frameScore.textContent = frame.metrics.chosen_score === null
    ? "Dense-line score unavailable"
    : `Dense-line score ${formatCompact(frame.metrics.chosen_score)}`;
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
    const nextIndex = (state.currentFrameIndex + 1) % state.data.frames.length;
    selectFrame(nextIndex, { keepPlayback: true });
  }, 220);
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function selectFrame(index, { keepPlayback = false } = {}) {
  const clampedIndex = Math.max(0, Math.min(index, state.data.frames.length - 1));
  state.currentFrameIndex = clampedIndex;
  elements.frameSlider.value = String(clampedIndex);
  renderCurrentFrame();
  if (!keepPlayback) {
    stopPlayback();
  }
}

function renderCurrentFrame() {
  const frame = state.data.frames[state.currentFrameIndex];
  updateFrameHeader(frame);
  renderFrameMetrics(frame);
  renderTimeline();
  renderDiagnostics(frame);
  renderFrameCanvas(frame);
}

function attachEvents() {
  elements.playToggle.addEventListener("click", togglePlayback);
  elements.frameSlider.addEventListener("input", (event) => {
    selectFrame(Number(event.target.value));
  });
  elements.metricSelect.addEventListener("change", (event) => {
    state.metricKey = event.target.value;
    renderTimeline();
  });
  elements.togglePoints.addEventListener("change", (event) => {
    state.showPoints = event.target.checked;
    renderFrameCanvas(state.data.frames[state.currentFrameIndex]);
  });
  elements.toggleBackbone.addEventListener("change", (event) => {
    state.showBackbone = event.target.checked;
    renderFrameCanvas(state.data.frames[state.currentFrameIndex]);
  });
  elements.toggleAxis.addEventListener("change", (event) => {
    state.showAxis = event.target.checked;
    renderFrameCanvas(state.data.frames[state.currentFrameIndex]);
  });
  window.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopPlayback();
    }
  });
}

function initializeUi() {
  renderSummary();
  renderMethod();
  renderMetricOptions();
  renderMsd();
  renderNotableFrames();
  elements.frameSlider.max = String(state.data.frames.length - 1);
  attachEvents();
  renderCurrentFrame();
}

async function main() {
  try {
    state.data = await loadData();
    initializeUi();
  } catch (error) {
    elements.frameTitle.textContent = "Failed to load analysis";
    elements.frameStatus.textContent = "Error";
    elements.frameScore.textContent = error.message;
    console.error(error);
  }
}

main();
