const state = {
  release: null,
  metrics: [],
  prompts: [],
  promptPage: 1,
  promptPages: 1,
  promptSort: "index",
  activePrompt: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const resource = (path) => new URL(path.replace(/^\/+/, ""), new URL(".", document.baseURI)).toString();
const number = (value, digits = 2) =>
  value === null || value === undefined || value === "" ? "—" : Number(value).toFixed(digits);

async function api(path) {
  const response = await fetch(resource(path));
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function optionValues(name, values) {
  const select = $(`[name="${name}"]`);
  const current = select.value;
  [...new Set(values)].sort((a, b) => Number(a) - Number(b)).forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
  select.value = current;
}

function applyDeepLink() {
  const params = new URLSearchParams(location.search);
  ["steps", "guidance", "language", "method"].forEach((name) => {
    if (!params.has(name)) return;
    const select = $(`[name="${name}"]`);
    const requested = params.get(name);
    const numericMatch = [...select.options].find(
      (option) => name === "guidance" && Number(option.value) === Number(requested)
    );
    select.value = numericMatch ? numericMatch.value : requested;
  });
  return params.get("prompt");
}

function syncUrl() {
  const params = new URLSearchParams();
  new FormData($("#filters")).forEach((value, key) => value && params.set(key, value));
  if (state.activePrompt) params.set("prompt", state.activePrompt.index);
  history.replaceState(null, "", `${location.pathname}?${params}`);
}

function renderSummary() {
  const accelerated = state.metrics.filter((row) => row.method !== "off");
  const speedups = accelerated.map((row) => Number(row.paired_speedup)).filter(Number.isFinite);
  const memories = state.metrics.map((row) => Number(row.peak_allocated_gib)).filter(Number.isFinite);
  $("#bestSpeedup").textContent = speedups.length ? `${Math.max(...speedups).toFixed(3)}×` : "—";
  $("#bestMemory").textContent = memories.length ? `${Math.min(...memories).toFixed(1)} GiB` : "—";
  $("#caseCount").textContent = String(state.metrics.length);
  const contract = state.release.contract;
  $("#duration").textContent = `${contract.duration_seconds}s · ${contract.num_frames}f`;
}

function renderTable() {
  const body = $("#metricsBody");
  body.replaceChildren();
  state.metrics.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.method}</td><td>${row.steps}</td><td>${row.guidance}</td><td>${row.language || "overall"}</td>
      <td>${number(row.latency_seconds)}s</td><td>${number(row.paired_speedup, 3)}×</td>
      <td>${number(row.latent_relative_l2, 4)}</td><td>${number(100 * Number(row.cfg_reuse_ratio || 0), 1)}%</td>
      <td>${number(row.peak_allocated_gib, 1)}</td>`;
    tr.addEventListener("click", () => {
      $("#detailsText").textContent = JSON.stringify(row, null, 2);
      $("#details").showModal();
    });
    body.append(tr);
  });
}

function renderPareto() {
  const svg = $("#pareto");
  svg.replaceChildren();
  const rows = state.metrics.filter(
    (row) => Number.isFinite(Number(row.paired_speedup)) && Number.isFinite(Number(row.latent_relative_l2))
  );
  if (!rows.length) return;
  const width = 900, height = 310, left = 70, right = 25, top = 22, bottom = 48;
  const xs = rows.map((row) => Number(row.latent_relative_l2));
  const ys = rows.map((row) => Number(row.paired_speedup));
  const xmax = Math.max(...xs, 0.01) * 1.08;
  const ymin = Math.min(...ys, 0.95) * 0.98;
  const ymax = Math.max(...ys, 1.01) * 1.02;
  const x = (value) => left + (value / xmax) * (width - left - right);
  const y = (value) => top + ((ymax - value) / (ymax - ymin)) * (height - top - bottom);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    svg.append(node);
    return node;
  };
  for (let i = 0; i <= 5; i++) {
    const xv = (xmax * i) / 5;
    const yv = ymin + ((ymax - ymin) * i) / 5;
    add("line", { x1: x(xv), y1: top, x2: x(xv), y2: height - bottom, class: "grid" });
    add("text", { x: x(xv), y: height - bottom + 20, "text-anchor": "middle", class: "tick" }, xv.toFixed(2));
    add("line", { x1: left, y1: y(yv), x2: width - right, y2: y(yv), class: "grid" });
    add("text", { x: left - 10, y: y(yv) + 4, "text-anchor": "end", class: "tick" }, `${yv.toFixed(2)}×`);
  }
  add("text", { x: (left + width - right) / 2, y: height - 8, "text-anchor": "middle", class: "chart-label" }, "Latent relative L2 (lower is better)");
  add("text", { x: 16, y: height / 2, transform: `rotate(-90 16 ${height / 2})`, "text-anchor": "middle", class: "chart-label" }, "Paired speedup (×, higher is better)");
  rows.forEach((row) => {
    const point = add("circle", {
      cx: x(Number(row.latent_relative_l2)),
      cy: y(Number(row.paired_speedup)),
      r: 6,
      class: `point ${row.method === "off" ? "exact" : ""}`,
    });
    point.append(document.createElementNS("http://www.w3.org/2000/svg", "title"));
    point.lastChild.textContent = `${row.method} · ${row.steps} steps · CFG ${row.guidance}`;
  });
}

async function loadMetrics() {
  const params = new URLSearchParams(new FormData($("#filters")));
  [...params].forEach(([key, value]) => !value && params.delete(key));
  state.metrics = (await api(`/api/metrics?${params}`)).metrics;
  renderSummary();
  renderTable();
  renderPareto();
}

function renderPromptList() {
  const list = $("#promptList");
  list.replaceChildren();
  state.prompts.forEach((prompt) => {
    const button = document.createElement("button");
    button.className = `prompt-button ${state.activePrompt?.index === prompt.index ? "active" : ""}`;
    button.innerHTML = `<strong>${prompt.pair_id} / ${prompt.language.toUpperCase()}</strong><small>seed ${prompt.seed} · index ${prompt.index}</small>`;
    button.addEventListener("click", () => selectPrompt(prompt));
    list.append(button);
  });
  $("#pageLabel").textContent = `${state.promptPage} / ${state.promptPages}`;
}

async function loadPrompts(targetIndex = null) {
  const params = new URLSearchParams({
    page: state.promptPage,
    page_size: 24,
    query: $("#promptQuery").value,
    language: $('[name="language"]').value,
    sort: state.promptSort,
  });
  const payload = await api(`/api/prompts?${params}`);
  state.prompts = payload.prompts;
  state.promptPages = Math.max(1, Math.ceil(payload.total / payload.page_size));
  renderPromptList();
  const target = targetIndex && state.prompts.find((prompt) => String(prompt.index) === String(targetIndex));
  if (target) await selectPrompt(target);
  else if (targetIndex) await selectPrompt(await api(`/api/prompt/${targetIndex}`));
  else if (!state.activePrompt && state.prompts.length) await selectPrompt(state.prompts[0]);
}

async function selectPrompt(prompt) {
  state.activePrompt = await api(`/api/prompt/${prompt.index}?language=${prompt.language}`);
  renderPromptList();
  $("#promptTitle").textContent = `${state.activePrompt.pair_id} · ${state.activePrompt.language.toUpperCase()} · seed ${state.activePrompt.seed}`;
  $("#promptText").textContent = state.activePrompt.prompt;
  const steps = $('[name="steps"]').value;
  const guidance = $('[name="guidance"]').value;
  const videos = (state.activePrompt.videos || []).filter(
    (video) => (!steps || String(video.steps) === steps) && (!guidance || String(video.guidance) === guidance)
  );
  videos.sort((a, b) => Number(Boolean(b.exact)) - Number(Boolean(a.exact)) || a.method.localeCompare(b.method));
  const wall = $("#videoWall");
  wall.replaceChildren();
  videos.forEach((item) => {
    const article = document.createElement("article");
    article.className = `video-card ${item.exact ? "exact" : ""}`;
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = $("#muteAll").checked;
    video.loop = $("#loopAll").checked;
    video.playsInline = true;
    video.src = resource(`media/${encodeURI(item.path)}`);
    if (item.poster_path) video.poster = resource(`media/${encodeURI(item.poster_path)}`);
    article.append(video);
    const meta = document.createElement("div");
    meta.className = "video-meta";
    meta.innerHTML = `<strong>${item.exact ? "Exact" : item.method}</strong><span>${number(item.speedup, 3)}×</span><small>${item.steps} steps · CFG ${item.guidance}</small><small>L2 ${number(item.latent_relative_l2, 4)}</small>`;
    article.append(meta);
    wall.append(article);
  });
  syncUrl();
}

function videos() { return $$("#videoWall video"); }
$("#playAll").addEventListener("click", () => videos().forEach((video) => video.play()));
$("#pauseAll").addEventListener("click", () => videos().forEach((video) => video.pause()));
$("#loopAll").addEventListener("change", (event) => videos().forEach((video) => { video.loop = event.target.checked; }));
$("#muteAll").addEventListener("change", (event) => videos().forEach((video) => { video.muted = event.target.checked; }));
$("#seekAll").addEventListener("input", (event) => {
  videos().forEach((video) => {
    if (Number.isFinite(video.duration)) video.currentTime = (Number(event.target.value) / 1000) * video.duration;
  });
});
setInterval(() => {
  const first = videos()[0];
  if (first && Number.isFinite(first.duration) && !$("#seekAll").matches(":active")) {
    $("#seekAll").value = Math.round((first.currentTime / first.duration) * 1000);
    videos().slice(1).forEach((video) => {
      if (Math.abs(video.currentTime - first.currentTime) > 0.12) video.currentTime = first.currentTime;
    });
  }
}, 150);

$("#filters").addEventListener("submit", async (event) => {
  event.preventDefault();
  await Promise.all([loadMetrics(), loadPrompts()]);
  if (state.activePrompt) await selectPrompt(state.activePrompt);
  syncUrl();
});
$("#promptQuery").addEventListener("change", () => { state.promptPage = 1; loadPrompts(); });
$("#worstPrompt").addEventListener("click", (event) => {
  state.promptSort = state.promptSort === "worst" ? "index" : "worst";
  event.target.textContent = state.promptSort === "worst" ? "Index order" : "Worst-case order";
  state.promptPage = 1;
  loadPrompts();
});
$("#prevPrompt").addEventListener("click", () => { if (state.promptPage > 1) { state.promptPage--; loadPrompts(); } });
$("#nextPrompt").addEventListener("click", () => { if (state.promptPage < state.promptPages) { state.promptPage++; loadPrompts(); } });
$("#closeDetails").addEventListener("click", () => $("#details").close());
$("#downloadCsv").addEventListener("click", () => {
  const fields = ["method", "steps", "guidance", "language", "latency_seconds", "paired_speedup", "latent_relative_l2", "cfg_reuse_ratio", "peak_allocated_gib"];
  const csv = [fields, ...state.metrics.map((row) => fields.map((field) => row[field] ?? ""))]
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: "leo2-acceleration-metrics.csv" });
  link.click();
  URL.revokeObjectURL(link.href);
});

async function initialize() {
  const health = await api("/api/health");
  state.release = await api("/api/release");
  $("#health").textContent = health.status;
  $("#releaseId").textContent = health.release_id;
  const c = state.release.contract;
  $("#contract").textContent = `${c.width}×${c.height} · ${c.duration_seconds}s / ${c.num_frames} frames @ ${c.fps} fps · shift ${c.shift} · CP${c.cp}/FSDP${c.fsdp}`;
  optionValues("steps", state.release.filters.steps);
  optionValues("guidance", state.release.filters.guidance);
  optionValues("method", state.release.filters.methods);
  const targetPrompt = applyDeepLink();
  await Promise.all([loadMetrics(), loadPrompts(targetPrompt)]);
  const figures = $("#figures");
  (state.release.figures || []).forEach((figure) => {
    const node = document.createElement("figure");
    node.innerHTML = `<img loading="lazy" src="${resource(`media/${encodeURI(figure.path)}`)}" alt="${figure.title}"><figcaption>${figure.title}</figcaption>`;
    figures.append(node);
  });
  $("#figuresSection").hidden = !figures.children.length;
}

initialize().catch((error) => {
  $("#health").textContent = "release error";
  $("#contract").textContent = error.message;
  throw error;
});
