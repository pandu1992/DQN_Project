/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * main.js — rendering, training loop, UI wiring
 *
 * Features:
 *   - Real-time DQN training over the Bintulu-style channel map
 *   - Waypoint / route visit HEATMAP overlay (RouteHeatmap-style)
 *   - CSV benchmark export of per-episode metrics (Sprint 7.3/7.4)
 *   - North (L02) vs South (L08) per-lane comparison stats
 *   - Representative episode REPLAY (best / worst reward)
 * ============================================================ */

(function () {
  const { VesselEnv, ACTION_NAMES, MAP_W, MAP_H } = window.BintuluEnv;
  const { DQNAgent } = window.BintuluDQN;

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const mapCanvas = $("mapCanvas");
  const mapCtx = mapCanvas.getContext("2d");
  const chartCanvas = $("chartCanvas");
  const chartCtx = chartCanvas.getContext("2d");

  const btnTrain = $("btnTrain");
  const btnPause = $("btnPause");
  const btnReset = $("btnReset");
  const btnGreedy = $("btnGreedy");
  const speedSlider = $("speedSlider");
  const speedLabel = $("speedLabel");

  // feature controls (may be null until index.html adds them)
  const btnHeatmap = $("btnHeatmap");
  const btnExport = $("btnExport");
  const btnReplayBest = $("btnReplayBest");
  const btnReplayWorst = $("btnReplayWorst");

  // ---------- State ----------
  let env, agent;
  let running = false;
  let greedyMode = false;
  let episode = 0;
  let rewardHistory = [];
  let successHistory = [];
  let obs, lastQ = [0, 0, 0], lastAction = -1;
  let animFrame = null;

  // feature state
  let showHeatmap = false;
  let visitCounts = {};     // waypointId -> visits
  let maxVisit = 1;
  let episodeRecords = [];  // per-episode benchmark rows (Sprint 7.3/7.4)
  let laneStats = {};       // lane -> {n, success, reward, steps}
  let bestEpisode = null;   // {reward, trajectory, start, goal, lane}
  let worstEpisode = null;
  let replay = null;        // {trajectory, index, label} while replaying

  // current-episode trajectory accumulator
  let curTrajectory = [];
  let curActionCounts = [0, 0, 0];
  let curInvalid = 0;

  const TOTAL_STEPS = 50000;

  function init() {
    env = new VesselEnv(42);
    agent = new DQNAgent(env.obsDim, env.nActions, { totalSteps: TOTAL_STEPS });
    episode = 0;
    rewardHistory = [];
    successHistory = [];
    visitCounts = {};
    maxVisit = 1;
    episodeRecords = [];
    laneStats = {};
    bestEpisode = null;
    worstEpisode = null;
    replay = null;
    curTrajectory = [];
    curActionCounts = [0, 0, 0];
    curInvalid = 0;
    obs = env.reset();
    lastQ = agent.qValues(obs);
    lastAction = -1;
    startNewEpisodeTracking();
    updateMissionPanel();
    render();
    updateMetrics();
    updateFeaturePanels();
    drawChart();
  }

  function startNewEpisodeTracking() {
    curTrajectory = [env.currentWp];
    curActionCounts = [0, 0, 0];
    curInvalid = 0;
  }

  function recordVisit(wpId) {
    visitCounts[wpId] = (visitCounts[wpId] || 0) + 1;
    if (visitCounts[wpId] > maxVisit) maxVisit = visitCounts[wpId];
  }

  // ---------- One environment step ----------
  function stepOnce() {
    const action = agent.act(obs, greedyMode);
    lastAction = action;
    lastQ = agent.qValues(obs);
    curActionCounts[action]++;
    const { obs: nextObs, reward, terminated, truncated, info } = env.step(action);
    if (info.invalid) curInvalid++;
    if (!greedyMode) {
      agent.observe(obs, action, reward, nextObs, terminated);
    }
    obs = nextObs;

    // track visit + trajectory
    recordVisit(env.currentWp);
    curTrajectory.push(env.currentWp);

    if (terminated || truncated) {
      episode++;
      const totalR = env.totalReward;
      rewardHistory.push(totalR);
      successHistory.push(info.reachedGoal ? 1 : 0);
      if (rewardHistory.length > 5000) rewardHistory.shift();
      if (successHistory.length > 5000) successHistory.shift();

      finalizeEpisode(totalR, info);

      // greedy demo runs a single episode then stops
      if (greedyMode) {
        greedyMode = false;
        running = false;
        setButtons();
      }
      obs = env.reset();
      startNewEpisodeTracking();
      updateMissionPanel();
    }
  }

  function finalizeEpisode(totalR, info) {
    const lane = env.states[env.startWp].lane;
    const steps = curTrajectory.length - 1;
    const success = info.reachedGoal ? 1 : 0;

    // per-episode record (Sprint 7.3/7.4 benchmark row)
    episodeRecords.push({
      episode,
      lane,
      start: env.startWp,
      goal: env.goalWp,
      success,
      reward: +totalR.toFixed(3),
      steps,
      wait: curActionCounts[0],
      forward: curActionCounts[1],
      backward: curActionCounts[2],
      invalid: curInvalid,
      plannedDistance: +env.plannedDistance.toFixed(2),
      plannedCost: +env.plannedCost.toFixed(2),
      epsilon: +agent.epsilon.toFixed(4),
    });
    if (episodeRecords.length > 20000) episodeRecords.shift();

    // per-lane aggregate
    if (!laneStats[lane]) laneStats[lane] = { n: 0, success: 0, reward: 0, steps: 0 };
    const ls = laneStats[lane];
    ls.n++;
    ls.success += success;
    ls.reward += totalR;
    ls.steps += steps;

    // best / worst episode by reward (store trajectory for replay)
    const snapshot = {
      episode,
      reward: totalR,
      trajectory: curTrajectory.slice(),
      planned: env.plannedPath.slice(),
      start: env.startWp,
      goal: env.goalWp,
      lane,
      success,
    };
    if (!bestEpisode || totalR > bestEpisode.reward) bestEpisode = snapshot;
    if (!worstEpisode || totalR < worstEpisode.reward) worstEpisode = snapshot;
  }

  // ---------- Main loop ----------
  function loop() {
    if (!running) return;

    if (replay) {
      // replay mode: advance one trajectory node per frame (visual)
      replay.index++;
      if (replay.index >= replay.trajectory.length) {
        running = false;
        setButtons();
      }
      render();
      animFrame = requestAnimationFrame(loop);
      return;
    }

    const stepsPerFrame = parseInt(speedSlider.value, 10);
    for (let i = 0; i < stepsPerFrame; i++) {
      stepOnce();
      if (!running) break; // greedy episode may have stopped mid-frame
    }
    render();
    updateMetrics();
    updateFeaturePanels();
    drawChart();
    animFrame = requestAnimationFrame(loop);
  }

  // ============================================================
  // RENDERING
  // ============================================================
  function laneColor(lane) {
    return { L02: "#2f6db8", L08: "#8a5cc0", L01: "#3aa06a" }[lane] || "#456";
  }

  // YlOrRd-ish ramp for heatmap intensity t in [0,1]
  function heatColor(t, alpha) {
    t = Math.max(0, Math.min(1, t));
    const stops = [
      [255, 255, 178],
      [254, 204, 92],
      [253, 141, 60],
      [240, 59, 32],
      [189, 0, 38],
    ];
    const x = t * (stops.length - 1);
    const i = Math.floor(x);
    const f = x - i;
    const a = stops[i];
    const b = stops[Math.min(i + 1, stops.length - 1)];
    const r = Math.round(a[0] + (b[0] - a[0]) * f);
    const g = Math.round(a[1] + (b[1] - a[1]) * f);
    const bl = Math.round(a[2] + (b[2] - a[2]) * f);
    return `rgba(${r},${g},${bl},${alpha})`;
  }

  function render() {
    const ctx = mapCtx;
    ctx.clearRect(0, 0, MAP_W, MAP_H);

    // water background gradient
    const g = ctx.createLinearGradient(0, 0, 0, MAP_H);
    g.addColorStop(0, "#0a1c33");
    g.addColorStop(1, "#071324");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, MAP_W, MAP_H);

    // land masses (simple stylized shores top-left & bottom-right)
    ctx.fillStyle = "#14202f";
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(260, 0); ctx.quadraticCurveTo(120, 70, 0, 90); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(MAP_W, MAP_H); ctx.lineTo(MAP_W, MAP_H - 150);
    ctx.quadraticCurveTo(MAP_W - 120, MAP_H - 60, MAP_W - 240, MAP_H); ctx.closePath(); ctx.fill();

    // ---- graph edges ----
    for (const id of Object.keys(env.graph)) {
      const a = env.states[id];
      for (const e of env.graph[id]) {
        const b = env.states[e.neighbor];
        if (!b) continue;
        if (e.edge_type === "virtual") {
          ctx.strokeStyle = "rgba(255,159,67,.25)";
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1;
        } else {
          ctx.strokeStyle = "rgba(90,120,170,.28)";
          ctx.setLineDash([]);
          ctx.lineWidth = 1;
        }
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // ---- HEATMAP overlay (waypoint visit frequency) ----
    if (showHeatmap) {
      for (const id of Object.keys(env.states)) {
        const s = env.states[id];
        const v = visitCounts[id] || 0;
        if (v <= 0) continue;
        const t = v / maxVisit;
        const radius = 8 + t * 20;
        ctx.fillStyle = heatColor(t, 0.55);
        ctx.beginPath();
        ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ---- planned path ----
    drawPath(env.plannedPath, "#37d67a", 3, 1);

    // ---- actual / replay path ----
    if (replay) {
      const shown = replay.trajectory.slice(0, replay.index + 1);
      drawPath(replay.planned, "#37d67a", 3, 0.7);
      drawPath(shown, "#ff9f43", 2.6, 0.95);
    } else {
      drawPath(env.actualPath, "#ff9f43", 2.5, 0.9);
    }

    // ---- waypoints ----
    for (const id of Object.keys(env.states)) {
      const s = env.states[id];
      ctx.fillStyle = laneColor(s.lane);
      ctx.beginPath();
      ctx.arc(s.x, s.y, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // ---- buoys ----
    for (const bo of env.buoys) {
      ctx.fillStyle = bo.color === "GREEN" ? "#37d67a" : "#ff5d5d";
      ctx.beginPath();
      ctx.moveTo(bo.x, bo.y - 5);
      ctx.lineTo(bo.x + 4, bo.y + 4);
      ctx.lineTo(bo.x - 4, bo.y + 4);
      ctx.closePath();
      ctx.fill();
    }

    // ---- start & goal markers (use replay's mission if replaying) ----
    const startId = replay ? replay.start : env.startWp;
    const goalId = replay ? replay.goal : env.goalWp;
    markNode(startId, "#37d67a", "S");
    markNode(goalId, "#ffd54a", "G");

    // ---- ship ----
    let shipId;
    if (replay) {
      shipId = replay.trajectory[Math.min(replay.index, replay.trajectory.length - 1)];
    } else {
      shipId = env.currentWp;
    }
    const sh = env.states[shipId];
    if (sh) {
      const ang = (sh.heading || 0) * (Math.PI / 180);
      ctx.save();
      ctx.translate(sh.x, sh.y);
      ctx.rotate(ang);
      ctx.fillStyle = "#4da3ff";
      ctx.strokeStyle = "#cfe6ff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(11, 0);
      ctx.lineTo(-7, -6);
      ctx.lineTo(-4, 0);
      ctx.lineTo(-7, 6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // ---- replay banner ----
    if (replay) {
      ctx.fillStyle = "rgba(4,18,31,.75)";
      ctx.fillRect(10, 10, 260, 30);
      ctx.fillStyle = "#ffd54a";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(
        `REPLAY: ${replay.label} · reward ${replay.reward.toFixed(1)}`,
        20, 26
      );
    }

    // ---- heatmap legend ----
    if (showHeatmap && !replay) {
      const lx = MAP_W - 150, ly = 16, lw = 120, lh = 10;
      for (let i = 0; i < lw; i++) {
        mapCtx.fillStyle = heatColor(i / lw, 0.9);
        mapCtx.fillRect(lx + i, ly, 1, lh);
      }
      mapCtx.fillStyle = "#cfe0ff";
      mapCtx.font = "10px sans-serif";
      mapCtx.textAlign = "left";
      mapCtx.fillText("low", lx, ly + 22);
      mapCtx.textAlign = "right";
      mapCtx.fillText(`high (max ${maxVisit})`, lx + lw, ly + 22);
      mapCtx.textAlign = "center";
      mapCtx.fillText("waypoint visits", lx + lw / 2, ly - 4);
    }
  }

  function drawPath(pathIds, color, width, alpha) {
    if (!pathIds || pathIds.length < 2) return;
    mapCtx.save();
    mapCtx.globalAlpha = alpha;
    mapCtx.strokeStyle = color;
    mapCtx.lineWidth = width;
    mapCtx.lineJoin = "round";
    mapCtx.beginPath();
    let started = false;
    for (const id of pathIds) {
      const s = env.states[id];
      if (!s) continue;
      if (!started) {
        mapCtx.moveTo(s.x, s.y);
        started = true;
      } else {
        mapCtx.lineTo(s.x, s.y);
      }
    }
    mapCtx.stroke();
    mapCtx.restore();
  }

  function markNode(id, color, label) {
    const s = env.states[id];
    if (!s) return;
    mapCtx.fillStyle = color;
    mapCtx.beginPath();
    mapCtx.arc(s.x, s.y, 8, 0, Math.PI * 2);
    mapCtx.fill();
    mapCtx.fillStyle = "#04121f";
    mapCtx.font = "bold 10px sans-serif";
    mapCtx.textAlign = "center";
    mapCtx.textBaseline = "middle";
    mapCtx.fillText(label, s.x, s.y);
  }

  // ============================================================
  // CHART (episode reward)
  // ============================================================
  function drawChart() {
    const ctx = chartCtx;
    const W = chartCanvas.width;
    const H = chartCanvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#071324";
    ctx.fillRect(0, 0, W, H);

    if (rewardHistory.length < 2) {
      ctx.fillStyle = "#8aa0c6";
      ctx.font = "12px sans-serif";
      ctx.fillText("Reward per episode will appear here once training starts…", 14, 24);
      return;
    }

    const data = rewardHistory;
    let min = Math.min(...data);
    let max = Math.max(...data);
    if (min === max) { min -= 1; max += 1; }
    const pad = 26;

    const yFor = (v) => H - pad - ((v - min) / (max - min)) * (H - pad * 2);
    ctx.strokeStyle = "rgba(138,160,198,.25)";
    ctx.lineWidth = 1;
    if (min < 0 && max > 0) {
      ctx.beginPath();
      ctx.moveTo(pad, yFor(0));
      ctx.lineTo(W - pad, yFor(0));
      ctx.stroke();
    }

    // raw reward line
    ctx.strokeStyle = "rgba(255,159,67,.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + (i / (data.length - 1)) * (W - pad * 2);
      const y = yFor(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // moving average (window 20)
    const win = 20;
    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < data.length; i++) {
      const s = Math.max(0, i - win + 1);
      let sum = 0;
      for (let k = s; k <= i; k++) sum += data[k];
      const avg = sum / (i - s + 1);
      const x = pad + (i / (data.length - 1)) * (W - pad * 2);
      const y = yFor(avg);
      started ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), (started = true));
    }
    ctx.stroke();

    ctx.fillStyle = "#8aa0c6";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(max.toFixed(0), 4, yFor(max) + 3);
    ctx.fillText(min.toFixed(0), 4, yFor(min) + 3);
    $("chartLabel").textContent = `${data.length} episodes · orange=raw · blue=MA20`;
  }

  // ============================================================
  // METRICS + Q-BARS
  // ============================================================
  function updateMetrics() {
    $("mEpisode").textContent = episode;
    $("mSteps").textContent = agent.stepCount.toLocaleString();
    $("mEps").textContent = agent.epsilon.toFixed(3);
    $("mLoss").textContent = agent.lastLoss ? agent.lastLoss.toFixed(4) : "—";
    $("mBuffer").textContent = agent.buffer.size.toLocaleString();
    $("mReward").textContent = env.totalReward.toFixed(1);

    const last100 = rewardHistory.slice(-100);
    if (last100.length) {
      const avg = last100.reduce((a, b) => a + b, 0) / last100.length;
      $("mAvg").textContent = avg.toFixed(1);
    }
    const succ100 = successHistory.slice(-100);
    if (succ100.length) {
      const rate = (100 * succ100.reduce((a, b) => a + b, 0)) / succ100.length;
      $("mSucc").textContent = rate.toFixed(0) + "%";
    }

    const maxAbs = Math.max(1, ...lastQ.map((q) => Math.abs(q)));
    for (let a = 0; a < 3; a++) {
      const bar = $("q" + a);
      const val = lastQ[a] || 0;
      const w = (Math.abs(val) / maxAbs) * 50;
      bar.style.width = w + "%";
      bar.style.left = val >= 0 ? "50%" : 50 - w + "%";
      bar.style.background = a === lastAction ? "#37d67a" : "#4da3ff";
      $("q" + a + "v").textContent = val.toFixed(2);
    }
    $("chosenAction").textContent = lastAction >= 0 ? ACTION_NAMES[lastAction] : "—";
  }

  function updateMissionPanel() {
    $("mStart").textContent = env.startWp;
    $("mGoal").textContent = env.goalWp;
    $("mLane").textContent = env.states[env.startWp].lane;
    $("mPlanDist").textContent = env.plannedDistance.toFixed(1);
    $("mPlanCost").textContent = env.plannedCost.toFixed(1);
  }

  // ============================================================
  // FEATURE PANELS: lane comparison + replay availability
  // ============================================================
  function updateFeaturePanels() {
    // North (L02) vs South (L08) comparison table
    const laneRow = (lane) => {
      const ls = laneStats[lane];
      if (!ls || ls.n === 0) return { succ: "—", reward: "—", steps: "—", n: 0 };
      return {
        succ: ((100 * ls.success) / ls.n).toFixed(0) + "%",
        reward: (ls.reward / ls.n).toFixed(1),
        steps: (ls.steps / ls.n).toFixed(1),
        n: ls.n,
      };
    };
    const north = laneRow("L02");
    const south = laneRow("L08");
    if ($("cmpNorthN")) {
      $("cmpNorthN").textContent = north.n;
      $("cmpNorthSucc").textContent = north.succ;
      $("cmpNorthReward").textContent = north.reward;
      $("cmpNorthSteps").textContent = north.steps;
      $("cmpSouthN").textContent = south.n;
      $("cmpSouthSucc").textContent = south.succ;
      $("cmpSouthReward").textContent = south.reward;
      $("cmpSouthSteps").textContent = south.steps;
    }

    // replay buttons + labels
    if (btnReplayBest) {
      btnReplayBest.disabled = !bestEpisode || running;
      btnReplayWorst.disabled = !worstEpisode || running;
      if ($("bestLabel"))
        $("bestLabel").textContent = bestEpisode
          ? `#${bestEpisode.episode} · ${bestEpisode.lane} · r=${bestEpisode.reward.toFixed(1)}`
          : "—";
      if ($("worstLabel"))
        $("worstLabel").textContent = worstEpisode
          ? `#${worstEpisode.episode} · ${worstEpisode.lane} · r=${worstEpisode.reward.toFixed(1)}`
          : "—";
    }
    if (btnExport) btnExport.disabled = episodeRecords.length === 0;
  }

  // ============================================================
  // CSV EXPORT (Sprint 7.3 / 7.4 benchmark dump)
  // ============================================================
  function exportCSV() {
    if (episodeRecords.length === 0) return;
    const cols = [
      "episode", "lane", "start", "goal", "success", "reward", "steps",
      "wait", "forward", "backward", "invalid",
      "plannedDistance", "plannedCost", "epsilon",
    ];
    const lines = [cols.join(",")];
    for (const r of episodeRecords) {
      lines.push(cols.map((c) => r[c]).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bintulu_dqn_benchmark_${episodeRecords.length}ep.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ============================================================
  // REPLAY (best / worst representative episode)
  // ============================================================
  function startReplay(which) {
    const snap = which === "best" ? bestEpisode : worstEpisode;
    if (!snap) return;
    if (animFrame) cancelAnimationFrame(animFrame);
    greedyMode = false;
    replay = {
      trajectory: snap.trajectory,
      planned: snap.planned,
      start: snap.start,
      goal: snap.goal,
      reward: snap.reward,
      label: which === "best" ? "BEST reward" : "WORST reward",
      index: 0,
    };
    running = true;
    setButtons();
    animFrame = requestAnimationFrame(loop);
  }

  // ============================================================
  // CONTROLS
  // ============================================================
  function setButtons() {
    btnTrain.disabled = running;
    btnPause.disabled = !running;
    if (btnReplayBest) {
      btnReplayBest.disabled = !bestEpisode || running;
      btnReplayWorst.disabled = !worstEpisode || running;
    }
  }

  function stopReplayIfAny() {
    if (replay) {
      replay = null;
      running = false;
    }
  }

  btnTrain.addEventListener("click", () => {
    if (running) return;
    stopReplayIfAny();
    greedyMode = false;
    running = true;
    setButtons();
    animFrame = requestAnimationFrame(loop);
  });

  btnPause.addEventListener("click", () => {
    running = false;
    replay = null;
    setButtons();
    if (animFrame) cancelAnimationFrame(animFrame);
    render();
  });

  btnReset.addEventListener("click", () => {
    running = false;
    setButtons();
    if (animFrame) cancelAnimationFrame(animFrame);
    init();
  });

  btnGreedy.addEventListener("click", () => {
    if (animFrame) cancelAnimationFrame(animFrame);
    stopReplayIfAny();
    greedyMode = true;
    running = true;
    obs = env.reset();
    startNewEpisodeTracking();
    updateMissionPanel();
    setButtons();
    animFrame = requestAnimationFrame(loop);
  });

  speedSlider.addEventListener("input", () => {
    speedLabel.textContent = speedSlider.value + "×";
  });

  if (btnHeatmap) {
    btnHeatmap.addEventListener("click", () => {
      showHeatmap = !showHeatmap;
      btnHeatmap.classList.toggle("active", showHeatmap);
      btnHeatmap.textContent = showHeatmap ? "🔥 Heatmap: ON" : "🔥 Heatmap: OFF";
      render();
    });
  }
  if (btnExport) btnExport.addEventListener("click", exportCSV);
  if (btnReplayBest) btnReplayBest.addEventListener("click", () => startReplay("best"));
  if (btnReplayWorst) btnReplayWorst.addEventListener("click", () => startReplay("worst"));

  // ---------- boot ----------
  init();

  // expose a tiny hook for headless testing
  window.__bintulu = {
    getState: () => ({
      episode,
      records: episodeRecords.length,
      lanes: Object.keys(laneStats),
      best: bestEpisode ? bestEpisode.reward : null,
      worst: worstEpisode ? worstEpisode.reward : null,
      maxVisit,
    }),
    exportCSV,
  };
})();
