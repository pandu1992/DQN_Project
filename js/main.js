/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * main.js — rendering, training loop, UI wiring
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

  // ---------- State ----------
  let env, agent;
  let running = false;
  let greedyMode = false;
  let episode = 0;
  let rewardHistory = [];
  let successHistory = [];
  let obs, lastQ = [0, 0, 0], lastAction = -1;
  let animFrame = null;

  const TOTAL_STEPS = 50000;

  function init() {
    env = new VesselEnv(42);
    agent = new DQNAgent(env.obsDim, env.nActions, { totalSteps: TOTAL_STEPS });
    episode = 0;
    rewardHistory = [];
    successHistory = [];
    obs = env.reset();
    lastQ = agent.qValues(obs);
    lastAction = -1;
    updateMissionPanel();
    render();
    updateMetrics();
    drawChart();
  }

  // ---------- One environment step ----------
  function stepOnce() {
    const action = agent.act(obs, greedyMode);
    lastAction = action;
    lastQ = agent.qValues(obs);
    const { obs: nextObs, reward, terminated, truncated, info } = env.step(action);
    if (!greedyMode) {
      agent.observe(obs, action, reward, nextObs, terminated);
    }
    obs = nextObs;

    if (terminated || truncated) {
      episode++;
      rewardHistory.push(env.totalReward);
      successHistory.push(info.reachedGoal ? 1 : 0);
      if (rewardHistory.length > 5000) rewardHistory.shift();
      if (successHistory.length > 5000) successHistory.shift();
      // greedy demo runs a single episode then stops
      if (greedyMode) {
        greedyMode = false;
        running = false;
        setButtons();
      }
      obs = env.reset();
      updateMissionPanel();
    }
  }

  // ---------- Main loop ----------
  function loop() {
    if (!running) return;
    const stepsPerFrame = parseInt(speedSlider.value, 10);
    for (let i = 0; i < stepsPerFrame; i++) {
      stepOnce();
      if (!running) break; // greedy episode may have stopped mid-frame
    }
    render();
    updateMetrics();
    drawChart();
    animFrame = requestAnimationFrame(loop);
  }

  // ============================================================
  // RENDERING
  // ============================================================
  function laneColor(lane) {
    return { L02: "#2f6db8", L08: "#8a5cc0", L01: "#3aa06a" }[lane] || "#456";
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

    // ---- planned path ----
    drawPath(env.plannedPath, "#37d67a", 3, 1);

    // ---- actual path (recent trail) ----
    drawPath(env.actualPath, "#ff9f43", 2.5, 0.9);

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

    // ---- start & goal markers ----
    markNode(env.startWp, "#37d67a", "S");
    markNode(env.goalWp, "#ffd54a", "G");

    // ---- ship ----
    const sh = env.states[env.currentWp];
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

    // zero line
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

    // axis labels
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

    // Q-bars
    const maxAbs = Math.max(1, ...lastQ.map((q) => Math.abs(q)));
    for (let a = 0; a < 3; a++) {
      const bar = $("q" + a);
      const val = lastQ[a] || 0;
      const w = (Math.abs(val) / maxAbs) * 50; // % of half-width
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
  // CONTROLS
  // ============================================================
  function setButtons() {
    btnTrain.disabled = running;
    btnPause.disabled = !running;
  }

  btnTrain.addEventListener("click", () => {
    if (running) return;
    greedyMode = false;
    running = true;
    setButtons();
    animFrame = requestAnimationFrame(loop);
  });

  btnPause.addEventListener("click", () => {
    running = false;
    setButtons();
    if (animFrame) cancelAnimationFrame(animFrame);
  });

  btnReset.addEventListener("click", () => {
    running = false;
    setButtons();
    if (animFrame) cancelAnimationFrame(animFrame);
    init();
  });

  btnGreedy.addEventListener("click", () => {
    // run one deterministic (greedy) episode using the learned policy
    if (animFrame) cancelAnimationFrame(animFrame);
    greedyMode = true;
    running = true;
    obs = env.reset();
    updateMissionPanel();
    setButtons();
    animFrame = requestAnimationFrame(loop);
  });

  speedSlider.addEventListener("input", () => {
    speedLabel.textContent = speedSlider.value + "×";
  });

  // ---------- boot ----------
  init();
})();
