/* ============================================================
 * BINTULU PORT AUTONOMOUS NAVIGATION (BPAN) — live simulation UI
 * bintulu_main.js
 *
 * Renders the DRL vessel navigating over the REAL Bintulu Port chart
 * (assets/bintulu/bintulu_chart.png) as background. Separate from the
 * Synthetic Port simulation (js/main.js); reuses the generic DQNAgent.
 * ============================================================ */
(function () {
  "use strict";
  const { BintuluEnv, MAP_W, MAP_H } = window.BintuluPortEnv;
  const { DQNAgent } = window.BintuluDQN;

  const canvas = document.getElementById("bintuluCanvas");
  const ctx = canvas.getContext("2d");

  // background chart
  const chart = new Image();
  let chartLoaded = false;
  chart.onload = () => { chartLoaded = true; draw(); };
  chart.src = "assets/bintulu/bintulu_chart.png";

  // state
  let env = new BintuluEnv(42, {});
  let agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: "DQN", totalSteps: 6000 });
  let obs = env.reset();
  let training = false, episode = 0, totalSteps = 0;
  let rewardMA = [], lastReward = 0, bestReward = -Infinity;
  let speed = 6, rafId = null;

  // ---- canvas sizing: keep the chart aspect ratio (1536x1024 -> 3:2) ----
  function resize() {
    const wrap = canvas.parentElement;
    const w = Math.min(wrap.clientWidth, 1100);
    canvas.width = w; canvas.height = Math.round(w * (MAP_H / MAP_W));
    draw();
  }
  window.addEventListener("resize", resize);

  const SX = () => canvas.width / MAP_W;   // scale map px -> canvas px
  const SY = () => canvas.height / MAP_H;
  const mx = (x) => x * SX();
  const my = (y) => y * SY();

  function setStatus() {
    const g = (id) => document.getElementById(id);
    g("statEpisode").textContent = episode;
    g("statSteps").textContent = totalSteps;
    g("statEps").textContent = agent.epsilon != null ? agent.epsilon.toFixed(3) : "—";
    g("statReward").textContent = lastReward.toFixed(1);
    const ma = rewardMA.length ? (rewardMA.reduce((a, b) => a + b, 0) / rewardMA.length) : 0;
    g("statMA").textContent = ma.toFixed(1);
    g("statBest").textContent = isFinite(bestReward) ? bestReward.toFixed(1) : "—";
    g("statStart").textContent = env.startWp;
    g("statGoal").textContent = env.goalWp;
    g("statPlanCost").textContent = env.optimalCost ? env.optimalCost.toFixed(1) : "—";
    g("statColl").textContent = env.collisionCount;
    g("statIala").textContent = env.ialaViolations;
    g("statCte").textContent = (env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(1);
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // chart background
    if (chartLoaded) ctx.drawImage(chart, 0, 0, canvas.width, canvas.height);
    else { ctx.fillStyle = "#0a1220"; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    // subtle darken so overlays read on the bright chart
    ctx.fillStyle = "rgba(6,14,26,0.12)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // planned path (Dijkstra) — green
    const plan = env.plannedPath || [];
    if (plan.length > 1) {
      ctx.strokeStyle = "rgba(55,214,122,0.95)"; ctx.lineWidth = 3; ctx.setLineDash([]);
      ctx.beginPath();
      for (let i = 0; i < plan.length; i++) {
        const s = env.states[plan[i]]; if (!s) continue;
        if (i === 0) ctx.moveTo(mx(s.x), my(s.y)); else ctx.lineTo(mx(s.x), my(s.y));
      }
      ctx.stroke();
    }
    // actual path (vessel) — orange
    if (env.actualPath && env.actualPath.length > 1) {
      ctx.strokeStyle = "rgba(255,159,67,0.95)"; ctx.lineWidth = 2.5; ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i < env.actualPath.length; i++) {
        const s = env.states[env.actualPath[i]]; if (!s) continue;
        if (i === 0) ctx.moveTo(mx(s.x), my(s.y)); else ctx.lineTo(mx(s.x), my(s.y));
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    // waypoints
    for (const id in env.states) {
      const s = env.states[id];
      ctx.fillStyle = s.lane === "HARBOUR" ? "rgba(129,114,178,0.9)" : "rgba(120,170,255,0.75)";
      ctx.beginPath(); ctx.arc(mx(s.x), my(s.y), 3, 0, 2 * Math.PI); ctx.fill();
    }
    // obstacles
    for (const o of env.obstacles) {
      ctx.strokeStyle = "rgba(255,93,93,0.85)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(mx(o.x), my(o.y), o.r * SX(), 0, 2 * Math.PI); ctx.stroke();
    }
    // buoys (chart already shows them; draw crisp markers to tie the overlay)
    for (const b of env.buoys) {
      ctx.fillStyle = b.color === "GREEN" ? "#1ecb6b" : "#ff4d4d";
      ctx.strokeStyle = "#041018"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(mx(b.x), my(b.y), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
    // goal (berth)
    const g = env.states[env.goalWp];
    if (g) {
      ctx.fillStyle = "#ffd24a"; ctx.strokeStyle = "#041018"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(mx(g.x), my(g.y), 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#1a1200"; ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("B", mx(g.x), my(g.y));
    }
    // vessel (continuous pose) — triangle oriented along heading-to-current
    const vx = mx(env.posX), vy = my(env.posY);
    ctx.save(); ctx.translate(vx, vy);
    ctx.fillStyle = "#4da3ff"; ctx.strokeStyle = "#eaf3ff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(-7, 6); ctx.lineTo(-7, -6); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function stepOnce(greedy) {
    const a = agent.act(obs, greedy);
    const r = env.step(a);
    if (!greedy && agent.observe) agent.observe(obs, a, r.reward, r.obs, r.terminated);
    obs = r.obs; totalSteps++;
    if (r.terminated || r.truncated) {
      episode++; lastReward = env.totalReward;
      rewardMA.push(lastReward); if (rewardMA.length > 20) rewardMA.shift();
      if (lastReward > bestReward) bestReward = lastReward;
      obs = env.reset();
    }
  }

  function loop() {
    if (training) { for (let i = 0; i < speed; i++) stepOnce(false); }
    draw(); setStatus();
    rafId = requestAnimationFrame(loop);
  }

  // ---- controls ----
  function cfgFromUI() {
    return {
      noiseStd: parseFloat(document.getElementById("noiseStd").value) || 0,
      packetErrorRate: parseFloat(document.getElementById("packetErr").value) || 0,
      obstacleCount: parseInt(document.getElementById("obstacleCount").value, 10) || 6,
    };
  }
  function rebuild() {
    const algo = document.getElementById("algoSel").value;
    env = new BintuluEnv(42, cfgFromUI());
    agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: algo, totalSteps: 6000 });
    obs = env.reset(); episode = 0; totalSteps = 0; rewardMA = []; bestReward = -Infinity; lastReward = 0;
    draw(); setStatus();
  }

  document.getElementById("btnTrain").addEventListener("click", () => { training = true; });
  document.getElementById("btnPause").addEventListener("click", () => { training = false; });
  document.getElementById("btnReset").addEventListener("click", () => { training = false; rebuild(); });
  document.getElementById("btnGreedy").addEventListener("click", () => {
    training = false; obs = env.reset();
    const run = () => { stepOnce(true); draw(); setStatus(); if (!env.done) setTimeout(run, 1000 / (speed * 4)); };
    run();
  });
  document.getElementById("speed").addEventListener("input", (e) => { speed = parseInt(e.target.value, 10); document.getElementById("speedVal").textContent = speed + "×"; });
  document.getElementById("algoSel").addEventListener("change", rebuild);
  document.getElementById("noiseStd").addEventListener("change", rebuild);
  document.getElementById("packetErr").addEventListener("change", rebuild);
  document.getElementById("obstacleCount").addEventListener("change", rebuild);

  // expose for debugging / live checks
  window.__bpan = { get env() { return env; }, get agent() { return agent; } };

  resize(); setStatus(); loop();
})();
