/* ============================================================
 * Record trained-DQN greedy trajectories on the Bintulu env, for GIF rendering.
 * Trains a DQN per scenario, then plays greedy episodes and logs the continuous
 * vessel pose at every kinematic sub-step (so the GIF is smooth), plus the
 * planned path, buoys, obstacles, goal, and per-episode metrics.
 * Output: results_bintulu/trajectories/<name>.json
 * ============================================================ */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function load() {
  const jsDir = path.join(__dirname, "..", "js");
  const sb = {};
  sb.window = sb; sb.globalThis = sb; sb.Math = Math; sb.Float64Array = Float64Array; sb.Array = Array; sb.console = console; sb.module = undefined;
  vm.createContext(sb);
  for (const f of ["dqn.js", "environmentBintulu.js"]) vm.runInContext(fs.readFileSync(path.join(jsDir, f), "utf8"), sb, { filename: f });
  return { BintuluPortEnv: sb.window.BintuluPortEnv, BintuluDQN: sb.window.BintuluDQN };
}
const { BintuluPortEnv, BintuluDQN } = load();
const { BintuluEnv } = BintuluPortEnv;
const { DQNAgent } = BintuluDQN;

function mulberry32(seed) { let s = seed >>> 0; return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

// Instrument the env to capture continuous pose at every sub-step by wrapping
// _traverseEdge: after it runs, we sample posX/posY (end of edge). To get smooth
// motion we re-interpolate between waypoints the vessel actually visited using
// the recorded actualPath + final pose. Simpler + robust: record pose after each
// macro-step, then the renderer interpolates between consecutive macro poses.
function playEpisode(env, agent, missionBias) {
  let obs = env.reset(missionBias || null);
  const frames = [{ x: env.posX, y: env.posY, wp: env.currentWp, collided: 0 }];
  let steps = 0;
  while (true) {
    const a = agent.act(obs, true);
    const r = env.step(a);
    frames.push({ x: env.posX, y: env.posY, wp: env.currentWp, collided: r.info.collided ? 1 : 0 });
    obs = r.obs; steps++;
    if (r.terminated || r.truncated) {
      return {
        frames, success: r.info.reachedGoal ? 1 : 0, steps,
        start: env.startWp, goal: env.goalWp,
        collisions: env.collisionCount, iala: env.ialaViolations,
        cte: +(env.cteSamples ? env.cteSum / env.cteSamples : 0).toFixed(2),
        docking: env.dockingAccuracy != null ? +env.dockingAccuracy.toFixed(1) : null,
        plannedPath: env.plannedPath.map((id) => ({ id, x: env.states[id].x, y: env.states[id].y })),
      };
    }
  }
}

function trainAgent(env, episodes, budget) {
  const agent = new DQNAgent(env.obsDim, env.nActions, { algorithm: "DQN", per: false, noisy: false, totalSteps: budget });
  for (let ep = 0; ep < episodes; ep++) {
    let obs = env.reset();
    while (true) {
      const a = agent.act(obs, false);
      const r = env.step(a);
      agent.observe(obs, a, r.reward, r.obs, r.terminated);
      obs = r.obs;
      if (r.terminated || r.truncated) break;
    }
  }
  return agent;
}

function staticScene(env) {
  return {
    mapW: env.MAX_X, mapH: env.MAX_Y,
    buoys: env.buoys.map((b) => ({ x: b.x, y: b.y, color: b.color })),
    obstacles: env.obstacles.map((o) => ({ x: o.x, y: o.y, r: o.r })),
    states: Object.fromEntries(Object.entries(env.states).map(([id, s]) => [id, { x: s.x, y: s.y, lane: s.lane }])),
  };
}

function makeScenario(name, cfg, opts) {
  const seed = opts.seed || 42;
  Math.random = mulberry32(seed * 1000003 + 12345);
  const env = new BintuluEnv(seed, cfg);
  const agent = trainAgent(env, opts.trainEpisodes || 150, opts.budget || 4500);
  // play several greedy episodes; pick the ones matching the desired outcome
  const episodes = [];
  let attempts = 0;
  while (episodes.length < (opts.want || 2) && attempts < 40) {
    attempts++;
    const ep = playEpisode(env, agent);
    if (opts.filter ? opts.filter(ep) : true) episodes.push(ep);
  }
  if (!episodes.length) episodes.push(playEpisode(env, agent));
  return { name, cfg, scene: staticScene(env), episodes };
}

function main() {
  const OUT = path.join(__dirname, "..", "results_bintulu", "trajectories");
  fs.mkdirSync(OUT, { recursive: true });

  const scenarios = [
    // 1) a clean successful transit (prefer a success)
    { name: "clean_success", cfg: { noiseStd: 0, packetErrorRate: 0, obstacleCount: 6 },
      opts: { seed: 6, want: 1, filter: (e) => e.success === 1 } },
    // 2) harsh condition — show drift / collisions (prefer an episode with collisions)
    { name: "harsh_drift", cfg: { noiseStd: 0.25, packetErrorRate: 0.4, obstacleCount: 6 },
      opts: { seed: 2, want: 1, filter: (e) => e.collisions > 0 || e.iala > 0 } },
    // 3) a successful mid-condition transit for the hero loop
    { name: "mid_transit", cfg: { noiseStd: 0.1, packetErrorRate: 0.2, obstacleCount: 6 },
      opts: { seed: 4, want: 1, filter: (e) => e.success === 1 } },
  ];

  for (const s of scenarios) {
    const data = makeScenario(s.name, s.cfg, s.opts);
    fs.writeFileSync(path.join(OUT, s.name + ".json"), JSON.stringify(data));
    const e = data.episodes[0];
    console.log(`${s.name}: ${data.episodes.length} ep(s) | first: ${e.start}->${e.goal} success=${e.success} steps=${e.steps} coll=${e.collisions} iala=${e.iala} frames=${e.frames.length}`);
  }
  console.log("-> results_bintulu/trajectories/*.json");
}
main();
