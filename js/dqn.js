/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * dqn.js
 *
 * Pure-JavaScript value-based RL agents (no external ML library).
 *
 * Base algorithms (cfg.algorithm):
 *   - "DQN"             : vanilla Deep Q-Network
 *   - "DoubleDQN"       : Double DQN (online selects a', target evaluates)
 *   - "DuelingDQN"      : Dueling architecture (V(s) + A(s,a) streams)
 *   - "DuelingDoubleDQN": Dueling network + Double-Q target
 *
 * Toggle-able enhancements (stackable on any base algorithm):
 *   - cfg.per   : Prioritized Experience Replay (proportional, TD-error
 *                 priorities + importance-sampling weights)
 *   - cfg.noisy : Noisy Nets (NoisyDense factorized Gaussian noise on the
 *                 head layers; exploration comes from weight noise so
 *                 epsilon-greedy is disabled)
 *
 * Common:
 *   - MLP torso obsDim -> 128 -> 128 (ReLU)
 *   - Target network with periodic hard update
 *   - Huber loss + Adam optimizer
 *
 * Hyperparameters mirror the notebook's RL_CONFIG.
 * ============================================================ */

const ALGORITHMS = ["DQN", "DoubleDQN", "DuelingDQN", "DuelingDoubleDQN"];
const ALGO_LABELS = {
  DQN: "DQN",
  DoubleDQN: "Double DQN",
  DuelingDQN: "Dueling DQN",
  DuelingDoubleDQN: "Dueling Double DQN",
};

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ============================================================
 * Dense: fully-connected layer with Adam optimizer state
 * ============================================================ */
class Dense {
  constructor(nIn, nOut, activation) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.activation = activation; // "relu" | "linear"
    this.noisy = false;
    const scale = Math.sqrt(2 / nIn); // He init
    this.W = new Float64Array(nIn * nOut);
    this.b = new Float64Array(nOut);
    for (let i = 0; i < this.W.length; i++) this.W[i] = randn() * scale;

    this.mW = new Float64Array(nIn * nOut);
    this.vW = new Float64Array(nIn * nOut);
    this.mB = new Float64Array(nOut);
    this.vB = new Float64Array(nOut);
    this.gW = new Float64Array(nIn * nOut);
    this.gB = new Float64Array(nOut);
  }

  resetNoise() {} // no-op for plain dense

  forward(x) {
    const z = new Float64Array(this.nOut);
    for (let o = 0; o < this.nOut; o++) {
      let sum = this.b[o];
      for (let i = 0; i < this.nIn; i++) sum += x[i] * this.W[i * this.nOut + o];
      z[o] = sum;
    }
    const a = new Float64Array(this.nOut);
    if (this.activation === "relu") {
      for (let o = 0; o < this.nOut; o++) a[o] = z[o] > 0 ? z[o] : 0;
    } else {
      a.set(z);
    }
    return { z, a };
  }

  // Given dL/da (length nOut), accumulate grads and return dL/dx (length nIn)
  backwardInto(cacheIn, aInput, dA) {
    const dZ = new Float64Array(this.nOut);
    if (this.activation === "relu") {
      for (let o = 0; o < this.nOut; o++) dZ[o] = cacheIn.z[o] > 0 ? dA[o] : 0;
    } else {
      dZ.set(dA);
    }
    const dX = new Float64Array(this.nIn);
    for (let o = 0; o < this.nOut; o++) {
      const g = dZ[o];
      this.gB[o] += g;
      for (let i = 0; i < this.nIn; i++) {
        this.gW[i * this.nOut + o] += aInput[i] * g;
        dX[i] += this.W[i * this.nOut + o] * g;
      }
    }
    return dX;
  }

  zeroGrad() { this.gW.fill(0); this.gB.fill(0); }

  adamStep(lr, t) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (let i = 0; i < this.W.length; i++) {
      const g = this.gW[i];
      this.mW[i] = b1 * this.mW[i] + (1 - b1) * g;
      this.vW[i] = b2 * this.vW[i] + (1 - b2) * g * g;
      this.W[i] -= (lr * (this.mW[i] / bc1)) / (Math.sqrt(this.vW[i] / bc2) + eps);
    }
    for (let o = 0; o < this.nOut; o++) {
      const g = this.gB[o];
      this.mB[o] = b1 * this.mB[o] + (1 - b1) * g;
      this.vB[o] = b2 * this.vB[o] + (1 - b2) * g * g;
      this.b[o] -= (lr * (this.mB[o] / bc1)) / (Math.sqrt(this.vB[o] / bc2) + eps);
    }
  }

  cloneWeightsFrom(other) { this.W.set(other.W); this.b.set(other.b); }
}

/* ============================================================
 * NoisyDense: NoisyNet layer (Fortunato et al. 2018)
 *   W = mu_W + sigma_W * eps_W   (factorized Gaussian noise)
 *   b = mu_b + sigma_b * eps_b
 *   eps_W[i,o] = f(eps_in[i]) * f(eps_out[o]),  f(x)=sign(x)*sqrt(|x|)
 * Exploration comes from the sampled weight noise; caller resets noise
 * once per forward pass (and per gradient step for the training sample).
 * Same interface as Dense so networks can swap it in transparently.
 * ============================================================ */
class NoisyDense {
  constructor(nIn, nOut, activation) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.activation = activation;
    this.noisy = true;

    const muRange = 1 / Math.sqrt(nIn);
    const sigma0 = 0.5;
    this.muW = new Float64Array(nIn * nOut);
    this.sigW = new Float64Array(nIn * nOut);
    this.muB = new Float64Array(nOut);
    this.sigB = new Float64Array(nOut);
    for (let i = 0; i < this.muW.length; i++) {
      this.muW[i] = (Math.random() * 2 - 1) * muRange;
      this.sigW[i] = (sigma0 / Math.sqrt(nIn));
    }
    for (let o = 0; o < nOut; o++) {
      this.muB[o] = (Math.random() * 2 - 1) * muRange;
      this.sigB[o] = sigma0 / Math.sqrt(nIn);
    }

    // factorized noise vectors
    this.epsIn = new Float64Array(nIn);
    this.epsOut = new Float64Array(nOut);
    this.resetNoise();

    // Adam moments for the 4 parameter tensors
    this.mMuW = new Float64Array(nIn * nOut); this.vMuW = new Float64Array(nIn * nOut);
    this.mSigW = new Float64Array(nIn * nOut); this.vSigW = new Float64Array(nIn * nOut);
    this.mMuB = new Float64Array(nOut); this.vMuB = new Float64Array(nOut);
    this.mSigB = new Float64Array(nOut); this.vSigB = new Float64Array(nOut);

    // gradient accumulators
    this.gMuW = new Float64Array(nIn * nOut);
    this.gSigW = new Float64Array(nIn * nOut);
    this.gMuB = new Float64Array(nOut);
    this.gSigB = new Float64Array(nOut);
  }

  static f(x) { return Math.sign(x) * Math.sqrt(Math.abs(x)); }

  resetNoise() {
    for (let i = 0; i < this.nIn; i++) this.epsIn[i] = NoisyDense.f(randn());
    for (let o = 0; o < this.nOut; o++) this.epsOut[o] = NoisyDense.f(randn());
  }

  // effective weight/bias given current noise
  _wEff(i, o) {
    const idx = i * this.nOut + o;
    return this.muW[idx] + this.sigW[idx] * (this.epsIn[i] * this.epsOut[o]);
  }
  _bEff(o) {
    return this.muB[o] + this.sigB[o] * this.epsOut[o];
  }

  forward(x) {
    const z = new Float64Array(this.nOut);
    for (let o = 0; o < this.nOut; o++) {
      let sum = this._bEff(o);
      const eo = this.epsOut[o];
      for (let i = 0; i < this.nIn; i++) {
        const idx = i * this.nOut + o;
        const w = this.muW[idx] + this.sigW[idx] * (this.epsIn[i] * eo);
        sum += x[i] * w;
      }
      z[o] = sum;
    }
    const a = new Float64Array(this.nOut);
    if (this.activation === "relu") {
      for (let o = 0; o < this.nOut; o++) a[o] = z[o] > 0 ? z[o] : 0;
    } else {
      a.set(z);
    }
    return { z, a };
  }

  backwardInto(cacheIn, aInput, dA) {
    const dZ = new Float64Array(this.nOut);
    if (this.activation === "relu") {
      for (let o = 0; o < this.nOut; o++) dZ[o] = cacheIn.z[o] > 0 ? dA[o] : 0;
    } else {
      dZ.set(dA);
    }
    const dX = new Float64Array(this.nIn);
    for (let o = 0; o < this.nOut; o++) {
      const g = dZ[o];
      const eo = this.epsOut[o];
      // bias grads
      this.gMuB[o] += g;
      this.gSigB[o] += g * eo;
      for (let i = 0; i < this.nIn; i++) {
        const idx = i * this.nOut + o;
        const noise = this.epsIn[i] * eo;
        this.gMuW[idx] += aInput[i] * g;
        this.gSigW[idx] += aInput[i] * g * noise;
        // dX uses effective weight
        const wEff = this.muW[idx] + this.sigW[idx] * noise;
        dX[i] += wEff * g;
      }
    }
    return dX;
  }

  zeroGrad() {
    this.gMuW.fill(0); this.gSigW.fill(0);
    this.gMuB.fill(0); this.gSigB.fill(0);
  }

  adamStep(lr, t) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    const upd = (param, m, v, g) => {
      for (let i = 0; i < param.length; i++) {
        const gi = g[i];
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        param[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
      }
    };
    upd(this.muW, this.mMuW, this.vMuW, this.gMuW);
    upd(this.sigW, this.mSigW, this.vSigW, this.gSigW);
    upd(this.muB, this.mMuB, this.vMuB, this.gMuB);
    upd(this.sigB, this.mSigB, this.vSigB, this.gSigB);
  }

  cloneWeightsFrom(other) {
    this.muW.set(other.muW); this.sigW.set(other.sigW);
    this.muB.set(other.muB); this.sigB.set(other.sigB);
  }
}

function makeLayer(nIn, nOut, activation, noisy) {
  return noisy ? new NoisyDense(nIn, nOut, activation) : new Dense(nIn, nOut, activation);
}

/* ------------------------------------------------------------
 * Standard Q-network: obsDim -> H -> H -> nActions
 * (head layer optionally noisy)
 * ------------------------------------------------------------ */
class QNetwork {
  constructor(nIn, nHidden, nOut, noisy) {
    this.type = "mlp";
    this.noisy = !!noisy;
    this.l1 = new Dense(nIn, nHidden, "relu");
    this.l2 = new Dense(nHidden, nHidden, "relu");
    this.l3 = makeLayer(nHidden, nOut, "linear", this.noisy);
    this.nIn = nIn;
    this.nOut = nOut;
  }

  resetNoise() { this.l3.resetNoise(); }

  forward(x) {
    const o1 = this.l1.forward(x);
    const o2 = this.l2.forward(o1.a);
    const o3 = this.l3.forward(o2.a);
    return { x, o1, o2, o3, q: o3.a };
  }

  predict(x) { return this.forward(x).q; }

  zeroGrad() { this.l1.zeroGrad(); this.l2.zeroGrad(); this.l3.zeroGrad(); }

  backward(cache, dq) {
    const dA2 = this.l3.backwardInto(cache.o3, cache.o2.a, dq);
    const dA1 = this.l2.backwardInto(cache.o2, cache.o1.a, dA2);
    this.l1.backwardInto(cache.o1, cache.x, dA1);
  }

  adamStep(lr, t) { this.l1.adamStep(lr, t); this.l2.adamStep(lr, t); this.l3.adamStep(lr, t); }

  cloneFrom(other) {
    this.l1.cloneWeightsFrom(other.l1);
    this.l2.cloneWeightsFrom(other.l2);
    this.l3.cloneWeightsFrom(other.l3);
  }
}

/* ------------------------------------------------------------
 * Dueling Q-network (value + advantage streams; heads optionally noisy)
 * ------------------------------------------------------------ */
class DuelingQNetwork {
  constructor(nIn, nHidden, nOut, streamHidden, noisy) {
    this.type = "dueling";
    this.noisy = !!noisy;
    this.nIn = nIn;
    this.nOut = nOut;
    const hs = streamHidden || Math.max(32, Math.round(nHidden / 2));
    this.l1 = new Dense(nIn, nHidden, "relu");
    this.l2 = new Dense(nHidden, nHidden, "relu");
    this.vHidden = new Dense(nHidden, hs, "relu");
    this.vOut = makeLayer(hs, 1, "linear", this.noisy);
    this.aHidden = new Dense(nHidden, hs, "relu");
    this.aOut = makeLayer(hs, nOut, "linear", this.noisy);
  }

  resetNoise() { this.vOut.resetNoise(); this.aOut.resetNoise(); }

  forward(x) {
    const o1 = this.l1.forward(x);
    const o2 = this.l2.forward(o1.a);
    const vh = this.vHidden.forward(o2.a);
    const vo = this.vOut.forward(vh.a);
    const ah = this.aHidden.forward(o2.a);
    const ao = this.aOut.forward(ah.a);

    let meanA = 0;
    for (let k = 0; k < this.nOut; k++) meanA += ao.a[k];
    meanA /= this.nOut;

    const q = new Float64Array(this.nOut);
    for (let k = 0; k < this.nOut; k++) q[k] = vo.a[0] + (ao.a[k] - meanA);
    return { x, o1, o2, vh, vo, ah, ao, meanA, q };
  }

  predict(x) { return this.forward(x).q; }

  zeroGrad() {
    this.l1.zeroGrad(); this.l2.zeroGrad();
    this.vHidden.zeroGrad(); this.vOut.zeroGrad();
    this.aHidden.zeroGrad(); this.aOut.zeroGrad();
  }

  backward(cache, dq) {
    const n = this.nOut;
    let sumdq = 0;
    for (let k = 0; k < n; k++) sumdq += dq[k];
    const dV = new Float64Array(1);
    dV[0] = sumdq;
    const dA = new Float64Array(n);
    for (let j = 0; j < n; j++) dA[j] = dq[j] - sumdq / n;

    const dVh = this.vOut.backwardInto(cache.vo, cache.vh.a, dV);
    const dV2fromV = this.vHidden.backwardInto(cache.vh, cache.o2.a, dVh);
    const dAh = this.aOut.backwardInto(cache.ao, cache.ah.a, dA);
    const dV2fromA = this.aHidden.backwardInto(cache.ah, cache.o2.a, dAh);

    const dO2 = new Float64Array(dV2fromV.length);
    for (let i = 0; i < dO2.length; i++) dO2[i] = dV2fromV[i] + dV2fromA[i];
    const dA1 = this.l2.backwardInto(cache.o2, cache.o1.a, dO2);
    this.l1.backwardInto(cache.o1, cache.x, dA1);
  }

  adamStep(lr, t) {
    this.l1.adamStep(lr, t); this.l2.adamStep(lr, t);
    this.vHidden.adamStep(lr, t); this.vOut.adamStep(lr, t);
    this.aHidden.adamStep(lr, t); this.aOut.adamStep(lr, t);
  }

  cloneFrom(other) {
    this.l1.cloneWeightsFrom(other.l1);
    this.l2.cloneWeightsFrom(other.l2);
    this.vHidden.cloneWeightsFrom(other.vHidden);
    this.vOut.cloneWeightsFrom(other.vOut);
    this.aHidden.cloneWeightsFrom(other.aHidden);
    this.aOut.cloneWeightsFrom(other.aOut);
  }
}

/* ============================================================
 * Replay buffers
 * ============================================================ */
class ReplayBuffer {
  constructor(cap) { this.cap = cap; this.data = []; this.pos = 0; }
  push(t) {
    if (this.data.length < this.cap) this.data.push(t);
    else this.data[this.pos] = t;
    this.pos = (this.pos + 1) % this.cap;
  }
  // returns { items, indices, weights }
  sample(n) {
    const items = [], indices = [], weights = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * this.data.length);
      items.push(this.data[idx]); indices.push(idx); weights.push(1);
    }
    return { items, indices, weights };
  }
  updatePriorities() {} // no-op
  get size() { return this.data.length; }
}

/* Prioritized Experience Replay (proportional variant).
 * priority p_i = (|TD_i| + eps)^alpha ; sample prob ∝ p_i
 * IS weight w_i = (N * P(i))^(-beta), normalized by max w.
 */
class PrioritizedReplayBuffer {
  constructor(cap, alpha = 0.6, betaStart = 0.4, betaSteps = 20000) {
    this.cap = cap;
    this.alpha = alpha;
    this.betaStart = betaStart;
    this.betaSteps = betaSteps;
    this.eps = 1e-4;
    this.data = [];
    this.prio = []; // priority^alpha already applied at store? keep raw p^alpha
    this.pos = 0;
    this.maxPrio = 1.0;
    this.frame = 0;
  }
  push(t) {
    const p = this.maxPrio; // new samples get max priority
    if (this.data.length < this.cap) { this.data.push(t); this.prio.push(p); }
    else { this.data[this.pos] = t; this.prio[this.pos] = p; }
    this.pos = (this.pos + 1) % this.cap;
  }
  _beta() {
    const f = Math.min(1, this.frame / this.betaSteps);
    return this.betaStart + f * (1 - this.betaStart);
  }
  sample(n) {
    this.frame++;
    const N = this.data.length;
    let total = 0;
    for (let i = 0; i < N; i++) total += this.prio[i];
    const beta = this._beta();
    const items = [], indices = [], weights = [];
    let maxW = 0;
    for (let k = 0; k < n; k++) {
      const r = Math.random() * total;
      let acc = 0, idx = 0;
      for (let i = 0; i < N; i++) { acc += this.prio[i]; if (acc >= r) { idx = i; break; } }
      const P = this.prio[idx] / total;
      const w = Math.pow(N * P, -beta);
      if (w > maxW) maxW = w;
      items.push(this.data[idx]); indices.push(idx); weights.push(w);
    }
    for (let k = 0; k < weights.length; k++) weights[k] /= (maxW || 1);
    return { items, indices, weights };
  }
  updatePriorities(indices, tdErrors) {
    for (let k = 0; k < indices.length; k++) {
      const p = Math.pow(Math.abs(tdErrors[k]) + this.eps, this.alpha);
      this.prio[indices[k]] = p;
      if (p > this.maxPrio) this.maxPrio = p;
    }
  }
  get size() { return this.data.length; }
}

/* ============================================================
 * Agent
 * ============================================================ */
class DQNAgent {
  constructor(obsDim, nActions, cfg = {}) {
    this.obsDim = obsDim;
    this.nActions = nActions;
    this.cfg = Object.assign(
      {
        algorithm: "DQN",
        per: false,
        noisy: false,
        lr: 1e-3,
        gamma: 0.99,
        bufferSize: 50000,
        batchSize: 64,
        learningStarts: 1000,
        trainFreq: 4,
        targetUpdate: 1000,
        epsStart: 1.0,
        epsEnd: 0.05,
        epsFraction: 0.2,
        totalSteps: 50000,
        hidden: 128,
        perAlpha: 0.6,
        perBeta: 0.4,
      },
      cfg
    );
    if (!ALGORITHMS.includes(this.cfg.algorithm)) this.cfg.algorithm = "DQN";
    this.algorithm = this.cfg.algorithm;
    this.usePER = !!this.cfg.per;
    this.useNoisy = !!this.cfg.noisy;

    this.useDouble = this.algorithm.indexOf("Double") !== -1;
    this.useDueling = this.algorithm.indexOf("Dueling") !== -1;

    if (this.useDueling) {
      const hs = this.cfg.streamHidden || 64;
      this.q = new DuelingQNetwork(obsDim, this.cfg.hidden, nActions, hs, this.useNoisy);
      this.target = new DuelingQNetwork(obsDim, this.cfg.hidden, nActions, hs, this.useNoisy);
    } else {
      this.q = new QNetwork(obsDim, this.cfg.hidden, nActions, this.useNoisy);
      this.target = new QNetwork(obsDim, this.cfg.hidden, nActions, this.useNoisy);
    }
    this.target.cloneFrom(this.q);

    this.buffer = this.usePER
      ? new PrioritizedReplayBuffer(this.cfg.bufferSize, this.cfg.perAlpha, this.cfg.perBeta, this.cfg.totalSteps / 2)
      : new ReplayBuffer(this.cfg.bufferSize);

    this.stepCount = 0;
    this.trainSteps = 0;
    this.lastLoss = 0;
    this.epsilon = this.useNoisy ? 0 : this.cfg.epsStart;
  }

  get label() {
    let s = ALGO_LABELS[this.algorithm] || this.algorithm;
    const mods = [];
    if (this.usePER) mods.push("PER");
    if (this.useNoisy) mods.push("Noisy");
    if (mods.length) s += " +" + mods.join("+");
    return s;
  }

  computeEpsilon() {
    if (this.useNoisy) { this.epsilon = 0; return 0; } // exploration via weight noise
    const frac = Math.min(1, this.stepCount / (this.cfg.epsFraction * this.cfg.totalSteps));
    this.epsilon = this.cfg.epsStart + frac * (this.cfg.epsEnd - this.cfg.epsStart);
    return this.epsilon;
  }

  argmax(arr) {
    let best = 0;
    for (let a = 1; a < arr.length; a++) if (arr[a] > arr[best]) best = a;
    return best;
  }

  act(obs, greedy = false) {
    this.computeEpsilon();
    if (this.useNoisy) {
      // resample noise each action; exploration is intrinsic
      this.q.resetNoise();
      return this.argmax(this.q.predict(obs));
    }
    if (!greedy && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.nActions);
    }
    return this.argmax(this.q.predict(obs));
  }

  qValues(obs) { return Array.from(this.q.predict(obs)); }

  observe(s, a, r, s2, done) {
    this.buffer.push({ s, a, r, s2, done });
    this.stepCount++;
    if (this.buffer.size >= this.cfg.learningStarts && this.stepCount % this.cfg.trainFreq === 0) {
      this._train();
    }
    if (this.trainSteps > 0 && this.trainSteps % this.cfg.targetUpdate === 0) {
      this.target.cloneFrom(this.q);
    }
  }

  _train() {
    const { items: batch, indices, weights } = this.buffer.sample(this.cfg.batchSize);
    if (this.useNoisy) { this.q.resetNoise(); this.target.resetNoise(); }
    this.q.zeroGrad();
    let lossSum = 0;
    const invN = 1 / batch.length;
    const delta = 1.0;
    const tdErrors = new Array(batch.length);

    for (let bi = 0; bi < batch.length; bi++) {
      const tr = batch[bi];
      const w = weights[bi];

      let targetVal = tr.r;
      if (!tr.done) {
        if (this.useDouble) {
          const qOnlineNext = this.q.predict(tr.s2);
          const aStar = this.argmax(qOnlineNext);
          const qTargetNext = this.target.predict(tr.s2);
          targetVal += this.cfg.gamma * qTargetNext[aStar];
        } else {
          const q2 = this.target.predict(tr.s2);
          let m = q2[0];
          for (let a = 1; a < this.nActions; a++) if (q2[a] > m) m = q2[a];
          targetVal += this.cfg.gamma * m;
        }
      }

      const cache = this.q.forward(tr.s);
      const pred = cache.q[tr.a];
      const err = pred - targetVal;
      tdErrors[bi] = err;

      // Huber loss + importance-sampling weight
      let dqA;
      if (Math.abs(err) <= delta) {
        dqA = err;
        lossSum += w * 0.5 * err * err;
      } else {
        dqA = delta * Math.sign(err);
        lossSum += w * delta * (Math.abs(err) - 0.5 * delta);
      }
      const dq = new Float64Array(this.nActions);
      dq[tr.a] = dqA * w * invN;
      this.q.backward(cache, dq);
    }

    this.trainSteps++;
    this.q.adamStep(this.cfg.lr, this.trainSteps);
    this.lastLoss = lossSum * invN;

    if (this.usePER) this.buffer.updatePriorities(indices, tdErrors);
  }
}

window.BintuluDQN = {
  DQNAgent, QNetwork, DuelingQNetwork,
  Dense, NoisyDense, ReplayBuffer, PrioritizedReplayBuffer,
  ALGORITHMS, ALGO_LABELS,
};
