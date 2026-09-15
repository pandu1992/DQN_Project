/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * dqn.js
 *
 * Pure-JavaScript Deep Q-Network (no external ML library).
 *   - MLP policy net: obsDim -> 128 -> 128 -> nActions (ReLU)
 *   - Target network with periodic hard update
 *   - Experience replay buffer
 *   - Epsilon-greedy exploration (linear decay)
 *   - Huber loss + SGD w/ Adam-style per-parameter moments
 *
 * Hyperparameters mirror the notebook's RL_CONFIG:
 *   learning_rate 1e-3, buffer 50k, batch 64, gamma 0.99,
 *   train_freq 4, target_update 1000, eps 1.0 -> 0.05.
 * ============================================================ */

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// A single fully-connected layer with Adam optimizer state
class Dense {
  constructor(nIn, nOut, activation) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.activation = activation; // "relu" | "linear"
    // He initialization
    const scale = Math.sqrt(2 / nIn);
    this.W = new Float64Array(nIn * nOut);
    this.b = new Float64Array(nOut);
    for (let i = 0; i < this.W.length; i++) this.W[i] = randn() * scale;

    // Adam moments
    this.mW = new Float64Array(nIn * nOut);
    this.vW = new Float64Array(nIn * nOut);
    this.mB = new Float64Array(nOut);
    this.vB = new Float64Array(nOut);

    // gradient accumulators
    this.gW = new Float64Array(nIn * nOut);
    this.gB = new Float64Array(nOut);
  }

  forward(x) {
    // x: Float64Array length nIn -> returns {z, a}
    const z = new Float64Array(this.nOut);
    for (let o = 0; o < this.nOut; o++) {
      let sum = this.b[o];
      const base = o; // column-major-ish: W[i*nOut + o]
      for (let i = 0; i < this.nIn; i++) sum += x[i] * this.W[i * this.nOut + base];
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

  zeroGrad() {
    this.gW.fill(0);
    this.gB.fill(0);
  }

  adamStep(lr, t) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (let i = 0; i < this.W.length; i++) {
      const g = this.gW[i];
      this.mW[i] = b1 * this.mW[i] + (1 - b1) * g;
      this.vW[i] = b2 * this.vW[i] + (1 - b2) * g * g;
      const mHat = this.mW[i] / bc1;
      const vHat = this.vW[i] / bc2;
      this.W[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
    for (let o = 0; o < this.nOut; o++) {
      const g = this.gB[o];
      this.mB[o] = b1 * this.mB[o] + (1 - b1) * g;
      this.vB[o] = b2 * this.vB[o] + (1 - b2) * g * g;
      const mHat = this.mB[o] / bc1;
      const vHat = this.vB[o] / bc2;
      this.b[o] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }

  cloneWeightsFrom(other) {
    this.W.set(other.W);
    this.b.set(other.b);
  }
}

class QNetwork {
  constructor(nIn, nHidden, nOut) {
    this.l1 = new Dense(nIn, nHidden, "relu");
    this.l2 = new Dense(nHidden, nHidden, "relu");
    this.l3 = new Dense(nHidden, nOut, "linear");
    this.nIn = nIn;
    this.nOut = nOut;
  }

  forward(x) {
    const o1 = this.l1.forward(x);
    const o2 = this.l2.forward(o1.a);
    const o3 = this.l3.forward(o2.a);
    return { x, o1, o2, o3, q: o3.a };
  }

  predict(x) {
    return this.forward(x).q;
  }

  zeroGrad() {
    this.l1.zeroGrad();
    this.l2.zeroGrad();
    this.l3.zeroGrad();
  }

  // Accumulate gradients for one sample given dL/dq (length nOut)
  backward(cache, dq) {
    const { x, o1, o2 } = cache;
    // layer 3 (linear)
    const dA2 = new Float64Array(this.l2.nOut);
    for (let o = 0; o < this.l3.nOut; o++) {
      const grad = dq[o];
      this.l3.gB[o] += grad;
      for (let i = 0; i < this.l3.nIn; i++) {
        this.l3.gW[i * this.l3.nOut + o] += o2.a[i] * grad;
        dA2[i] += this.l3.W[i * this.l3.nOut + o] * grad;
      }
    }
    // layer 2 (relu)
    const dZ2 = new Float64Array(this.l2.nOut);
    for (let o = 0; o < this.l2.nOut; o++) dZ2[o] = o2.z[o] > 0 ? dA2[o] : 0;
    const dA1 = new Float64Array(this.l1.nOut);
    for (let o = 0; o < this.l2.nOut; o++) {
      const grad = dZ2[o];
      this.l2.gB[o] += grad;
      for (let i = 0; i < this.l2.nIn; i++) {
        this.l2.gW[i * this.l2.nOut + o] += o1.a[i] * grad;
        dA1[i] += this.l2.W[i * this.l2.nOut + o] * grad;
      }
    }
    // layer 1 (relu)
    const dZ1 = new Float64Array(this.l1.nOut);
    for (let o = 0; o < this.l1.nOut; o++) dZ1[o] = o1.z[o] > 0 ? dA1[o] : 0;
    for (let o = 0; o < this.l1.nOut; o++) {
      const grad = dZ1[o];
      this.l1.gB[o] += grad;
      for (let i = 0; i < this.l1.nIn; i++) {
        this.l1.gW[i * this.l1.nOut + o] += x[i] * grad;
      }
    }
  }

  adamStep(lr, t) {
    this.l1.adamStep(lr, t);
    this.l2.adamStep(lr, t);
    this.l3.adamStep(lr, t);
  }

  cloneFrom(other) {
    this.l1.cloneWeightsFrom(other.l1);
    this.l2.cloneWeightsFrom(other.l2);
    this.l3.cloneWeightsFrom(other.l3);
  }
}

class ReplayBuffer {
  constructor(cap) {
    this.cap = cap;
    this.data = [];
    this.pos = 0;
  }
  push(t) {
    if (this.data.length < this.cap) this.data.push(t);
    else this.data[this.pos] = t;
    this.pos = (this.pos + 1) % this.cap;
  }
  sample(n) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(this.data[Math.floor(Math.random() * this.data.length)]);
    return out;
  }
  get size() {
    return this.data.length;
  }
}

class DQNAgent {
  constructor(obsDim, nActions, cfg = {}) {
    this.obsDim = obsDim;
    this.nActions = nActions;
    this.cfg = Object.assign(
      {
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
      },
      cfg
    );
    this.q = new QNetwork(obsDim, this.cfg.hidden, nActions);
    this.target = new QNetwork(obsDim, this.cfg.hidden, nActions);
    this.target.cloneFrom(this.q);
    this.buffer = new ReplayBuffer(this.cfg.bufferSize);
    this.stepCount = 0;
    this.trainSteps = 0;
    this.lastLoss = 0;
    this.epsilon = this.cfg.epsStart;
  }

  computeEpsilon() {
    const frac = Math.min(1, this.stepCount / (this.cfg.epsFraction * this.cfg.totalSteps));
    this.epsilon = this.cfg.epsStart + frac * (this.cfg.epsEnd - this.cfg.epsStart);
    return this.epsilon;
  }

  act(obs, greedy = false) {
    this.computeEpsilon();
    if (!greedy && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.nActions);
    }
    const q = this.q.predict(obs);
    let best = 0;
    for (let a = 1; a < this.nActions; a++) if (q[a] > q[best]) best = a;
    return best;
  }

  qValues(obs) {
    return Array.from(this.q.predict(obs));
  }

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
    const batch = this.buffer.sample(this.cfg.batchSize);
    this.q.zeroGrad();
    let lossSum = 0;
    const invN = 1 / batch.length;

    for (const tr of batch) {
      // target: r + gamma * max_a' Q_target(s2)  (0 if done)
      let targetVal = tr.r;
      if (!tr.done) {
        const q2 = this.target.predict(tr.s2);
        let m = q2[0];
        for (let a = 1; a < this.nActions; a++) if (q2[a] > m) m = q2[a];
        targetVal += this.cfg.gamma * m;
      }
      const cache = this.q.forward(tr.s);
      const pred = cache.q[tr.a];
      let err = pred - targetVal;
      // Huber loss derivative (delta = 1)
      const delta = 1.0;
      let dqA;
      if (Math.abs(err) <= delta) {
        dqA = err;
        lossSum += 0.5 * err * err;
      } else {
        dqA = delta * Math.sign(err);
        lossSum += delta * (Math.abs(err) - 0.5 * delta);
      }
      const dq = new Float64Array(this.nActions); // gradient only on taken action
      dq[tr.a] = dqA * invN;
      this.q.backward(cache, dq);
    }

    this.trainSteps++;
    this.q.adamStep(this.cfg.lr, this.trainSteps);
    this.lastLoss = lossSum * invN;
  }
}

window.BintuluDQN = { DQNAgent, QNetwork };
