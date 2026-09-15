/* ============================================================
 * BINTULU PORT — Web DQN Simulation
 * dqn.js
 *
 * Pure-JavaScript value-based RL agents (no external ML library).
 *
 * Supported algorithms (select via cfg.algorithm):
 *   - "DQN"            : vanilla Deep Q-Network
 *   - "DoubleDQN"      : Double DQN (online net selects a', target evaluates)
 *   - "DuelingDQN"     : Dueling architecture (V(s) + A(s,a) streams)
 *   - "DuelingDoubleDQN": Dueling network + Double-Q target
 *
 * Common to all:
 *   - MLP torso obsDim -> 128 -> 128 (ReLU)
 *   - Target network with periodic hard update
 *   - Experience replay buffer
 *   - Epsilon-greedy exploration (linear decay)
 *   - Huber loss + Adam optimizer
 *
 * Hyperparameters mirror the notebook's RL_CONFIG:
 *   learning_rate 1e-3, buffer 50k, batch 64, gamma 0.99,
 *   train_freq 4, target_update 1000, eps 1.0 -> 0.05.
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

// A single fully-connected layer with Adam optimizer state
class Dense {
  constructor(nIn, nOut, activation) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.activation = activation; // "relu" | "linear"
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

/* ------------------------------------------------------------
 * Standard Q-network: obsDim -> H -> H -> nActions
 * ------------------------------------------------------------ */
class QNetwork {
  constructor(nIn, nHidden, nOut) {
    this.type = "mlp";
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

  predict(x) { return this.forward(x).q; }

  zeroGrad() { this.l1.zeroGrad(); this.l2.zeroGrad(); this.l3.zeroGrad(); }

  // dq: dL/dQ (length nOut)
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
 * Dueling Q-network:
 *   torso obsDim -> H -> H
 *   value stream    : H -> Hs -> 1        => V(s)
 *   advantage stream: H -> Hs -> nActions => A(s,a)
 *   Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )
 * ------------------------------------------------------------ */
class DuelingQNetwork {
  constructor(nIn, nHidden, nOut, streamHidden) {
    this.type = "dueling";
    this.nIn = nIn;
    this.nOut = nOut;
    // narrower stream heads keep pure-JS compute light for the browser
    const hs = streamHidden || Math.max(32, Math.round(nHidden / 2));
    this.l1 = new Dense(nIn, nHidden, "relu");
    this.l2 = new Dense(nHidden, nHidden, "relu");
    // value stream
    this.vHidden = new Dense(nHidden, hs, "relu");
    this.vOut = new Dense(hs, 1, "linear");
    // advantage stream
    this.aHidden = new Dense(nHidden, hs, "relu");
    this.aOut = new Dense(hs, nOut, "linear");
  }

  forward(x) {
    const o1 = this.l1.forward(x);
    const o2 = this.l2.forward(o1.a);

    const vh = this.vHidden.forward(o2.a);
    const vo = this.vOut.forward(vh.a); // length 1
    const ah = this.aHidden.forward(o2.a);
    const ao = this.aOut.forward(ah.a); // length nOut

    // mean advantage
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

  // dq: dL/dQ (length nOut)
  backward(cache, dq) {
    const n = this.nOut;
    // Q_k = V + A_k - mean(A)
    //   dL/dV   = sum_k dq_k
    //   dL/dA_j = dq_j - (1/n) * sum_k dq_k
    let sumdq = 0;
    for (let k = 0; k < n; k++) sumdq += dq[k];

    const dV = new Float64Array(1);
    dV[0] = sumdq;
    const dA = new Float64Array(n);
    for (let j = 0; j < n; j++) dA[j] = dq[j] - sumdq / n;

    // value stream back
    const dVh = this.vOut.backwardInto(cache.vo, cache.vh.a, dV);
    const dV2fromV = this.vHidden.backwardInto(cache.vh, cache.o2.a, dVh);
    // advantage stream back
    const dAh = this.aOut.backwardInto(cache.ao, cache.ah.a, dA);
    const dV2fromA = this.aHidden.backwardInto(cache.ah, cache.o2.a, dAh);

    // merge gradients into torso o2 activation
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

class ReplayBuffer {
  constructor(cap) { this.cap = cap; this.data = []; this.pos = 0; }
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
  get size() { return this.data.length; }
}

class DQNAgent {
  constructor(obsDim, nActions, cfg = {}) {
    this.obsDim = obsDim;
    this.nActions = nActions;
    this.cfg = Object.assign(
      {
        algorithm: "DQN",
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
    if (!ALGORITHMS.includes(this.cfg.algorithm)) this.cfg.algorithm = "DQN";
    this.algorithm = this.cfg.algorithm;

    // Double-Q if algorithm name contains "Double"
    this.useDouble = this.algorithm.indexOf("Double") !== -1;
    // Dueling network if algorithm name contains "Dueling"
    this.useDueling = this.algorithm.indexOf("Dueling") !== -1;

    if (this.useDueling) {
      const hs = this.cfg.streamHidden || 64;
      this.q = new DuelingQNetwork(obsDim, this.cfg.hidden, nActions, hs);
      this.target = new DuelingQNetwork(obsDim, this.cfg.hidden, nActions, hs);
    } else {
      this.q = new QNetwork(obsDim, this.cfg.hidden, nActions);
      this.target = new QNetwork(obsDim, this.cfg.hidden, nActions);
    }
    this.target.cloneFrom(this.q);

    this.buffer = new ReplayBuffer(this.cfg.bufferSize);
    this.stepCount = 0;
    this.trainSteps = 0;
    this.lastLoss = 0;
    this.epsilon = this.cfg.epsStart;
  }

  get label() { return ALGO_LABELS[this.algorithm] || this.algorithm; }

  computeEpsilon() {
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
    const batch = this.buffer.sample(this.cfg.batchSize);
    this.q.zeroGrad();
    let lossSum = 0;
    const invN = 1 / batch.length;
    const delta = 1.0; // Huber

    for (const tr of batch) {
      let targetVal = tr.r;
      if (!tr.done) {
        if (this.useDouble) {
          // Double DQN: online net picks a', target net evaluates it
          const qOnlineNext = this.q.predict(tr.s2);
          const aStar = this.argmax(qOnlineNext);
          const qTargetNext = this.target.predict(tr.s2);
          targetVal += this.cfg.gamma * qTargetNext[aStar];
        } else {
          // vanilla: max over target net
          const q2 = this.target.predict(tr.s2);
          let m = q2[0];
          for (let a = 1; a < this.nActions; a++) if (q2[a] > m) m = q2[a];
          targetVal += this.cfg.gamma * m;
        }
      }

      const cache = this.q.forward(tr.s);
      const pred = cache.q[tr.a];
      const err = pred - targetVal;

      let dqA;
      if (Math.abs(err) <= delta) {
        dqA = err;
        lossSum += 0.5 * err * err;
      } else {
        dqA = delta * Math.sign(err);
        lossSum += delta * (Math.abs(err) - 0.5 * delta);
      }
      const dq = new Float64Array(this.nActions);
      dq[tr.a] = dqA * invN;
      this.q.backward(cache, dq);
    }

    this.trainSteps++;
    this.q.adamStep(this.cfg.lr, this.trainSteps);
    this.lastLoss = lossSum * invN;
  }
}

window.BintuluDQN = { DQNAgent, QNetwork, DuelingQNetwork, ALGORITHMS, ALGO_LABELS };
