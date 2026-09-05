// Topic Class Colors & Icons
const TOPIC_CONFIG = {
  "Neural_Networks": { color: "#6366f1", icon: "fa-brain" },
  "Probabilistic_Methods": { color: "#06b6d4", icon: "fa-chart-column" },
  "Reinforcement_Learning": { color: "#10b981", icon: "fa-robot" },
  "Theory": { color: "#f59e0b", icon: "fa-book-open" },
  "Genetic_Algorithms": { color: "#ec4899", icon: "fa-dna" },
  "Rule_Learning": { color: "#8b5cf6", icon: "fa-scale-balanced" },
  "Case_Based": { color: "#f43f5e", icon: "fa-briefcase" }
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initTabs();
  initCoraExplorer();
  initCustomPredictor();
  initModelInfo();
});

// --- Health Check ---
async function initHealthCheck() {
  const statusBadge = document.getElementById("status-badge");
  const statusText = document.getElementById("status-text");

  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "healthy") {
      statusText.textContent = "ONNX Model Online";
      statusBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
    } else {
      statusText.textContent = "Model Offline";
      statusBadge.querySelector(".status-dot").style.backgroundColor = "#f43f5e";
    }
  } catch (err) {
    statusText.textContent = "API Disconnected";
    statusBadge.querySelector(".status-dot").style.backgroundColor = "#f43f5e";
  }
}

// --- Tabs ---
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(tabId).classList.add("active");
    });
  });
}

// --- Cora Explorer ---
function initCoraExplorer() {
  const nodeInput = document.getElementById("node-id-input");
  const btnPredict = document.getElementById("btn-predict-cora");
  const btnRandom = document.getElementById("btn-random-node");
  const chips = document.querySelectorAll(".chip[data-node]");

  btnPredict.addEventListener("click", () => {
    predictCoraNode(nodeInput.value);
  });

  btnRandom.addEventListener("click", () => {
    const randomId = Math.floor(Math.random() * 2708);
    nodeInput.value = randomId;
    predictCoraNode(randomId);
  });

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const nodeId = parseInt(chip.getAttribute("data-node"), 10);
      nodeInput.value = nodeId;
      predictCoraNode(nodeId);
    });
  });

  // Initial predict for Node #0 removed
}

async function predictCoraNode(nodeIdsStr) {
  const emptyState = document.getElementById("cora-empty-state");
  const resultsContainer = document.getElementById("cora-results-container");
  const graphNodeCount = document.getElementById("graph-node-count");

  // Parse input into an array of integers
  const nodeIds = String(nodeIdsStr)
    .split(',')
    .map(s => parseInt(s.trim(), 10))
    .filter(n => !isNaN(n));
    
  if (nodeIds.length === 0) return alert("Please enter valid node IDs.");

  graphNodeCount.textContent = `Nodes: ${nodeIds.join(", ")}`;

  try {
    const res = await fetch("/predict/cora_node", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_indices: nodeIds })
    });

    if (!res.ok) {
      let errorMsg = `HTTP Error ${res.status}`;
      try {
        const errJson = await res.json();
        errorMsg = errJson.detail || errorMsg;
      } catch (e) {}
      throw new Error(errorMsg);
    }

    const data = await res.json();

    emptyState.classList.add("hidden");
    resultsContainer.classList.remove("hidden");
    resultsContainer.innerHTML = ""; // clear old results

    const classMapping = {
      0: "Case_Based", 1: "Genetic_Algorithms", 2: "Neural_Networks",
      3: "Probabilistic_Methods", 4: "Reinforcement_Learning", 5: "Rule_Learning", 6: "Theory"
    };

    data.predictions.forEach(pred => {
      const topicName = pred.predicted_label;
      const topicCfg = TOPIC_CONFIG[topicName] || { color: "#6366f1", icon: "fa-graduation-cap" };
      
      const probs = pred.probabilities || [];
      const maxProb = probs.length ? Math.max(...probs) : 0;
      
      const paired = probs.map((prob, idx) => ({
        name: classMapping[idx] || `Class_${idx}`,
        prob: prob,
        isTop: prob === maxProb
      }));
      paired.sort((a, b) => b.prob - a.prob);

      let probHTML = '';
      paired.forEach(item => {
        const cfg = TOPIC_CONFIG[item.name] || { color: "#6366f1" };
        const pctStr = (item.prob * 100).toFixed(1);
        probHTML += `
          <div class="prob-item">
            <div class="prob-info">
              <span class="prob-name">${item.name.replace(/_/g, " ")}</span>
              <span class="prob-pct">${pctStr}%</span>
            </div>
            <div class="prob-bar-track">
              <div class="prob-bar-fill ${item.isTop ? 'highest' : ''}" style="width: ${pctStr}%; background: ${item.isTop ? cfg.color : ''}"></div>
            </div>
          </div>
        `;
      });

      const cardHTML = `
        <div style="border: 1px solid var(--bg-card-border); padding: 1.5rem; border-radius: var(--radius-md); background: rgba(0,0,0,0.2);">
          <div class="prediction-hero-card" style="margin-bottom: 1rem;">
            <div class="pred-label">Node #${pred.node_index} Prediction</div>
            <div class="pred-topic-badge" style="color: ${topicCfg.color}">
              <i class="fa-solid ${topicCfg.icon}"></i> <span>${topicName.replace(/_/g, " ")}</span>
            </div>
            <div class="pred-confidence">
              Confidence Score: <strong>${(maxProb * 100).toFixed(1)}%</strong>
            </div>
          </div>
          
          <div class="probs-section">
            <h4><i class="fa-solid fa-bars-staggered"></i> Topic Probabilities</h4>
            <div class="prob-bars-list">
              ${probHTML}
            </div>
          </div>

          <details class="details-card" style="margin-top: 1rem;">
            <summary><i class="fa-solid fa-code"></i> Raw ONNX Model Logits</summary>
            <pre class="json-code">${JSON.stringify(pred.logits, null, 2)}</pre>
          </details>
        </div>
      `;
      
      const div = document.createElement('div');
      div.innerHTML = cardHTML;
      resultsContainer.appendChild(div);
      
      // Save to history
      saveToHistory("Cora Node", {
        label: topicName.replace(/_/g, " "),
        confidence: (maxProb * 100).toFixed(1),
        icon: topicCfg.icon,
        color: topicCfg.color,
        nodeId: pred.node_index
      });
    });

    // Draw Citation Graph Canvases for all nodes
    const canvasesContainer = document.getElementById("citation-canvases-container");
    if (canvasesContainer) {
      canvasesContainer.innerHTML = "";
      data.predictions.forEach(pred => {
        const cfg = TOPIC_CONFIG[pred.predicted_label] || { color: "#6366f1" };
        const canvas = document.createElement("canvas");
        canvas.width = 450;
        canvas.height = 260;
        canvasesContainer.appendChild(canvas);
        drawCitationGraph(pred.node_index, cfg.color, canvas);
      });
    }

  } catch (err) {
    console.error("Prediction failed:", err);
    alert(`Failed to get GCN prediction: ${err.message}`);
  }
}

// Draw Canvas Citation Network
function drawCitationGraph(centralNodeId, nodeColor, canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;

  // Generate 6 simulated cited neighbor nodes
  const numNeighbors = 6;
  const neighbors = [];
  const radius = 80;

  for (let i = 0; i < numNeighbors; i++) {
    const angle = (i * 2 * Math.PI) / numNeighbors;
    neighbors.push({
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      id: (centralNodeId + (i + 1) * 37) % 2708
    });
  }

  // Draw Edge Lines
  ctx.lineWidth = 1.5;
  neighbors.forEach(n => {
    ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(n.x, n.y);
    ctx.stroke();
  });

  // Draw Central Node
  ctx.fillStyle = nodeColor;
  ctx.shadowColor = nodeColor;
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(centerX, centerY, 18, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Central Node Label
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`#${centralNodeId}`, centerX, centerY);

  // Draw Neighbor Nodes
  neighbors.forEach(n => {
    ctx.fillStyle = "rgba(30, 41, 59, 0.9)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(`#${n.id}`, n.x, n.y);
  });
}

// --- Custom Predictor ---
function initCustomPredictor() {
  const numNodesSelect = document.getElementById("num-custom-nodes");
  const radioButtons = document.querySelectorAll("input[name='feature-mode']");
  const payloadEditor = document.getElementById("custom-payload-editor");
  const btnPredictCustom = document.getElementById("btn-predict-custom");

  function updatePayload() {
    const numNodes = parseInt(numNodesSelect.value, 10);
    const mode = document.querySelector("input[name='feature-mode']:checked").value;

    const nodeFeatures = [];
    for (let i = 0; i < numNodes; i++) {
      const vec = new Array(1433).fill(0.0);
      if (mode === "preset_ai") {
        // Activate words associated with Neural Nets / AI
        [10, 25, 42, 100, 250, 500, 800, 1200].forEach(idx => vec[idx] = 1.0);
      } else if (mode === "preset_theory") {
        // Activate words associated with Math / Theory
        [5, 15, 30, 90, 300, 600, 900, 1400].forEach(idx => vec[idx] = 1.0);
      } else {
        // Random sparse indicators
        for (let k = 0; k < 10; k++) {
          vec[Math.floor(Math.random() * 1433)] = 1.0;
        }
      }
      nodeFeatures.push(vec);
    }

    // Build edges (chain or triangle)
    let edges = [[0, 1], [1, 0]];
    if (numNodes === 3) {
      edges = [[0, 1, 2, 1], [1, 2, 0, 0]];
    } else if (numNodes === 5) {
      edges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]];
    }

    const payload = {
      node_features: nodeFeatures,
      edge_indices: edges
    };

    payloadEditor.value = JSON.stringify(payload, null, 2);
  }

  numNodesSelect.addEventListener("change", updatePayload);
  radioButtons.forEach(r => r.addEventListener("change", updatePayload));
  updatePayload();

  btnPredictCustom.addEventListener("click", async () => {
    const resultsWrapper = document.getElementById("custom-results-wrapper");
    const resultsGrid = document.getElementById("custom-nodes-results-grid");

    try {
      const payload = JSON.parse(payloadEditor.value);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      resultsWrapper.classList.remove("hidden");
      resultsGrid.innerHTML = "";

      data.predictions.forEach(pred => {
        const topicName = pred.predicted_label;
        const cfg = TOPIC_CONFIG[topicName] || { color: "#6366f1", icon: "fa-file" };
        const probs = pred.probabilities || pred.probabilites || [];
        const maxProb = probs.length ? Math.max(...probs) : 0;

        const card = document.createElement("div");
        card.className = "card glass-card";
        card.innerHTML = `
          <div class="card-header">
            <h3><i class="fa-solid ${cfg.icon}" style="color: ${cfg.color}"></i> Paper #${pred.node_index + 1}</h3>
          </div>
          <div class="card-body">
            <div style="font-size: 1.1rem; font-weight: 700; color: ${cfg.color}; margin-bottom: 0.5rem">
              ${topicName.replace(/_/g, " ")}
            </div>
            <div style="font-size: 0.85rem; color: #9ca3af">
              Confidence: <strong style="color: #10b981">${(maxProb * 100).toFixed(1)}%</strong>
            </div>
          </div>
        `;
        resultsGrid.appendChild(card);
      });

      // Save custom graph result to history based on first node
      const firstPred = data.predictions[0];
      const maxProb = firstPred.probabilities ? Math.max(...firstPred.probabilities) : 0;
      const topicName = firstPred.predicted_label;
      const cfg = TOPIC_CONFIG[topicName] || { color: "#6366f1", icon: "fa-file" };
      
      saveToHistory("Custom Graph", {
        label: topicName.replace(/_/g, " "),
        confidence: (maxProb * 100).toFixed(1),
        icon: cfg.icon,
        color: cfg.color,
        nodes: data.predictions.length
      });

    } catch (err) {
      alert(`Custom prediction error: ${err.message}`);
    }
  });
}

// --- Model Info & Benchmark ---
async function initModelInfo() {
  const specFile = document.getElementById("spec-file");
  const specProvider = document.getElementById("spec-provider");
  const specInputX = document.getElementById("spec-input-x");
  const specInputEdge = document.getElementById("spec-input-edge");
  const specOutput = document.getElementById("spec-output");
  const btnBenchmark = document.getElementById("btn-run-benchmark");
  const benchMs = document.getElementById("bench-ms");
  const benchDetails = document.getElementById("bench-details");

  try {
    const res = await fetch("/info");
    const data = await res.json();
    const modelName = data["Model Name"] || data.model_name || "SimpleGCN";
    specFile.textContent = `${modelName} (simple_gcn_cora.onnx)`;
    if (data.inputs && data.inputs.length >= 2) {
      specInputX.textContent = `${data.inputs[0].name.replace(/_/g, " ")}: [num nodes, ${data.Feature_dimension || data.feature_dimension}]`;
      specInputEdge.textContent = `${data.inputs[1].name.replace(/_/g, " ")}: [2, num edges]`;
    }
    if (data.outputs && data.outputs.length >= 1) {
      specOutput.textContent = `${data.outputs[0].name.replace(/_/g, " ")}: [num nodes, ${data.num_classes}]`;
    }
  } catch (err) {
    console.error("Failed to load model info:", err);
  }

  btnBenchmark.addEventListener("click", async () => {
    btnBenchmark.disabled = true;
    btnBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Benchmark...';
    benchMs.textContent = "Calculating...";

    const timings = [];
    try {
      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        await fetch("/predict/cora_node", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_indices: [0, 10, 42, 100, 500] })
        });
        const elapsed = performance.now() - start;
        timings.push(elapsed);
      }

      const avgLatency = (timings.reduce((a, b) => a + b, 0) / timings.length).toFixed(2);
      benchMs.textContent = `${avgLatency} ms`;
      benchDetails.innerHTML = `
        <p><i class="fa-solid fa-circle-check" style="color: #10b981"></i> Average end-to-end API response time over 5 runs: <strong>${avgLatency} ms</strong>.</p>
      `;

    } catch (err) {
      benchMs.textContent = "Error";
      alert("Benchmark failed.");
    } finally {
      btnBenchmark.disabled = false;
      btnBenchmark.innerHTML = '<i class="fa-solid fa-stopwatch"></i> Run Latency Benchmark';
    }
  });
}

// ==========================================
// HISTORY & INTERACTIONS MODULE
// ==========================================

const HISTORY_KEY = 'cora_gnn_history';

document.addEventListener('DOMContentLoaded', () => {
  // History Sidebar DOM
  const btnToggleHistory = document.getElementById("btn-toggle-history");
  const btnCloseHistory = document.getElementById("btn-close-history");
  const btnClearHistory = document.getElementById("btn-clear-history");
  const historySidebar = document.getElementById("history-sidebar");
  const historyOverlay = document.getElementById("history-overlay");
  
  // Toggle Logic
  if (btnToggleHistory) btnToggleHistory.addEventListener('click', () => toggleHistory(true));
  if (btnCloseHistory) btnCloseHistory.addEventListener('click', () => toggleHistory(false));
  if (historyOverlay) historyOverlay.addEventListener('click', () => toggleHistory(false));
  
  if (btnClearHistory) {
    btnClearHistory.addEventListener('click', () => {
      localStorage.removeItem(HISTORY_KEY);
      renderHistory();
    });
  }

  function toggleHistory(show) {
    if (show) {
      historySidebar.classList.add('active');
      historyOverlay.classList.add('active');
      renderHistory();
    } else {
      historySidebar.classList.remove('active');
      historyOverlay.classList.remove('active');
    }
  }

  renderHistory();
  initInteractions();
});

// Run network canvas initialization immediately since script is at the bottom of the body
initNetworkCanvas();

function saveToHistory(type, data) {
  let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  history.unshift({
    type,
    data,
    timestamp: new Date().toISOString()
  });
  if (history.length > 50) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const historyList = document.getElementById("history-list");
  if (!historyList) return;
  
  const history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  
  if (history.length === 0) {
    historyList.innerHTML = `<li class="empty-history">No history yet. Run a prediction!</li>`;
    return;
  }
  
  historyList.innerHTML = '';
  history.forEach(item => {
    const time = new Date(item.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    const isCora = item.type === "Cora Node";
    const subtext = isCora ? `Node ID: #${item.data.nodeId}` : `Graph Size: ${item.data.nodes} nodes`;
    
    const li = document.createElement("li");
    li.className = "history-item";
    li.innerHTML = `
      <div class="history-item-header">
        <span>${item.type}</span>
        <span>${time}</span>
      </div>
      <div class="history-item-label" style="color: ${item.data.color}">
        <i class="fa-solid ${item.data.icon}"></i> ${item.data.label}
      </div>
      <div class="history-item-conf">
        ${subtext} &bull; Conf: <strong>${item.data.confidence}%</strong>
      </div>
    `;
    historyList.appendChild(li);
  });
}

function initInteractions() {
  // 3D Card Tilt
  document.querySelectorAll('.glass-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -3; 
      const rotateY = ((x - centerX) / centerX) * 3;
      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.01, 1.01, 1.01)`;
      card.style.borderColor = 'rgba(255,255,255,0.2)';
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
      card.style.borderColor = 'var(--bg-card-border)';
    });
  });

  // Ripple Buttons
  document.querySelectorAll('.btn-primary, .btn-emerald').forEach(btn => {
    btn.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.left = `${x}px`;
      ripple.style.top = `${y}px`;
      ripple.style.width = ripple.style.height = `${Math.max(rect.width, rect.height)}px`;
      ripple.style.transform = 'translate(-50%, -50%) scale(0)';
      
      this.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
    });
  });
}

// ==========================================
// BACKGROUND PARTICLE NETWORK ANIMATION
// ==========================================
function initNetworkCanvas() {
  const canvas = document.getElementById("network-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let width, height;
  let particles = [];

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  window.addEventListener("resize", resize);
  resize();

  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.7;
      this.vy = (Math.random() - 0.5) * 0.7;
      this.baseRadius = Math.random() * 1.5 + 1;
      this.pulsePhase = Math.random() * Math.PI * 2;
      this.pulseSpeed = Math.random() * 0.03 + 0.01;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;
      
      this.pulsePhase += this.pulseSpeed;
      this.radius = this.baseRadius + Math.sin(this.pulsePhase) * 1.2;
      if (this.radius < 0.5) this.radius = 0.5;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.shadowBlur = 15;
      ctx.shadowColor = "rgba(6, 182, 212, 0.8)";
      ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  for (let i = 0; i < 70; i++) {
    particles.push(new Particle());
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    
    particles.forEach(p => {
      p.update();
      p.draw();
    });

    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 160) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          
          const opacity = 0.3 * (1 - dist / 160);
          ctx.strokeStyle = `rgba(99, 102, 241, ${opacity})`;
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  
  animate();
}
