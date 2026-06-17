// =====================================================================
// SIMULADOR VISUAL E INTERATIVO DE ALGORITMOS (TURNOVER)
// =====================================================================

// 1. Dados Fixos e Estruturas para Regressão Logística
const lrPoints = [
  // Classe 0: Fica (azul)
  {x: 0.55, y: 0.25, label: 0},
  {x: 0.65, y: 0.30, label: 0},
  {x: 0.75, y: 0.20, label: 0},
  {x: 0.85, y: 0.25, label: 0},
  {x: 0.50, y: 0.40, label: 0},
  {x: 0.60, y: 0.45, label: 0},
  {x: 0.70, y: 0.35, label: 0},
  {x: 0.80, y: 0.40, label: 0},
  {x: 0.90, y: 0.30, label: 0},
  {x: 0.55, y: 0.55, label: 0},
  {x: 0.65, y: 0.50, label: 0},
  {x: 0.75, y: 0.60, label: 0},
  {x: 0.82, y: 0.48, label: 0},
  {x: 0.70, y: 0.52, label: 0},
  {x: 0.60, y: 0.35, label: 0},
  // Classe 1: Sai (vermelho)
  {x: 0.15, y: 0.75, label: 1},
  {x: 0.25, y: 0.70, label: 1},
  {x: 0.35, y: 0.80, label: 1},
  {x: 0.10, y: 0.85, label: 1},
  {x: 0.20, y: 0.90, label: 1},
  {x: 0.30, y: 0.65, label: 1},
  {x: 0.45, y: 0.75, label: 1},
  {x: 0.40, y: 0.85, label: 1},
  {x: 0.25, y: 0.55, label: 1},
  {x: 0.85, y: 0.80, label: 1},
  {x: 0.90, y: 0.85, label: 1},
  {x: 0.95, y: 0.75, label: 1},
  {x: 0.80, y: 0.90, label: 1},
  {x: 0.75, y: 0.78, label: 1},
  {x: 0.32, y: 0.92, label: 1}
];

// 2. Estrutura de Comitê (3 Árvores) para Random Forest
const rfTrees = [
  {
    nodes: [
      {x: 80, y: 20, label: "Satis ≤ 0,5", feature: "sat", threshold: 0.5, leftChild: 1, rightChild: 2},
      {x: 40, y: 70, label: "Horas ≤ 250", feature: "horas", threshold: 250, leftChild: 3, rightChild: 4},
      {x: 120, y: 70, label: "Tempo ≤ 4", feature: "tempo", threshold: 4, leftChild: 5, rightChild: 6},
      {x: 20, y: 120, leaf: true, prediction: 0, label: "Fica 🔵" },
      {x: 60, y: 120, leaf: true, prediction: 1, label: "Sai 🔴" },
      {x: 100, y: 120, leaf: true, prediction: 0, label: "Fica 🔵" },
      {x: 140, y: 120, leaf: true, prediction: 1, label: "Sai 🔴" }
    ],
    edges: [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
  },
  {
    nodes: [
      {x: 80, y: 20, label: "Horas ≤ 220", feature: "horas", threshold: 220, leftChild: 1, rightChild: 2},
      {x: 40, y: 70, label: "Satis ≤ 0,2", feature: "sat", threshold: 0.2, leftChild: 3, rightChild: 4},
      {x: 120, y: 70, label: "Projetos ≤ 5", feature: "projetos", threshold: 5, leftChild: 5, rightChild: 6},
      {x: 20, y: 120, leaf: true, prediction: 1, label: "Sai 🔴" },
      {x: 60, y: 120, leaf: true, prediction: 0, label: "Fica 🔵" },
      {x: 100, y: 120, leaf: true, prediction: 0, label: "Fica 🔵" },
      {x: 140, y: 120, leaf: true, prediction: 1, label: "Sai 🔴" }
    ],
    edges: [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
  },
  {
    nodes: [
      {x: 80, y: 20, label: "Satis ≤ 0,3", feature: "sat", threshold: 0.3, leftChild: 1, rightChild: 2},
      {x: 40, y: 70, leaf: true, prediction: 1, label: "Sai 🔴" },
      {x: 120, y: 70, label: "Horas ≤ 260", feature: "horas", threshold: 260, leftChild: 3, rightChild: 4},
      {x: 100, y: 120, leaf: true, prediction: 0, label: "Fica 🔵" },
      {x: 140, y: 120, leaf: true, prediction: 1, label: "Sai 🔴" }
    ],
    edges: [[0, 1], [0, 2], [2, 3], [2, 4]]
  }
];

// 3. Pontos de Teste para Ajuste Sequencial de Gradient Boosting
const gbPoints = [
  {x: 0.1, y: 0.90, label: 'A'},
  {x: 0.3, y: 0.80, label: 'B'},
  {x: 0.5, y: 0.10, label: 'C'},
  {x: 0.7, y: 0.20, label: 'D'},
  {x: 0.9, y: 0.85, label: 'E'}
];

// 4. Grade de Parâmetros e Resultados para GridSearchCV
const gsCombinations = [
  {d: 2, est: 10, score: 0.78},
  {d: 2, est: 50, score: 0.84},
  {d: 2, est: 100, score: 0.85},
  {d: 4, est: 10, score: 0.89},
  {d: 4, est: 50, score: 0.94},
  {d: 4, est: 100, score: 0.96}, // Melhor
  {d: 6, est: 10, score: 0.87},
  {d: 6, est: 50, score: 0.93},
  {d: 6, est: 100, score: 0.95}
];

// =====================================================================
// GERENCIADOR DE NAVEGAÇÃO DE ABAS
// =====================================================================
function switchVisTab(tabId) {
  document.querySelectorAll('.vis-tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  document.querySelectorAll('.vis-tab-content').forEach(content => {
    content.classList.remove('active');
  });
  
  // Encontra o botão correspondente e ativa
  const activeBtn = Array.from(document.querySelectorAll('.vis-tab-btn')).find(btn => btn.getAttribute('onclick').includes(tabId));
  if (activeBtn) activeBtn.classList.add('active');
  
  const activeContent = document.getElementById(tabId);
  if (activeContent) activeContent.classList.add('active');
  
  // Carrega e desenha dinamicamente os elementos de cada aba
  if (tabId === 'tab-logistic') {
    initLogisticRegression();
    updateLogisticRegression();
  } else if (tabId === 'tab-forest') {
    initRandomForest();
  } else if (tabId === 'tab-boosting') {
    updateGBPlot();
  } else if (tabId === 'tab-gridsearch') {
    initGridSearchVisuals();
  }
}

// =====================================================================
// LOGIC REGRESSION (REGRESSÃO LOGÍSTICA) LOGIC
// =====================================================================
let lrInitialized = false;

function initLogisticRegression() {
  if (lrInitialized) return;
  const g = document.getElementById('lr-points-group');
  if (!g) return;
  g.innerHTML = '';
  
  lrPoints.forEach((pt, idx) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const cx = 20 + pt.x * 260;
    const cy = 280 - pt.y * 260;
    circle.setAttribute('cx', cx);
    circle.setAttribute('cy', cy);
    circle.setAttribute('r', 6.5);
    circle.setAttribute('id', `lr-point-${idx}`);
    circle.style.cursor = 'pointer';
    
    circle.addEventListener('mouseover', () => highlightLRPoint(pt));
    circle.addEventListener('mouseout', () => clearLRHighlight());
    
    g.appendChild(circle);
  });
  
  drawSigmoidCurve();
  lrInitialized = true;
}

function getSigmoidY(z) {
  const p = 1.0 / (1.0 + Math.exp(-z));
  return 110 - p * 100;
}

function drawSigmoidCurve() {
  let d = "";
  for (let x = 20; x <= 280; x++) {
    const z = -6.0 + ((x - 20) / 260.0) * 12.0;
    const y = getSigmoidY(z);
    if (x === 20) d += `M ${x} ${y}`;
    else d += ` L ${x} ${y}`;
  }
  document.getElementById('lr-sigmoid-curve').setAttribute('d', d);
}

function updateLogisticRegression() {
  const w1 = +document.getElementById('lr-slider-w1').value;
  const w2 = +document.getElementById('lr-slider-w2').value;
  const b = +document.getElementById('lr-slider-b').value;
  
  document.getElementById('lr-val-w1').textContent = w1.toFixed(1).replace('.', ',');
  document.getElementById('lr-val-w2').textContent = w2.toFixed(1).replace('.', ',');
  document.getElementById('lr-val-b').textContent = b.toFixed(1).replace('.', ',');
  
  const mathBox = document.getElementById('lr-math-display');
  if (mathBox) {
    mathBox.innerHTML = `
      z = (${w1.toFixed(1)}) × Satis + (${w2.toFixed(1)}) × Horas + (${b.toFixed(1)})<br>
      P(Turnover) = 1 / (1 + e<sup>-z</sup>)
    `;
  }
  
  // Desenha a reta limite de decisão (z = w1 * x + w2 * y + b = 0 => y = -(w1*x + b)/w2)
  const line = document.getElementById('lr-boundary-line');
  if (line) {
    if (Math.abs(w2) > 0.001) {
      const y0 = -b / w2;
      const y1 = -(w1 + b) / w2;
      
      const Y0_svg = 280 - y0 * 260;
      const Y1_svg = 280 - y1 * 260;
      
      line.setAttribute('x1', 20);
      line.setAttribute('y1', Y0_svg);
      line.setAttribute('x2', 280);
      line.setAttribute('y2', Y1_svg);
    } else {
      const x = -b / w1;
      const X_svg = 20 + x * 260;
      line.setAttribute('x1', X_svg);
      line.setAttribute('y1', 20);
      line.setAttribute('x2', X_svg);
      line.setAttribute('y2', 280);
    }
  }
  
  // Classifica os pontos em tempo real e calcula estatísticas
  let tp = 0, fp = 0, tn = 0, fn = 0;
  lrPoints.forEach((pt, idx) => {
    const z = w1 * pt.x + w2 * pt.y + b;
    const p = 1.0 / (1.0 + Math.exp(-z));
    const pred = p >= 0.5 ? 1 : 0;
    
    if (pt.label === 1 && pred === 1) tp++;
    else if (pt.label === 0 && pred === 1) fp++;
    else if (pt.label === 0 && pred === 0) tn++;
    else if (pt.label === 1 && pred === 0) fn++;
    
    const circle = document.getElementById(`lr-point-${idx}`);
    if (circle) {
      circle.setAttribute('fill', pred === 1 ? 'rgba(239, 68, 68, 0.25)' : 'rgba(59, 130, 246, 0.25)');
      circle.setAttribute('stroke', pt.label === 1 ? '#ef4444' : '#3b82f6');
      circle.setAttribute('stroke-width', pt.label !== pred ? '3.5' : '1.5');
      if (pt.label !== pred) {
        circle.setAttribute('stroke-dasharray', '2 1.5');
      } else {
        circle.removeAttribute('stroke-dasharray');
      }
    }
  });
  
  const total = lrPoints.length;
  const acc = (tp + tn) / total;
  const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  
  document.getElementById('lr-metric-acc').textContent = (acc * 100).toFixed(0) + '%';
  document.getElementById('lr-metric-prec').textContent = (prec * 100).toFixed(0) + '%';
  document.getElementById('lr-metric-rec').textContent = (rec * 100).toFixed(0) + '%';
}

function highlightLRPoint(pt) {
  const w1 = +document.getElementById('lr-slider-w1').value;
  const w2 = +document.getElementById('lr-slider-w2').value;
  const b = +document.getElementById('lr-slider-b').value;
  
  const z = w1 * pt.x + w2 * pt.y + b;
  const p = 1.0 / (1.0 + Math.exp(-z));
  
  const zClamped = Math.max(-6, Math.min(6, z));
  const xSig = 20 + ((zClamped + 6) / 12.0) * 260;
  const ySig = 110 - p * 100;
  
  const sigLine = document.getElementById('lr-sig-proj-line');
  const sigDot = document.getElementById('lr-sig-proj-dot');
  
  if (sigLine && sigDot) {
    sigLine.setAttribute('x1', xSig);
    sigLine.setAttribute('y1', 60);
    sigLine.setAttribute('x2', xSig);
    sigLine.setAttribute('y2', ySig);
    sigLine.style.display = 'block';
    
    sigDot.setAttribute('cx', xSig);
    sigDot.setAttribute('cy', ySig);
    sigDot.style.display = 'block';
  }
  
  const mathBox = document.getElementById('lr-math-display');
  if (mathBox) {
    mathBox.innerHTML = `
      <strong>Ponto Selecionado:</strong> Satis: ${pt.x.toFixed(2)}, Horas: ${pt.y.toFixed(2)}<br>
      z = (${w1.toFixed(1)})×${pt.x.toFixed(2)} + (${w2.toFixed(1)})×${pt.y.toFixed(2)} + (${b.toFixed(1)}) = <strong>${z.toFixed(2)}</strong><br>
      P(Turnover) = 1 / (1 + e<sup>-${z.toFixed(2)}</sup>) = <strong>${(p * 100).toFixed(1)}%</strong>
    `;
  }
}

function clearLRHighlight() {
  const sigLine = document.getElementById('lr-sig-proj-line');
  const sigDot = document.getElementById('lr-sig-proj-dot');
  if (sigLine) sigLine.style.display = 'none';
  if (sigDot) sigDot.style.display = 'none';
  updateLogisticRegression();
}

// =====================================================================
// RANDOM FOREST LOGIC
// =====================================================================
let rfInitialized = false;

function initRandomForest() {
  if (rfInitialized) return;
  drawForestTrees();
  updateForestPrediction();
  rfInitialized = true;
}

function drawForestTrees() {
  rfTrees.forEach((tree, treeIdx) => {
    const nodesG = document.getElementById(`rf-tree-${treeIdx+1}-nodes`);
    const edgesG = document.getElementById(`rf-tree-${treeIdx+1}-edges`);
    if (!nodesG || !edgesG) return;
    nodesG.innerHTML = '';
    edgesG.innerHTML = '';
    
    // Conecta as arestas
    tree.edges.forEach(edge => {
      const fromNode = tree.nodes[edge[0]];
      const toNode = tree.nodes[edge[1]];
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute('x1', fromNode.x);
      line.setAttribute('y1', fromNode.y);
      line.setAttribute('x2', toNode.x);
      line.setAttribute('y2', toNode.y);
      line.setAttribute('id', `rf-edge-${treeIdx+1}-${edge[0]}->${edge[1]}`);
      line.setAttribute('class', 'rf-edge');
      edgesG.appendChild(line);
    });
    
    // Desenha os nós
    tree.nodes.forEach((node, nodeIdx) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute('cx', node.x);
      circle.setAttribute('cy', node.y);
      circle.setAttribute('r', 7.5);
      circle.setAttribute('id', `rf-node-${treeIdx+1}-${nodeIdx}`);
      let nodeClass = 'rf-node';
      if (node.leaf) {
        nodeClass += node.prediction === 0 ? ' leaf-fica' : ' leaf-sai';
      }
      circle.setAttribute('class', nodeClass);
      nodesG.appendChild(circle);
      
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute('x', node.x);
      text.setAttribute('y', node.leaf ? node.y + 16 : node.y - 11);
      text.setAttribute('font-size', '6.5px');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#020617');
      text.setAttribute('font-weight', node.leaf ? '700' : '600');
      text.textContent = node.label;
      nodesG.appendChild(text);
    });
  });
}

function traverseTree(nodes, inputs) {
  let path = [0];
  let edges = [];
  let curr = 0;
  while (!nodes[curr].leaf) {
    const node = nodes[curr];
    const val = inputs[node.feature];
    let nextNodeIdx = (val <= node.threshold) ? node.leftChild : node.rightChild;
    edges.push(`${curr}->${nextNodeIdx}`);
    path.push(nextNodeIdx);
    curr = nextNodeIdx;
  }
  return { prediction: nodes[curr].prediction, path, edges };
}

function updateForestPrediction() {
  if (rfAnimating) return;
  
  const sat = +document.getElementById('rf-slider-sat').value;
  const horas = +document.getElementById('rf-slider-horas').value;
  const projetos = +document.getElementById('rf-slider-projetos').value;
  const tempo = +document.getElementById('rf-slider-tempo').value;
  
  document.getElementById('rf-val-sat').textContent = sat.toFixed(2).replace('.', ',');
  document.getElementById('rf-val-horas').textContent = horas;
  document.getElementById('rf-val-projetos').textContent = projetos;
  document.getElementById('rf-val-tempo').textContent = tempo;
  
  const inputs = { sat, horas, projetos, tempo };
  
  const paths = [
    traverseTree(rfTrees[0].nodes, inputs),
    traverseTree(rfTrees[1].nodes, inputs),
    traverseTree(rfTrees[2].nodes, inputs)
  ];
  
  paths.forEach((p, treeIdx) => {
    const treeNum = treeIdx + 1;
    document.querySelectorAll(`#rf-tree-${treeNum} .rf-node`).forEach(el => el.classList.remove('active-node'));
    document.querySelectorAll(`#rf-tree-${treeNum} .rf-edge`).forEach(el => el.classList.remove('active-edge'));
    
    p.path.forEach(nodeIdx => {
      document.getElementById(`rf-node-${treeNum}-${nodeIdx}`).classList.add('active-node');
    });
    p.edges.forEach(edgeStr => {
      const edgeEl = document.getElementById(`rf-edge-${treeNum}-${edgeStr}`);
      if (edgeEl) edgeEl.classList.add('active-edge');
    });
    
    const leafNode = rfTrees[treeIdx].nodes[p.path[p.path.length-1]];
    document.getElementById(`rf-tree-${treeNum}-pred`).innerHTML = leafNode.prediction === 1 ? 'Voto: <span style="color:var(--danger)">Sai 🔴</span>' : 'Voto: <span style="color:var(--success)">Fica 🔵</span>';
  });
  
  updateConsensusUI(paths[0].prediction, paths[1].prediction, paths[2].prediction);
}

function updateConsensusUI(p1, p2, p3) {
  const votes = [p1, p2, p3];
  const fica = votes.filter(v => v === 0).length;
  const sai = votes.filter(v => v === 1).length;
  
  document.getElementById('rf-vote-fica-count').textContent = `🔵 Fica: ${fica}`;
  document.getElementById('rf-vote-sai-count').textContent = `🔴 Sai: ${sai}`;
  
  const consensusEl = document.getElementById('rf-final-consensus');
  if (sai > fica) {
    consensusEl.innerHTML = 'Consenso: <span style="color:var(--danger)">🔴 RISCO DE TURNOVER (Maioria: ' + sai + ' vs ' + fica + ')</span>';
  } else {
    consensusEl.innerHTML = 'Consenso: <span style="color:var(--success)">🔵 BAIXO RISCO (Maioria: ' + fica + ' vs ' + sai + ')</span>';
  }
}

let rfAnimating = false;
function animateForestPath() {
  if (rfAnimating) return;
  rfAnimating = true;
  
  const sat = +document.getElementById('rf-slider-sat').value;
  const horas = +document.getElementById('rf-slider-horas').value;
  const projetos = +document.getElementById('rf-slider-projetos').value;
  const tempo = +document.getElementById('rf-slider-tempo').value;
  
  const inputs = { sat, horas, projetos, tempo };
  
  document.getElementById('rf-slider-sat').disabled = true;
  document.getElementById('rf-slider-horas').disabled = true;
  document.getElementById('rf-slider-projetos').disabled = true;
  document.getElementById('rf-slider-tempo').disabled = true;
  
  for (let t = 1; t <= 3; t++) {
    document.querySelectorAll(`#rf-tree-${t} .rf-node`).forEach(el => el.classList.remove('active-node'));
    document.querySelectorAll(`#rf-tree-${t} .rf-edge`).forEach(el => el.classList.remove('active-edge'));
    document.getElementById(`rf-tree-${t}-pred`).textContent = "Voto: simulando...";
  }
  document.getElementById('rf-final-consensus').textContent = "Calculando votos...";
  
  const paths = [
    traverseTree(rfTrees[0].nodes, inputs),
    traverseTree(rfTrees[1].nodes, inputs),
    traverseTree(rfTrees[2].nodes, inputs)
  ];
  
  let finishedCount = 0;
  
  paths.forEach((p, treeIdx) => {
    const treeNum = treeIdx + 1;
    const coords = p.path.map(nodeIdx => {
      const nd = rfTrees[treeIdx].nodes[nodeIdx];
      return { x: nd.x, y: nd.y, idx: nodeIdx };
    });
    
    const particle = document.getElementById(`rf-tree-${treeNum}-particle`);
    if (!particle) return;
    particle.setAttribute('cx', coords[0].x);
    particle.setAttribute('cy', coords[0].y);
    particle.style.display = 'block';
    
    document.getElementById(`rf-node-${treeNum}-0`).classList.add('active-node');
    
    let step = 0;
    function nextStep() {
      if (step >= coords.length - 1) {
        particle.style.display = 'none';
        const leafNodeIdx = p.path[p.path.length-1];
        const leafNode = rfTrees[treeIdx].nodes[leafNodeIdx];
        document.getElementById(`rf-tree-${treeNum}-pred`).innerHTML = leafNode.prediction === 1 ? 'Voto: <span style="color:var(--danger)">Sai 🔴</span>' : 'Voto: <span style="color:var(--success)">Fica 🔵</span>';
        
        finishedCount++;
        if (finishedCount === 3) {
          rfAnimating = false;
          document.getElementById('rf-slider-sat').disabled = false;
          document.getElementById('rf-slider-horas').disabled = false;
          document.getElementById('rf-slider-projetos').disabled = false;
          document.getElementById('rf-slider-tempo').disabled = false;
          updateConsensusUI(paths[0].prediction, paths[1].prediction, paths[2].prediction);
        }
        return;
      }
      
      const start = coords[step];
      const end = coords[step+1];
      let startTime = null;
      const duration = 280;
      
      function animate(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1.0);
        const cx = start.x + (end.x - start.x) * progress;
        const cy = start.y + (end.y - start.y) * progress;
        particle.setAttribute('cx', cx);
        particle.setAttribute('cy', cy);
        
        if (progress < 1.0) {
          requestAnimationFrame(animate);
        } else {
          const nextNodeIdx = end.idx;
          document.getElementById(`rf-node-${treeNum}-${nextNodeIdx}`).classList.add('active-node');
          
          const edgeId = `rf-edge-${treeNum}-${start.idx}->${end.idx}`;
          const edgeEl = document.getElementById(edgeId);
          if (edgeEl) edgeEl.classList.add('active-edge');
          
          step++;
          nextStep();
        }
      }
      requestAnimationFrame(animate);
    }
    nextStep();
  });
}

// =====================================================================
// GRADIENT BOOSTING LOGIC
// =====================================================================
let gbCurrentStep = 0;

function changeGBStep(delta) {
  gbCurrentStep = Math.max(0, Math.min(3, gbCurrentStep + delta));
  
  document.getElementById('gb-btn-prev').disabled = (gbCurrentStep === 0);
  document.getElementById('gb-btn-next').disabled = (gbCurrentStep === 3);
  
  updateGBPlot();
}

function getGBPrediction(x, step, lr) {
  let val = 0.5; // F0 (Prior)
  if (step >= 1) {
    const h1 = (x < 0.4 || x > 0.8) ? 0.65 : -0.65;
    val += lr * h1;
  }
  if (step >= 2) {
    let h2;
    if (x < 0.25) h2 = 0.25;
    else if (x > 0.6 && x < 0.85) h2 = -0.3;
    else h2 = 0.08;
    val += lr * h2;
  }
  if (step >= 3) {
    const h3 = (x > 0.4 && x < 0.6) ? -0.35 : 0.07;
    val += lr * h3;
  }
  return Math.max(0, Math.min(1, val));
}

function getWeakPrediction(x, treeIdx) {
  if (treeIdx === 1) {
    return (x < 0.4 || x > 0.8) ? 0.65 : -0.65;
  } else if (treeIdx === 2) {
    if (x < 0.25) return 0.25;
    else if (x > 0.6 && x < 0.85) return -0.3;
    else return 0.08;
  } else if (treeIdx === 3) {
    return (x > 0.4 && x < 0.6) ? -0.35 : 0.07;
  }
  return 0;
}

function updateGBPlot() {
  const lr = +document.getElementById('gb-slider-lr').value;
  document.getElementById('gb-val-lr').textContent = lr.toFixed(2).replace('.', ',');
  
  const stepLabels = [
    "Passo 0: Valor Inicial (Prior)",
    "Passo 1: F₀(x) + η × h₁(x)",
    "Passo 2: F₁(x) + η × h₂(x)",
    "Passo 3: F₂(x) + η × h₃(x)"
  ];
  document.getElementById('gb-step-indicator').textContent = stepLabels[gbCurrentStep];
  
  const mathDisplay = document.getElementById('gb-math-display');
  if (gbCurrentStep === 0) {
    mathDisplay.innerHTML = `F₀(x) = 0,50 (Prior Constante)`;
  } else if (gbCurrentStep === 1) {
    mathDisplay.innerHTML = `F₁(x) = F₀(x) + ${lr.toFixed(2)} × h₁(x)`;
  } else if (gbCurrentStep === 2) {
    mathDisplay.innerHTML = `F₂(x) = F₁(x) + ${lr.toFixed(2)} × h₂(x)`;
  } else {
    mathDisplay.innerHTML = `F₃(x) = F₂(x) + ${lr.toFixed(2)} × h₃(x)`;
  }
  
  let predD = "";
  for (let Xsvg = 20; Xsvg <= 280; Xsvg++) {
    const x = (Xsvg - 20) / 260.0;
    const yPred = getGBPrediction(x, gbCurrentStep, lr);
    const Ysvg = 180 - yPred * 160;
    if (Xsvg === 20) predD += `M ${Xsvg} ${Ysvg}`;
    else predD += ` L ${Xsvg} ${Ysvg}`;
  }
  document.getElementById('gb-pred-line').setAttribute('d', predD);
  
  const residualsG = document.getElementById('gb-residuals-group');
  const pointsG = document.getElementById('gb-points-group');
  if (!residualsG || !pointsG) return;
  residualsG.innerHTML = '';
  pointsG.innerHTML = '';
  
  let sumSqError = 0;
  gbPoints.forEach(pt => {
    const cx = 20 + pt.x * 260;
    const cy = 180 - pt.y * 160;
    
    const predVal = getGBPrediction(pt.x, gbCurrentStep, lr);
    const cyPred = 180 - predVal * 160;
    
    const residual = pt.y - predVal;
    sumSqError += residual * residual;
    
    if (Math.abs(residual) > 0.005) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute('x1', cx);
      line.setAttribute('y1', cy);
      line.setAttribute('x2', cx);
      line.setAttribute('y2', cyPred);
      line.setAttribute('class', 'gb-residual-line');
      residualsG.appendChild(line);
    }
    
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute('cx', cx);
    circle.setAttribute('cy', cy);
    circle.setAttribute('r', 5);
    circle.setAttribute('fill', '#ef4444');
    circle.setAttribute('stroke', '#0f172a');
    circle.setAttribute('stroke-width', '1.5');
    
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `Ponto ${pt.label}: Alvo=${pt.y.toFixed(2)}, Pred=${predVal.toFixed(2)}, Residual=${residual.toFixed(2)}`;
    circle.appendChild(title);
    pointsG.appendChild(circle);
  });
  
  const mse = sumSqError / gbPoints.length;
  document.getElementById('gb-error-value').textContent = mse.toFixed(4).replace('.', ',');
  
  let weakD = "";
  if (gbCurrentStep === 0) {
    weakD = "M 20 50 L 280 50";
  } else {
    for (let Xsvg = 20; Xsvg <= 280; Xsvg++) {
      const x = (Xsvg - 20) / 260.0;
      const wPred = getWeakPrediction(x, gbCurrentStep);
      const Ysvg = 50 - wPred * 50;
      if (Xsvg === 20) weakD += `M ${Xsvg} ${Ysvg}`;
      else weakD += ` L ${Xsvg} ${Ysvg}`;
    }
  }
  document.getElementById('gb-weak-line').setAttribute('d', weakD);
}

// =====================================================================
// GRIDSEARCHCV + STRATIFIED K-FOLD LOGIC
// =====================================================================
let gsInitialized = false;
let gsRunning = false;
let gsCurrentIndex = 0;
let gsFoldsTimer = null;
let gsMainTimer = null;

function initGridSearchVisuals() {
  if (gsInitialized) return;
  
  const kfContainer = document.getElementById('kf-folds-box');
  if (!kfContainer) return;
  kfContainer.innerHTML = '';
  
  for (let fold = 1; fold <= 5; fold++) {
    const row = document.createElement('div');
    row.className = 'kf-fold-row';
    row.setAttribute('id', `kf-fold-row-${fold}`);
    
    const label = document.createElement('div');
    label.className = 'kf-fold-label';
    label.textContent = `Fold ${fold}`;
    row.appendChild(label);
    
    const blocksContainer = document.createElement('div');
    blocksContainer.className = 'kf-fold-blocks';
    
    for (let i = 0; i < 20; i++) {
      const block = document.createElement('div');
      block.className = 'kf-block';
      block.setAttribute('id', `kf-block-${fold}-${i}`);
      
      const isSai = i >= 15;
      const isBlueVal = !isSai && (i >= 3*(fold-1) && i < 3*fold);
      const isRedVal = isSai && (i - 15 === fold - 1);
      const isVal = isBlueVal || isRedVal;
      
      if (isVal) {
        block.className += isSai ? ' val-sai' : ' val-fica';
        block.textContent = 'V';
      } else {
        block.className += isSai ? ' train-sai' : ' train-fica';
        block.textContent = 'T';
      }
      
      block.title = isVal ? "Validação (Classe proporcional)" : "Treinamento (Classe proporcional)";
      blocksContainer.appendChild(block);
    }
    
    row.appendChild(blocksContainer);
    kfContainer.appendChild(row);
  }
  
  const gridContainer = document.getElementById('gs-grid-box');
  if (!gridContainer) return;
  gridContainer.innerHTML = '';
  
  const depths = [2, 4, 6];
  const estimators = [10, 50, 100];
  
  depths.forEach(d => {
    estimators.forEach(est => {
      const cell = document.createElement('div');
      cell.className = 'gs-cell';
      cell.setAttribute('id', `gs-cell-${d}-${est}`);
      
      const params = document.createElement('div');
      params.className = 'gs-cell-params';
      params.innerHTML = `max_depth: ${d}<br>n_est: ${est}`;
      cell.appendChild(params);
      
      const score = document.createElement('div');
      score.className = 'gs-cell-score';
      score.textContent = '-';
      score.setAttribute('id', `gs-score-${d}-${est}`);
      cell.appendChild(score);
      
      gridContainer.appendChild(cell);
    });
  });
  
  gsInitialized = true;
}

function startGridSearch() {
  const btn = document.getElementById('gs-btn-start');
  if (gsRunning) {
    gsRunning = false;
    btn.textContent = '▶ Retomar Otimização';
    clearTimeout(gsMainTimer);
    clearTimeout(gsFoldsTimer);
    if (gsCurrentIndex < gsCombinations.length) {
      const comb = gsCombinations[gsCurrentIndex];
      const cell = document.getElementById(`gs-cell-${comb.d}-${comb.est}`);
      if (cell) cell.classList.remove('evaluating');
    }
    
    document.querySelectorAll('.kf-fold-row').forEach(row => row.style.opacity = '1');
    document.querySelectorAll('.kf-block').forEach(b => b.classList.remove('inactive'));
    return;
  }
  
  gsRunning = true;
  btn.textContent = '⏸ Pausar Busca';
  document.getElementById('gs-summary-box').style.display = 'none';
  
  if (gsCurrentIndex >= gsCombinations.length) {
    resetGridSearch();
    gsRunning = true;
    btn.textContent = '⏸ Pausar Busca';
  }
  
  runGridSearchStep();
}

function resetGridSearch() {
  gsRunning = false;
  gsCurrentIndex = 0;
  clearTimeout(gsMainTimer);
  clearTimeout(gsFoldsTimer);
  
  const btn = document.getElementById('gs-btn-start');
  if (btn) btn.textContent = '▶ Iniciar Otimização';
  
  const summary = document.getElementById('gs-summary-box');
  if (summary) summary.style.display = 'none';
  
  gsCombinations.forEach(comb => {
    const cell = document.getElementById(`gs-cell-${comb.d}-${comb.est}`);
    const score = document.getElementById(`gs-score-${comb.d}-${comb.est}`);
    if (cell) cell.className = 'gs-cell';
    if (score) score.textContent = '-';
  });
  
  document.querySelectorAll('.kf-fold-row').forEach(row => row.style.opacity = '1');
  document.querySelectorAll('.kf-block').forEach(b => b.classList.remove('inactive'));
}

function runGridSearchStep() {
  if (!gsRunning) return;
  
  if (gsCurrentIndex >= gsCombinations.length) {
    gsRunning = false;
    document.getElementById('gs-btn-start').textContent = '▶ Iniciar Otimização';
    
    const bestCell = document.getElementById('gs-cell-4-100');
    if (bestCell) {
      bestCell.classList.add('best-cell');
      const score = document.getElementById('gs-score-4-100');
      if (score) score.innerHTML = '★ 0,960';
    }
    
    document.getElementById('gs-summary-box').style.display = 'block';
    return;
  }
  
  const comb = gsCombinations[gsCurrentIndex];
  const cell = document.getElementById(`gs-cell-${comb.d}-${comb.est}`);
  const scoreText = document.getElementById(`gs-score-${comb.d}-${comb.est}`);
  
  if (cell) cell.classList.add('evaluating');
  
  const speedVal = +document.getElementById('gs-slider-speed').value;
  const cellDuration = speedVal === 1 ? 2500 : speedVal === 2 ? 1000 : 350;
  const foldDuration = cellDuration / 5;
  
  let fold = 1;
  
  function animateFolds() {
    if (!gsRunning) return;
    if (fold > 5) {
      if (cell) {
        cell.classList.remove('evaluating');
        cell.classList.add('done');
      }
      if (scoreText) scoreText.textContent = comb.score.toFixed(3).replace('.', ',');
      
      document.querySelectorAll('.kf-fold-row').forEach(row => row.style.opacity = '1');
      document.querySelectorAll('.kf-block').forEach(b => b.classList.remove('inactive'));
      
      gsCurrentIndex++;
      gsMainTimer = setTimeout(runGridSearchStep, 150);
      return;
    }
    
    for (let f = 1; f <= 5; f++) {
      const row = document.getElementById(`kf-fold-row-${f}`);
      if (row) {
        if (f === fold) {
          row.style.opacity = '1';
          document.querySelectorAll(`#kf-fold-row-${f} .kf-block`).forEach(b => b.classList.remove('inactive'));
        } else {
          row.style.opacity = '0.3';
          document.querySelectorAll(`#kf-fold-row-${f} .kf-block`).forEach(b => b.classList.add('inactive'));
        }
      }
    }
    
    if (scoreText) {
      scoreText.textContent = `CV: ${fold}/5...`;
    }
    
    fold++;
    gsFoldsTimer = setTimeout(animateFolds, foldDuration);
  }
  
  animateFolds();
}

// =====================================================================
// EVENT LISTENERS DE INICIALIZAÇÃO E SLIDERS
// =====================================================================
document.addEventListener('DOMContentLoaded', () => {
  // Inicializa a primeira aba
  switchVisTab('tab-logistic');
  
  // Ouvintes da Regressão Logística
  document.getElementById('lr-slider-w1').addEventListener('input', updateLogisticRegression);
  document.getElementById('lr-slider-w2').addEventListener('input', updateLogisticRegression);
  document.getElementById('lr-slider-b').addEventListener('input', updateLogisticRegression);
  
  // Ouvintes do Random Forest
  document.getElementById('rf-slider-sat').addEventListener('input', updateForestPrediction);
  document.getElementById('rf-slider-horas').addEventListener('input', updateForestPrediction);
  document.getElementById('rf-slider-projetos').addEventListener('input', updateForestPrediction);
  document.getElementById('rf-slider-tempo').addEventListener('input', updateForestPrediction);
  
  // Ouvintes do Gradient Boosting
  document.getElementById('gb-slider-lr').addEventListener('input', updateGBPlot);
  
  // Ouvinte da velocidade do Grid Search
  document.getElementById('gs-slider-speed').addEventListener('input', function() {
    const val = +this.value;
    const labels = ["Lenta", "Média", "Rápida"];
    document.getElementById('gs-val-speed').textContent = labels[val-1];
  });
});
