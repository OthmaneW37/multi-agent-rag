/**
 * WAC Sport Analytics — Interface SPA Multi-Agents
 * Connectee au backend FastAPI avec donnees reelles ScrappingDataBotola
 */

const API_BASE = "http://localhost:8000";

/* ============ CONFIG ============ */
const AGENTS = {
  orchestrateur: {
    id: 'orchestrateur',
    name: 'Orchestrateur',
    role: 'Coordination du pipeline',
    icon: '⚙️',
    color: '#FFFFFF',
    class: 'agent-orchestrateur',
  },
  scout: {
    id: 'scout',
    name: 'Scout',
    role: 'Recherche & collecte RAG',
    icon: '🔍',
    color: '#06B6D4',
    class: 'agent-scout',
  },
  modelisateur: {
    id: 'modelisateur',
    name: 'Modélisateur',
    role: 'Analyse & tendances',
    icon: '📊',
    color: '#F59E0B',
    class: 'agent-modelisateur',
  },
  tacticien: {
    id: 'tacticien',
    name: 'Tacticien',
    role: 'Stratégie & composition',
    icon: '📋',
    color: '#10B981',
    class: 'agent-tacticien',
  },
  validateur: {
    id: 'validateur',
    name: 'Validateur',
    role: 'Qualité & compilation',
    icon: '✓',
    color: '#8B5CF6',
    class: 'agent-validateur',
  },
};

const ADVERSAIRES_VALIDES = [
  "Raja CA", "FAR Rabat", "FUS Rabat", "Ittihad Tanger",
  "Olympic Safi", "Hassania Agadir", "Maghreb Fes",
  "Kawkab Marrakech", "RSB Berkane", "DH El Jadida",
  "CODM Meknes", "Yacoub El Mansour", "UTS Rabat",
  "Olympique Dcheira", "CR Khemis Zemamra",
];

/* ============ STATE ============ */
let isRunning = false;
let currentAgent = null;

/* ============ DOM REFS ============ */
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const agentsList = document.getElementById('agentsList');
const animationBanner = document.getElementById('animationBanner');
const animStage = document.getElementById('animStage');
const animLabel = document.getElementById('animLabel');
const animProgress = document.getElementById('animProgress');
const animText = document.getElementById('animText');
const activeAgentIndicator = document.getElementById('activeAgentIndicator');

/* ============ MARKDOWN RENDERER ============ */
function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderMarkdown(text) {
  if (!text) return '';
  let raw = escapeHtml(text);
  raw = raw.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  raw = raw.replace(/\*(.+?)\*/g, '<em>$1</em>');

  const lines = raw.split('\n');
  const out = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    const trimmed = line.trim();

    if (trimmed === '') {
      if (inList) { out.push('</ul>'); inList = false; }
      out.push('<br>');
      continue;
    }

    const listMatch = trimmed.match(/^[•\-]\s+(.*)/);
    if (listMatch) {
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push(`<li>${listMatch[1]}</li>`);
      continue;
    }

    if (inList) { out.push('</ul>'); inList = false; }

    const h2Match = trimmed.match(/^##\s+(.+)/);
    if (h2Match) { out.push(`<h2>${h2Match[1]}</h2>`); continue; }

    const h3Match = trimmed.match(/^###\s+(.+)/);
    if (h3Match) { out.push(`<h3>${h3Match[1]}</h3>`); continue; }

    const bqMatch = trimmed.match(/^>\s*(.*)/);
    if (bqMatch) { out.push(`<blockquote>${bqMatch[1]}</blockquote>`); continue; }

    if (trimmed.includes('|') && trimmed.replace(/\|/g, '').trim() === '') {
      out.push('<hr>'); continue;
    }

    out.push(`<p>${line}</p>`);
  }

  if (inList) out.push('</ul>');
  let html = out.join('');
  html = html.replace(/(<br>\s*){2,}/g, '<br>');
  return html;
}

/* ============ INIT SIDEBAR ============ */
function initSidebar() {
  Object.values(AGENTS).forEach((agent) => {
    if (agent.id === 'orchestrateur') return;
    const li = document.createElement('li');
    li.className = `agent-item ${agent.class}`;
    li.id = `agent-row-${agent.id}`;
    li.dataset.agent = agent.id;
    li.innerHTML = `
      <div class="agent-icon">${agent.icon}</div>
      <div class="agent-info">
        <div class="agent-name">${agent.name}</div>
        <div class="agent-role">${agent.role}</div>
      </div>
      <div class="agent-status status-idle" id="status-${agent.id}">Inactif</div>
      <div class="agent-progress"><div class="fill" id="progress-${agent.id}"></div></div>
    `;
    agentsList.appendChild(li);
  });
}

/* ============ HELPERS ============ */
function nowTime() {
  return new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/* ============ MESSAGES ============ */
function addMessage(agentId, textOrHtml, isTyping = false) {
  const agent = AGENTS[agentId] || AGENTS.orchestrateur;
  const msg = document.createElement('div');
  msg.className = `message agent ${agent.class}`;

  let content = '';
  if (isTyping) {
    content = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  } else {
    content = renderMarkdown(textOrHtml);
  }

  msg.innerHTML = `
    <div class="message-avatar">${agent.icon}</div>
    <div class="message-content">
      <div class="message-header">
        <span class="message-name">${agent.name}</span>
        <span class="message-time">${nowTime()}</span>
      </div>
      <div class="message-bubble">${content}</div>
    </div>
  `;
  chatMessages.appendChild(msg);
  scrollToBottom();
  return msg;
}

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.innerHTML = `
    <div class="message-avatar">Vous</div>
    <div class="message-content">
      <div class="message-header">
        <span class="message-name" style="color:#CC0000">Vous</span>
        <span class="message-time">${nowTime()}</span>
      </div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  chatMessages.appendChild(msg);
  scrollToBottom();
}

function addErrorMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message agent agent-orchestrateur';
  msg.innerHTML = `
    <div class="message-avatar">⚠️</div>
    <div class="message-content">
      <div class="message-header">
        <span class="message-name" style="color:#EF4444">Erreur</span>
        <span class="message-time">${nowTime()}</span>
      </div>
      <div class="message-bubble" style="border-left-color:#EF4444">${escapeHtml(text)}</div>
    </div>
  `;
  chatMessages.appendChild(msg);
  scrollToBottom();
}

/* ============ ANIMATION BANNER ============ */
function showAnimation(agentId, stepText) {
  const agent = AGENTS[agentId];
  currentAgent = agentId;
  activeAgentIndicator.classList.add('active');
  activeAgentIndicator.querySelector('.label').textContent = `${agent.name} en action`;

  document.querySelectorAll('.agent-item').forEach((el) => el.classList.remove('active'));
  const row = document.getElementById(`agent-row-${agentId}`);
  if (row) row.classList.add('active');

  animationBanner.hidden = false;
  animLabel.textContent = `${agent.name} — ${agent.role}`;
  animLabel.style.color = agent.color;
  animText.textContent = stepText;
  animProgress.style.width = '10%';

  const tmplId = `tmpl-${agentId}`;
  const tmpl = document.getElementById(tmplId);
  if (tmpl) {
    animStage.innerHTML = '';
    const clone = tmpl.content.cloneNode(true);
    animStage.appendChild(clone);
  } else {
    animStage.innerHTML = `<div style="color:var(--text-muted);font-size:0.8rem">Chargement...</div>`;
  }
}

function updateProgress(pct, text) {
  animProgress.style.width = pct + '%';
  if (text) animText.textContent = text;
}

function hideAnimation() {
  animationBanner.hidden = true;
  animStage.innerHTML = '';
  currentAgent = null;
  activeAgentIndicator.classList.remove('active');
  activeAgentIndicator.querySelector('.label').textContent = 'En attente...';
  document.querySelectorAll('.agent-item').forEach((el) => el.classList.remove('active'));
}

/* ============ AGENT STATUS ============ */
function setAgentStatus(agentId, status) {
  const badge = document.getElementById(`status-${agentId}`);
  const bar = document.getElementById(`progress-${agentId}`);
  if (!badge) return;
  badge.className = 'agent-status';
  if (status === 'working') {
    badge.classList.add('status-working');
    badge.textContent = 'En cours';
    bar.style.width = '40%';
  } else if (status === 'done') {
    badge.classList.add('status-done');
    badge.textContent = 'Terminé';
    bar.style.width = '100%';
  } else {
    badge.classList.add('status-idle');
    badge.textContent = 'Inactif';
    bar.style.width = '0%';
  }
}

function resetAllStatuses() {
  ['scout', 'modelisateur', 'tacticien', 'validateur'].forEach((id) => setAgentStatus(id, 'idle'));
}

/* ============ DELAY ============ */
function wait(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

/* ============ API CALLS ============ */
async function apiAnalyse(adversaire, contexte = "") {
  const res = await fetch(`${API_BASE}/analyse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ adversaire, contexte }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function apiStats(club) {
  const res = await fetch(`${API_BASE}/stats/${encodeURIComponent(club)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function apiSquad(club) {
  const res = await fetch(`${API_BASE}/squad/${encodeURIComponent(club)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function apiFixtures(club) {
  const res = await fetch(`${API_BASE}/fixtures/${encodeURIComponent(club)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function apiCompare(question) {
  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

/* ============ EXTRACT ADVERSAIRE ============ */
function extractAdversaire(text) {
  const lower = text.toLowerCase();
  for (const adv of ADVERSAIRES_VALIDES) {
    const advLower = adv.toLowerCase();
    if (lower.includes(advLower)) return adv;
    // aliases
    if (adv === "Raja CA" && lower.includes("raja")) return adv;
    if (adv === "FAR Rabat" && lower.includes("far")) return adv;
    if (adv === "FUS Rabat" && lower.includes("fus")) return adv;
    if (adv === "DH El Jadida" && lower.includes("difaa")) return adv;
    if (adv === "CODM Meknes" && lower.includes("meknes")) return adv;
    if (adv === "CR Khemis Zemamra" && lower.includes("khemis")) return adv;
    if (adv === "Olympique Dcheira" && lower.includes("dcheira")) return adv;
    if (adv === "Yacoub El Mansour" && lower.includes("yacoub")) return adv;
  }
  return null;
}

/* ============ FORMATTERS ============ */
function formatStats(data) {
  if (data.erreur) return `**Erreur** : ${data.erreur}`;
  const rows = data.stats || [];
  if (!rows.length) return "Aucune statistique disponible pour ce club.";
  let lines = [`**Statistiques ${data.club} — Données FootyStats**`, ""];
  for (const r of rows.slice(0, 20)) {
    const stat = r["Stats"] || "";
    const overall = r["Overall"] || "";
    const home = r["At Home"] || "";
    const away = r["At Away"] || "";
    lines.push(`• **${stat}** : ${overall} (Domicile: ${home}, Extérieur: ${away})`);
  }
  return lines.join("\n");
}

function formatSquad(data) {
  const players = data.joueurs || [];
  if (!players.length) return `Aucun joueur trouvé pour **${data.club}** dans la base.`;
  let lines = [`**Effectif ${data.club} — Données FootyStats**`, ""];
  // Group by position prefix
  const byPos = {};
  for (const p of players) {
    let pos = p.position || "Autre";
    if (pos.toLowerCase().includes("goalkeeper")) pos = "Gardiens";
    else if (pos.toLowerCase().includes("defender")) pos = "Défenseurs";
    else if (pos.toLowerCase().includes("midfielder")) pos = "Milieux";
    else if (pos.toLowerCase().includes("forward")) pos = "Attaquants";
    else pos = "Autres";
    if (!byPos[pos]) byPos[pos] = [];
    byPos[pos].push(p);
  }
  for (const [pos, list] of Object.entries(byPos)) {
    lines.push(`**${pos}**`);
    for (const p of list) {
      lines.push(`• **${p.name}** — ${p.nationality} | ${p.matches} matchs, ${p.goals} buts, note ${p.rating}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

function formatFixtures(data) {
  const matches = data.matchs || [];
  if (!matches.length) return `Aucun match trouvé pour **${data.club}**.`;
  let lines = [`**Calendrier ${data.club} — Données FootyStats**`, ""];
  const recent = matches.slice(0, 10);
  for (const m of recent) {
    const status = m.status || "";
    const score = m.score || "? - ?";
    const scoreStr = score.trim() ? ` (${score})` : "";
    lines.push(`• ${m.date || "Date TBD"} : ${m.home} vs ${m.away}${scoreStr} ${status ? `[${status}]` : ""}`);
  }
  return lines.join("\n");
}

/* ============ PIPELINE ============ */
async function runPipeline(adversaire) {
  if (isRunning) return;
  isRunning = true;
  btnSend.disabled = true;
  chatInput.disabled = true;
  resetAllStatuses();

  addMessage('orchestrateur', `Je coordonne l'analyse complète pour le match **WAC vs ${adversaire}**. Lancement séquentiel des 4 agents...`);
  await wait(600);

  // Animation orchestrateur
  showAnimation('orchestrateur', 'Préparation du pipeline multi-agents...');
  updateProgress(5, 'Initialisation...');
  await wait(500);

  try {
    const response = await fetch(`${API_BASE}/analyse/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ adversaire, contexte: "" }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (!response.body) throw new Error("Pas de corps de réponse");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // dernier fragment incomplet

      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch (e) {
          console.warn("JSON parse error:", line);
          continue;
        }

        const ev = event.event;
        const agent = event.agent;
        const data = event.data;

        if (ev === "agent_start") {
          hideAnimation();
          const labels = {
            scout: 'Recherche dans la base FootyStats...',
            modelisateur: 'Analyse statistique en cours...',
            tacticien: 'Conception de la stratégie...',
            validateur: 'Vérification qualité du rapport...',
          };
          showAnimation(agent, labels[agent] || 'Traitement...');
          setAgentStatus(agent, 'working');
          // Marquer l'agent précédent comme terminé
          const order = ['scout', 'modelisateur', 'tacticien', 'validateur'];
          const idx = order.indexOf(agent);
          if (idx > 0) setAgentStatus(order[idx - 1], 'done');
          updateProgress(10 + idx * 22, labels[agent]);
        }

        if (ev === "agent_message" && data && data.output) {
          addMessage(agent, data.output);
          scrollToBottom();
        }

        if (ev === "agent_error" && data) {
          addErrorMessage(data.message || "Erreur inconnue");
        }

        if (ev === "agent_end") {
          setAgentStatus(agent, 'done');
        }

        if (ev === "pipeline_end") {
          hideAnimation();
          setAgentStatus('validateur', 'done');
          if (data && data.rapport_final) {
            addMessage('validateur', data.rapport_final);
          }
          await wait(400);
          addMessage('orchestrateur', `Pipeline terminé pour **WAC vs ${adversaire}**. Le rapport complet est disponible ci-dessus.`);
        }

        if (ev === "error") {
          hideAnimation();
          addErrorMessage(data?.message || "Erreur serveur");
        }

        if (ev === "done") {
          break;
        }
      }
    }

    reader.releaseLock();
  } catch (err) {
    hideAnimation();
    addErrorMessage(`Impossible de contacter le backend. Assurez-vous que l'API est lancée sur ${API_BASE}.\nDétails : ${err.message}`);
  }

  isRunning = false;
  btnSend.disabled = false;
  chatInput.disabled = false;
  chatInput.focus();
}

/* ============ COMMAND PARSER ============ */
async function handleUserInput(text) {
  text = text.trim();
  if (!text) return;
  addUserMessage(text);

  const lower = text.toLowerCase();

  // ANALYSE COMPLETE
  if (lower.includes('analyse') || lower.includes('vs') || lower.includes('match')) {
    const adv = extractAdversaire(text);
    if (!adv) {
      addMessage('orchestrateur', `Je n'ai pas reconnu l'adversaire. Voici les clubs disponibles :\n\n${ADVERSAIRES_VALIDES.map(a => `• ${a}`).join("\n")}`);
      return;
    }
    runPipeline(adv);
    return;
  }

  // STATS
  if (lower.includes('stats') || lower.includes('statistique') || lower.includes('bilan')) {
    const adv = extractAdversaire(text) || "Wydad AC";
    showAnimation('scout', `Chargement des stats ${adv}...`);
    setAgentStatus('scout', 'working');
    try {
      const data = await apiStats(adv);
      setAgentStatus('scout', 'done');
      hideAnimation();
      addMessage('scout', formatStats(data));
    } catch (err) {
      hideAnimation();
      addErrorMessage(`Erreur stats : ${err.message}`);
    }
    return;
  }

  // SQUAD / COMPOSITION
  if (lower.includes('compo') || lower.includes('composition') || lower.includes('formation') || lower.includes('effectif') || lower.includes('squad')) {
    const adv = extractAdversaire(text) || "Wydad AC";
    showAnimation('scout', `Chargement de l'effectif ${adv}...`);
    setAgentStatus('scout', 'working');
    try {
      const data = await apiSquad(adv);
      setAgentStatus('scout', 'done');
      hideAnimation();
      addMessage('scout', formatSquad(data));
    } catch (err) {
      hideAnimation();
      addErrorMessage(`Erreur effectif : ${err.message}`);
    }
    return;
  }

  // FIXTURES / CALENDRIER
  if (lower.includes('calendrier') || lower.includes('fixtures') || lower.includes('matchs')) {
    const adv = extractAdversaire(text) || "Wydad AC";
    showAnimation('scout', `Chargement du calendrier ${adv}...`);
    setAgentStatus('scout', 'working');
    try {
      const data = await apiFixtures(adv);
      setAgentStatus('scout', 'done');
      hideAnimation();
      addMessage('scout', formatFixtures(data));
    } catch (err) {
      hideAnimation();
      addErrorMessage(`Erreur calendrier : ${err.message}`);
    }
    return;
  }

  // RAG VS NO-RAG
  if (lower.includes('rag') || lower.includes('comparer') || lower.includes('vs no')) {
    showAnimation('modelisateur', 'Comparaison RAG vs No-RAG...');
    setAgentStatus('modelisateur', 'working');
    try {
      const question = "Quel est le bilan du WAC à domicile cette saison ?";
      const data = await apiCompare(question);
      setAgentStatus('modelisateur', 'done');
      hideAnimation();
      addMessage('modelisateur', `**Comparaison RAG vs No-RAG**

**Question** : ${data.question}

---

**Avec RAG** (données FootyStats) :
${data.rag}

---

**Sans RAG** (LLM seul) :
${data.no_rag}`);
    } catch (err) {
      hideAnimation();
      addErrorMessage(`Erreur comparaison : ${err.message}`);
    }
    return;
  }

  // DEFAULT HELP
  addMessage('orchestrateur', `Je peux vous aider avec :

• **Analyse complète** : *"Analyse WAC vs [adversaire]"*
• **Stats rapides** : *"Stats WAC"* ou *"Stats Raja CA"*
• **Effectif** : *"Composition WAC"* ou *"Squad FAR Rabat"*
• **Calendrier** : *"Calendrier WAC"*
• **Démo RAG** : *"RAG vs No-RAG"*

Adversaires disponibles : ${ADVERSAIRES_VALIDES.join(", ")}`);
}

/* ============ EVENTS ============ */
btnSend.addEventListener('click', () => {
  const val = chatInput.value.trim();
  if (val) { handleUserInput(val); chatInput.value = ''; }
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const val = chatInput.value.trim();
    if (val) { handleUserInput(val); chatInput.value = ''; }
  }
});

/* ============ WELCOME ============ */
function showWelcome() {
  addMessage('orchestrateur', `Bienvenue dans **WAC Sport Analytics**.

Système multi-agents RAG dédié au **Wydad Athletic Club**. Je coordonne 4 agents spécialisés pour vous fournir des analyses tactiques complètes basées sur les données FootyStats réelles scrappées.

💡 **Commencez par :** « Analyse WAC vs Raja CA »
📊 **Ou :** « Stats WAC » pour les données brutes`);
}

/* ============ BOOT ============ */
initSidebar();
showWelcome();
chatInput.focus();
