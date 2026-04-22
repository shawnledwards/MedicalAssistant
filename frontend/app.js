'use strict';

const PERSONA_LABELS = {
  scientist: 'Research Scientist',
  doctor: 'Physician',
  faq: 'Patient FAQ',
};

let activePersona = 'scientist';

// ── Persona selection ─────────────────────────────────────────────
document.querySelectorAll('.persona-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.persona-card').forEach(c => {
      c.classList.remove('active');
      c.setAttribute('aria-pressed', 'false');
    });
    card.classList.add('active');
    card.setAttribute('aria-pressed', 'true');
    activePersona = card.dataset.persona;
    document.getElementById('active-persona-label').textContent =
      PERSONA_LABELS[activePersona];
  });
});

// ── Form submit ───────────────────────────────────────────────────
const form = document.getElementById('chat-form');
const input = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const thread = document.getElementById('chat-thread');

form.addEventListener('submit', e => {
  e.preventDefault();
  submitQuestion();
});

// Ctrl+Enter shortcut
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submitQuestion();
  }
});

async function submitQuestion() {
  const question = input.value.trim();
  if (!question) return;

  appendMessage('user', question, activePersona);
  input.value = '';
  sendBtn.disabled = true;

  const typingId = appendTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, persona: activePersona }),
    });

    removeTyping(typingId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendError(err.detail || 'Unexpected server error.');
    } else {
      const data = await res.json();
      appendMessage('assistant', data.answer, data.persona, data.sources);
    }
  } catch (err) {
    removeTyping(typingId);
    appendError('Could not reach the server. Is the API running?');
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

// ── Render helpers ────────────────────────────────────────────────
function appendMessage(role, text, persona, sources = []) {
  const msg = document.createElement('div');
  msg.className = `msg ${role}${role === 'assistant' ? ` persona-${persona}` : ''}`;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : PERSONA_LABELS[persona] ?? persona;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  // FAQ persona: render bullet lines as an actual <ul>
  if (role === 'assistant' && persona === 'faq') {
    bubble.innerHTML = formatFAQ(text);
  } else {
    bubble.textContent = text;
  }

  msg.appendChild(label);
  msg.appendChild(bubble);

  if (sources.length) {
    const src = document.createElement('div');
    src.className = 'sources';
    src.textContent = `Source: ${sources.join(', ')}`;
    msg.appendChild(src);
  }

  thread.appendChild(msg);
  scrollToBottom();
}

function formatFAQ(text) {
  // Convert lines starting with "- " or "• " or "* " into <ul><li>
  const lines = text.split('\n').filter(l => l.trim());
  const isBullet = l => /^[\-\•\*]\s/.test(l.trim());

  if (!lines.some(isBullet)) return escapeHtml(text);

  let html = '';
  let inList = false;
  for (const line of lines) {
    if (isBullet(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${escapeHtml(line.replace(/^[\-\•\*]\s/, ''))}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<p>${escapeHtml(line)}</p>`;
    }
  }
  if (inList) html += '</ul>';
  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function appendTyping() {
  const id = `typing-${Date.now()}`;
  const msg = document.createElement('div');
  msg.className = `msg assistant persona-${activePersona}`;
  msg.id = id;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = PERSONA_LABELS[activePersona];

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

  msg.appendChild(label);
  msg.appendChild(bubble);
  thread.appendChild(msg);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  document.getElementById(id)?.remove();
}

function appendError(message) {
  const div = document.createElement('div');
  div.className = 'error-bubble';
  div.textContent = `Error: ${message}`;
  thread.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  thread.scrollTop = thread.scrollHeight;
}
