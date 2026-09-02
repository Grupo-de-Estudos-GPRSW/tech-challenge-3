/* Chat frontend for the GPRSW medical assistant.
   Talks to /api/chat, which streams the LangGraph execution back as SSE. */

const chatEl = document.getElementById("chat");
const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const newChatEl = document.getElementById("new-chat");

let sessionId = null;
let busy = false;
let pipelineReady = false;

/* ------------------------------------------------------------- helpers -- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideWelcome() {
  welcomeEl.classList.add("hidden");
}

function botAvatar() {
  const img = el("img", "avatar");
  img.src = "/static/logo.svg";
  img.alt = "Assistant";
  return img;
}

/* ------------------------------------------------------------ messages -- */

function addUserMessage(text) {
  const row = el("div", "msg msg-user");
  row.appendChild(el("div", "bubble", text));
  messagesEl.appendChild(row);
  scrollToBottom();
}

/** Creates the assistant row and returns handles used while the turn streams. */
function addBotMessage() {
  const row = el("div", "msg msg-bot");
  const stack = el("div", "stack");
  const steps = el("div", "steps");
  stack.appendChild(steps);
  row.append(botAvatar(), stack);
  messagesEl.appendChild(row);
  scrollToBottom();

  let pending = null; // step currently running

  function settlePending() {
    if (!pending) return;
    pending.classList.remove("active");
    pending.replaceChild(checkIcon(), pending.firstChild);
    pending = null;
  }

  function checkIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "tick");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "3");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", "M20 6 9 17l-5-5");
    svg.appendChild(path);
    return svg;
  }

  return {
    /** Marks the previous step done and starts a new one. */
    step(label) {
      settlePending();
      const step = el("div", "step active");
      step.append(el("div", "spinner"), el("span", null, label));
      steps.appendChild(step);
      pending = step;
      scrollToBottom();
    },
    finishSteps() {
      settlePending();
    },
    /** Collapses the step list once the turn is over. */
    collapseSteps(summaryText) {
      settlePending();
      if (!steps.childElementCount) {
        steps.remove();
        return;
      }
      const details = el("details", "context");
      const summary = el("summary", null, summaryText);
      const body = el("div", "context-body");
      body.appendChild(steps.cloneNode(true));
      body.firstChild.style.background = "transparent";
      body.firstChild.style.border = "0";
      body.firstChild.style.padding = "0";
      details.append(summary, body);
      steps.replaceWith(details);
    },
    append(node) {
      stack.appendChild(node);
      scrollToBottom();
    },
  };
}

function answerBubble(text) {
  return el("div", "bubble", text || "(empty response)");
}

function notice(kind, title, text) {
  const box = el("div", `notice notice-${kind}`);
  const content = el("div");
  content.appendChild(el("strong", null, title));
  content.appendChild(el("span", null, text));
  box.appendChild(content);
  return box;
}

/** Collapsible panel with the EHR context and the retrieved protocols. */
function contextPanel(result) {
  const hasContext = result.patient_context && Object.keys(result.patient_context).length > 0;
  const docs = result.retrieved_docs || [];
  if (!hasContext && !docs.length && !result.original_output) return null;

  const parts = [];
  if (result.patient_id) parts.push(result.patient_id);
  if (docs.length) parts.push(`${docs.length} protocol${docs.length > 1 ? "s" : ""}`);

  const details = el("details", "context");
  details.appendChild(el("summary", null, `Sources${parts.length ? " · " + parts.join(" · ") : ""}`));
  const body = el("div", "context-body");

  if (hasContext) {
    body.appendChild(el("h4", null, `Patient record${result.patient_id ? " · " + result.patient_id : ""}`));
    for (const [key, value] of Object.entries(result.patient_context)) {
      const label = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      const line = el("div", "field");
      line.appendChild(el("strong", null, label));
      const values = Array.isArray(value) ? value : [value];
      if (!values.length) {
        line.appendChild(el("span", "empty", "none"));
      } else {
        const tags = el("div", "tags");
        values.forEach((v) => tags.appendChild(el("span", "tag", String(v))));
        line.appendChild(tags);
      }
      body.appendChild(line);
    }
  }

  if (docs.length) {
    body.appendChild(el("h4", null, "Retrieved clinical protocols"));
    docs.forEach((doc) => body.appendChild(el("div", "doc", doc)));
  }

  if (result.original_output) {
    body.appendChild(el("h4", null, "Draft held for physician review"));
    body.appendChild(el("div", "doc", result.original_output));
  }

  if (result.log_filepath) {
    body.appendChild(el("h4", null, "Audit trail"));
    body.appendChild(el("div", "path", result.log_filepath));
  }

  details.appendChild(body);
  return details;
}

/* ---------------------------------------------------------------- state -- */

function setStatus(state, detail) {
  pipelineReady = state === "ready";
  statusPill.className = `pill pill-${state === "ready" ? "ready" : state === "error" ? "error" : "loading"}`;
  statusText.textContent =
    state === "ready" ? "Pipeline ready" : state === "error" ? "Pipeline error" : detail || "Loading…";
  statusPill.title = detail || "";
  inputEl.placeholder = pipelineReady
    ? "Ask about a patient… (e.g. P001)"
    : state === "error"
      ? "The pipeline could not be loaded — see the status above."
      : "Loading the pipeline… you can already type your question.";
  updateSendState();
}

function updateSendState() {
  sendEl.disabled = busy || !inputEl.value.trim();
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    setStatus(data.state, data.detail);
    if (data.state === "ready" || data.state === "error") return;
  } catch (err) {
    setStatus("error", String(err));
  }
  setTimeout(pollStatus, 2000);
}

/* ------------------------------------------------------------ streaming -- */

async function send(text) {
  if (busy) return;
  busy = true;
  updateSendState();
  hideWelcome();
  addUserMessage(text);

  const bot = addBotMessage();
  bot.step("Contacting the pipeline");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const line = raw.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        handleEvent(JSON.parse(line.slice(6)), bot);
      }
    }
  } catch (err) {
    bot.finishSteps();
    bot.append(notice("error", "Connection failed", String(err)));
  } finally {
    busy = false;
    updateSendState();
    inputEl.focus();
  }
}

function handleEvent(event, bot) {
  switch (event.type) {
    case "session":
      sessionId = event.session_id;
      break;

    case "node":
      bot.step(event.label);
      break;

    case "result": {
      bot.collapseSteps("Pipeline trace");
      if (event.missing_patient_id) {
        bot.append(notice("warn", "Patient identifier missing", event.answer));
      } else {
        bot.append(answerBubble(event.answer));
      }
      if (event.requires_human_approval) {
        bot.append(
          notice(
            "warn",
            "Guardrail triggered · physician sign-off required",
            "The draft contains sensitive clinical content and was routed to the human validation gate. The original draft is available under Sources.",
          ),
        );
      }
      const panel = contextPanel(event);
      if (panel) bot.append(panel);
      break;
    }

    case "error":
      bot.finishSteps();
      bot.append(notice("error", "Pipeline error", event.message));
      break;

    default:
      break;
  }
}

/* --------------------------------------------------------------- events -- */

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text || busy) return;
  inputEl.value = "";
  autosize();
  send(text);
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    formEl.requestSubmit();
  }
});

function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 168) + "px";
}

inputEl.addEventListener("input", () => {
  autosize();
  updateSendState();
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (busy) return;
    send(btn.textContent.trim());
  });
});

newChatEl.addEventListener("click", async () => {
  if (busy) return;
  try {
    const res = await fetch("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId || "" }),
    });
    const data = await res.json();
    sessionId = data.session_id;
  } catch (err) {
    sessionId = null;
  }
  messagesEl.replaceChildren();
  welcomeEl.classList.remove("hidden");
  inputEl.value = "";
  autosize();
  updateSendState();
  inputEl.focus();
});

pollStatus();
inputEl.focus();
