(() => {
  "use strict";

  const STORE_KEY = "medibot.conversations.v2";
  const THEME_KEY = "medibot.theme";
  const MAX_CONVERSATIONS = 30;

  const $ = (id) => document.getElementById(id);
  const els = {
    sidebar: $("sidebar"), scrim: $("scrim"),
    openSidebar: $("openSidebar"), closeSidebar: $("closeSidebar"),
    newChat: $("newChat"), history: $("history"),
    title: $("chatTitle"), status: $("status"), themeToggle: $("themeToggle"),
    thread: $("thread"), welcome: $("welcome"), messages: $("messages"),
    form: $("composer"), input: $("input"), send: $("send"), counter: $("counter"),
    toast: $("toast"),
  };

  const BOT_ICON = '<svg viewBox="0 0 24 24"><path d="M9.5 3h5v6.5H21v5h-6.5V21h-5v-6.5H3v-5h6.5z"/></svg>';
  const ICONS = {
    copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>',
    check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    retry: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4.5h4.5"/></svg>',
    trash: '<svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.01"/></svg>',
  };

  // ---------- Storage (wrapped: may be unavailable) ----------
  const storage = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
      catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
    },
  };

  // ---------- State ----------
  let conversations = storage.get(STORE_KEY, []);
  let current = null;           // active conversation object
  let controller = null;        // AbortController for an in-flight request

  const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  const save = () => {
    conversations.sort((a, b) => b.updatedAt - a.updatedAt);
    conversations = conversations.slice(0, MAX_CONVERSATIONS);
    storage.set(STORE_KEY, conversations);
  };

  // ---------- Helpers ----------
  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function renderMarkdown(text) {
    if (window.marked && window.DOMPurify) {
      return DOMPurify.sanitize(marked.parse(text, { breaks: true, gfm: true }));
    }
    return `<p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>`;
  }

  let toastTimer;
  function toast(message) {
    els.toast.textContent = message;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (els.toast.hidden = true), 2200);
  }

  const nearBottom = () =>
    els.thread.scrollHeight - els.thread.scrollTop - els.thread.clientHeight < 120;
  const scrollToBottom = (force = false) => {
    if (force || nearBottom()) els.thread.scrollTop = els.thread.scrollHeight;
  };

  // ---------- Theme ----------
  function isDark() {
    const t = document.documentElement.dataset.theme;
    if (t) return t === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  els.themeToggle.addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
  });

  // ---------- Sidebar ----------
  function setSidebar(open) {
    els.sidebar.classList.toggle("open", open);
    els.scrim.hidden = !open;
  }
  els.openSidebar.addEventListener("click", () => setSidebar(true));
  els.closeSidebar.addEventListener("click", () => setSidebar(false));
  els.scrim.addEventListener("click", () => setSidebar(false));

  function renderHistory() {
    els.history.innerHTML = "";
    if (!conversations.length) {
      els.history.innerHTML = '<div class="history-empty">Your conversations will appear here.</div>';
      return;
    }
    for (const conv of conversations) {
      const item = document.createElement("div");
      item.className = "history-item" + (current && conv.id === current.id ? " active" : "");

      const open = document.createElement("button");
      open.className = "history-open";
      open.textContent = conv.title;
      open.title = conv.title;
      open.addEventListener("click", () => { loadConversation(conv.id); setSidebar(false); });

      const del = document.createElement("button");
      del.className = "history-del";
      del.setAttribute("aria-label", `Delete "${conv.title}"`);
      del.innerHTML = ICONS.trash;
      del.addEventListener("click", () => deleteConversation(conv.id));

      item.append(open, del);
      els.history.append(item);
    }
  }

  function deleteConversation(id) {
    conversations = conversations.filter((c) => c.id !== id);
    save();
    if (current && current.id === id) newConversation();
    else renderHistory();
    toast("Conversation deleted");
  }

  // ---------- Conversation lifecycle ----------
  function newConversation() {
    abortRequest();
    current = null;
    els.messages.innerHTML = "";
    els.welcome.hidden = false;
    els.title.textContent = "New conversation";
    renderHistory();
    els.input.focus();
  }

  function loadConversation(id) {
    abortRequest();
    const conv = conversations.find((c) => c.id === id);
    if (!conv) return;
    current = conv;
    els.messages.innerHTML = "";
    els.welcome.hidden = true;
    els.title.textContent = conv.title;
    for (const m of conv.messages) {
      if (m.role === "user") addUserMessage(m.content);
      else finishBotMessage(addBotMessage(), m.content, m.sources || []);
    }
    renderHistory();
    scrollToBottom(true);
  }

  function ensureConversation(firstMessage) {
    if (current) return current;
    const title = firstMessage.length > 48 ? firstMessage.slice(0, 48).trimEnd() + "…" : firstMessage;
    current = { id: uid(), title, messages: [], updatedAt: Date.now() };
    conversations.unshift(current);
    els.title.textContent = title;
    return current;
  }

  // ---------- Message rendering ----------
  function addUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.append(bubble);
    els.messages.append(row);
    return row;
  }

  function addBotMessage() {
    const row = document.createElement("div");
    row.className = "msg bot";
    row.innerHTML = `
      <div class="avatar" aria-hidden="true">${BOT_ICON}</div>
      <div class="body">
        <div class="content"></div>
        <div class="meta" hidden></div>
      </div>`;
    els.messages.append(row);
    return row;
  }

  function showThinking(row) {
    row.querySelector(".content").innerHTML =
      '<div class="thinking"><span class="dots"><span></span><span></span><span></span></span>Searching the medical reference…</div>';
  }

  function renderMeta(row, text, sources, { retry } = {}) {
    const meta = row.querySelector(".meta");
    meta.innerHTML = "";
    if (sources && sources.length) {
      const label = document.createElement("span");
      label.className = "sources-label";
      label.textContent = "Sources";
      meta.append(label);
      for (const s of sources) {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        const pageLabel = s.page ? `p. ${s.page}` : "";
        chip.textContent = s.topic
          ? (pageLabel ? `${s.topic} · ${pageLabel}` : s.topic)
          : (pageLabel || "Reference");
        const tip = document.createElement("span");
        tip.className = "tip";
        tip.textContent = s.section ? `${s.section}: ${s.snippet}` : s.snippet;
        chip.append(tip);
        meta.append(chip);
      }
    }
    const actions = document.createElement("div");
    actions.className = "actions";
    if (text) {
      const copy = document.createElement("button");
      copy.className = "act";
      copy.innerHTML = `${ICONS.copy}<span>Copy</span>`;
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(text);
          copy.innerHTML = `${ICONS.check}<span>Copied</span>`;
          setTimeout(() => (copy.innerHTML = `${ICONS.copy}<span>Copy</span>`), 1600);
        } catch { toast("Couldn't copy to clipboard"); }
      });
      actions.append(copy);
    }
    if (retry) {
      const again = document.createElement("button");
      again.className = "act";
      again.innerHTML = `${ICONS.retry}<span>Retry</span>`;
      again.addEventListener("click", retry);
      actions.append(again);
    }
    meta.append(actions);
    meta.hidden = false;
  }

  function finishBotMessage(row, text, sources) {
    row.classList.remove("streaming");
    row.querySelector(".content").innerHTML = renderMarkdown(text);
    renderMeta(row, text, sources);
  }

  function showError(row, message, retry) {
    row.classList.remove("streaming");
    const content = row.querySelector(".content");
    const box = document.createElement("div");
    box.className = "error-box";
    box.innerHTML = ICONS.alert;
    const span = document.createElement("span");
    span.textContent = message;
    box.append(span);
    content.append(box);
    renderMeta(row, "", [], { retry });
  }

  // ---------- Networking ----------
  function abortRequest() {
    if (controller) { controller.abort(); controller = null; }
    setBusy(false);
  }

  function setBusy(busy) {
    els.form.classList.toggle("busy", busy);
    els.send.setAttribute("aria-label", busy ? "Stop generating" : "Send message");
    updateSendState();
  }

  function updateSendState() {
    const busy = els.form.classList.contains("busy");
    els.send.disabled = !busy && !els.input.value.trim();
    els.counter.textContent = `${els.input.value.length} / 2000`;
  }

  async function* readEvents(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("");
        if (data) yield JSON.parse(data);
      }
    }
  }

  async function ask(question) {
    const conv = ensureConversation(question);
    const history = conv.messages.map(({ role, content }) => ({ role, content }));

    els.welcome.hidden = true;
    addUserMessage(question);
    conv.messages.push({ role: "user", content: question });
    conv.updatedAt = Date.now();
    save();
    renderHistory();

    const row = addBotMessage();
    showThinking(row);
    scrollToBottom(true);

    controller = new AbortController();
    setBusy(true);

    let text = "";
    let sources = [];
    let frame = 0;
    const content = row.querySelector(".content");
    const paint = () => {
      frame = 0;
      content.innerHTML = renderMarkdown(text);
      scrollToBottom();
    };

    const retry = () => {
      // Remove the failed exchange, then ask again.
      row.remove();
      const userRow = els.messages.lastElementChild;
      if (userRow && userRow.classList.contains("user")) userRow.remove();
      const last = conv.messages[conv.messages.length - 1];
      if (last && last.role === "user") conv.messages.pop();
      ask(question);
    };

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
        signal: controller.signal,
      });

      if (!res.ok) {
        let detail = `Request failed (${res.status}).`;
        try {
          const body = await res.json();
          if (typeof body.detail === "string") detail = body.detail;
          else if (Array.isArray(body.detail)) detail = "Your message couldn't be processed. Please shorten it and try again.";
        } catch { /* keep default */ }
        throw new Error(detail);
      }

      for await (const event of readEvents(res)) {
        if (event.type === "sources") {
          sources = event.sources || [];
        } else if (event.type === "token") {
          if (!text) row.classList.add("streaming");
          text += event.content;
          if (!frame) frame = requestAnimationFrame(paint);
        } else if (event.type === "error") {
          throw new Error(event.message);
        }
      }

      if (frame) cancelAnimationFrame(frame);
      if (!text) throw new Error("The assistant returned an empty answer. Please try again.");
      finishBotMessage(row, text, sources);
      conv.messages.push({ role: "assistant", content: text, sources });
      conv.updatedAt = Date.now();
      save();
    } catch (err) {
      if (frame) cancelAnimationFrame(frame);
      if (err.name === "AbortError") {
        if (text) {
          finishBotMessage(row, text + "\n\n*— stopped —*", sources);
          conv.messages.push({ role: "assistant", content: text, sources });
          save();
        } else {
          row.remove();
        }
      } else {
        if (text) content.innerHTML = renderMarkdown(text);
        else content.innerHTML = "";
        const msg = err instanceof TypeError
          ? "Couldn't reach the server. Is it running?"
          : err.message;
        showError(row, msg, retry);
      }
    } finally {
      controller = null;
      setBusy(false);
      scrollToBottom();
    }
  }

  // ---------- Composer ----------
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + "px";
  }

  function submit() {
    if (controller) { abortRequest(); return; }
    const q = els.input.value.trim();
    if (!q) return;
    els.input.value = "";
    autosize();
    updateSendState();
    ask(q);
  }

  els.form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  els.input.addEventListener("input", () => { autosize(); updateSendState(); });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });
  els.newChat.addEventListener("click", () => { newConversation(); setSidebar(false); });
  document.querySelectorAll(".suggestion").forEach((b) =>
    b.addEventListener("click", () => { if (!controller) ask(b.dataset.q); }));

  // ---------- Health ----------
  async function checkHealth() {
    const set = (cls, text, title) => {
      els.status.className = "status " + cls;
      els.status.querySelector(".status-text").textContent = text;
      els.status.title = title;
    };
    try {
      const res = await fetch("/api/health");
      const h = await res.json();
      if (h.ready) set("ok", "Online", `Ready · ${h.model}`);
      else set("err", "Offline", h.error || "Service not ready");
    } catch {
      set("err", "Offline", "Can't reach the server");
    }
  }

  // ---------- Init ----------
  renderHistory();
  updateSendState();
  checkHealth();
  setInterval(checkHealth, 30000);
  els.input.focus();
})();
