const API_URL = "http://127.0.0.1:8000/chat";
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const messages = document.querySelector("#messages");
const emptyState = document.querySelector("#empty-state");
const status = document.querySelector("#status");
const errorNotice = document.querySelector("#error");
// This page owns its history. A refresh starts with an empty conversation.
const conversationHistory = [];
let isGenerating = false;

function updateSendButton() {
  sendButton.disabled = isGenerating || !input.value.trim();
}

function renderMarkdownSafely(container, text) {
  // A controlled subset: no HTML parsing, links, images, or arbitrary attributes.
  function appendInline(parent, value) {
    const tokens = /`([^`\n]+)`|\*\*([^*\n]+)\*\*/g;
    let position = 0;
    for (const match of value.matchAll(tokens)) {
      parent.append(document.createTextNode(value.slice(position, match.index)));
      const element = document.createElement(match[1] !== undefined ? "code" : "strong");
      element.textContent = match[1] ?? match[2];
      parent.append(element);
      position = match.index + match[0].length;
    }
    parent.append(document.createTextNode(value.slice(position)));
  }

  container.replaceChildren();
  let paragraph = null;
  let list = null;
  for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
    if (!line.trim()) {
      paragraph = null;
      list = null;
      continue;
    }

    const heading = line.match(/^(#{1,3})[ \t]+(.+)$/);
    if (heading) {
      paragraph = null;
      list = null;
      const element = document.createElement(`h${heading[1].length}`);
      appendInline(element, heading[2]);
      container.append(element);
      continue;
    }

    const bullet = line.match(/^[ \t]*-[ \t]+(.+)$/);
    const numbered = line.match(/^[ \t]*(\d{1,9})\.[ \t]+(.+)$/);
    if (bullet || numbered) {
      paragraph = null;
      const tag = bullet ? "ul" : "ol";
      if (!list || list.tagName.toLowerCase() !== tag) {
        list = document.createElement(tag);
        // The start value can only come from at most nine decimal digits.
        if (numbered) list.start = Number(numbered[1]);
        container.append(list);
      }
      const item = document.createElement("li");
      appendInline(item, bullet ? bullet[1] : numbered[2]);
      list.append(item);
      continue;
    }

    list = null;
    if (!paragraph) {
      paragraph = document.createElement("p");
      container.append(paragraph);
    } else {
      paragraph.append(document.createElement("br"));
    }
    appendInline(paragraph, line);
  }
}

function addMessage(role, text) {
  emptyState.hidden = true;
  const message = document.createElement("article");
  message.className = `message ${role}`;
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Qwen";
  const content = document.createElement(role === "user" ? "p" : "div");
  content.className = "message-text";
  if (role === "user") {
    content.textContent = text;
  } else {
    renderMarkdownSafely(content, text);
  }
  message.append(label, content);
  messages.append(message);
  conversation.scrollTop = conversation.scrollHeight;
  return message;
}

input.addEventListener("input", updateSendButton);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (isGenerating || !message) return;

  isGenerating = true;
  errorNotice.hidden = true;
  errorNotice.textContent = "";
  conversationHistory.push({ role: "user", content: message });
  const userMessage = addMessage("user", message);
  input.value = "";
  input.disabled = true;
  updateSendButton();
  status.textContent = "Qwen is thinking...";

  try {
    const result = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: conversationHistory }),
    });

    if (!result.ok) {
      throw new Error(`The server returned an error (${result.status}). Please try again.`);
    }

    let data;
    try {
      data = await result.json();
    } catch {
      throw new Error("The server returned an invalid response. Please try again.");
    }
    if (typeof data?.response !== "string" || !data.response.trim()) {
      throw new Error("The server returned an empty or invalid answer. Please try again.");
    }
    addMessage("qwen", data.response);
    conversationHistory.push({ role: "assistant", content: data.response });
  } catch (error) {
    // Roll back the unanswered turn so retrying sends it exactly once.
    conversationHistory.pop();
    userMessage.remove();
    emptyState.hidden = conversationHistory.length > 0;
    errorNotice.textContent = error instanceof TypeError
      ? "Could not reach Qwen. Check that the FastAPI server is running, then try again."
      : error.message || "Something went wrong. Please try again.";
    errorNotice.hidden = false;
    input.value = message;
  } finally {
    isGenerating = false;
    input.disabled = false;
    status.textContent = "";
    updateSendButton();
    input.focus();
  }
});
