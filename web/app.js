const API_URL = "http://127.0.0.1:8000/chat";
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const conversation = document.querySelector("#conversation");
const messages = document.querySelector("#messages");
const emptyState = document.querySelector("#empty-state");
const status = document.querySelector("#status");
const errorNotice = document.querySelector("#error");
let isGenerating = false;

function updateSendButton() {
  sendButton.disabled = isGenerating || !input.value.trim();
}

function addMessage(role, text) {
  emptyState.hidden = true;
  const message = document.createElement("article");
  message.className = `message ${role}`;
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Qwen";
  const content = document.createElement("p");
  content.className = "message-text";
  // Treat user input and model output as text, never as HTML.
  content.textContent = text;
  message.append(label, content);
  messages.append(message);
  conversation.scrollTop = conversation.scrollHeight;
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
  addMessage("user", message);
  input.value = "";
  input.disabled = true;
  updateSendButton();
  status.textContent = "Qwen is thinking...";

  try {
    const result = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
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
  } catch (error) {
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
