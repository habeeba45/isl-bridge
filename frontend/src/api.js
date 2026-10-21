
const API_BASE_URL = (
  import.meta.env.VITE_API_URL ||
  "https://isl-bridge-backend-yttq.onrender.com"
).replace(/\/+$/, "");

const apiUrl = (path) => `${API_BASE_URL}${path}`;

async function parse(res) {
  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const detail = data.detail || data.error || res.statusText;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
  }

  return data;
}

export const api = {
  health: () => fetch(apiUrl("/api/health")).then(parse),

  signs: () => fetch(apiUrl("/api/signs")).then(parse),

  events: () => fetch(apiUrl("/api/events")).then(parse),

  runs: () => fetch(apiUrl("/api/runs")).then(parse),

  latestConsent: () =>
    fetch(apiUrl("/api/consent/latest")).then(parse),

  consent: (body) =>
    fetch(apiUrl("/api/consent"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(parse),

  recognize: (frames, consentId) =>
    fetch(apiUrl("/api/recognize"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Consent-Id": String(consentId),
      },
      body: JSON.stringify({
        frames,
        consent_id: consentId,
      }),
    }).then(parse),

  speakSign: (signId, consentId) =>
    fetch(apiUrl("/api/speak-sign"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Consent-Id": String(consentId),
      },
      body: JSON.stringify({
        sign_id: signId,
        consent_id: consentId,
        approved: true,
      }),
    }).then(parse),

  stt: async (blob, consentId, language = "en") => {
    const form = new FormData();

    form.append("audio", blob, "speech.webm");
    form.append("consent_id", String(consentId));
    form.append("language", language);
    form.append("approved", "true");

    return parse(
      await fetch(apiUrl("/api/stt"), {
        method: "POST",
        body: form,
      })
    );
  },

  reverse: (transcript, corrected) =>
    fetch(apiUrl("/api/reverse"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcript,
        corrected_transcript: corrected,
      }),
    }).then(parse),

  saveRun: (body) =>
    fetch(apiUrl("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(parse),
};

export function playBase64Audio(b64, mime = "audio/mpeg") {
  const audio = new Audio(`data:${mime};base64,${b64}`);
  return audio.play();
}

export function browserSpeak(text, lang) {
  return new Promise((resolve, reject) => {
    if (!window.speechSynthesis) {
      reject(new Error("speechSynthesis unavailable"));
      return;
    }

    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = lang.startsWith("hi") ? "hi-IN" : "en-IN";
    utter.onend = resolve;
    utter.onerror = reject;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  });
}

