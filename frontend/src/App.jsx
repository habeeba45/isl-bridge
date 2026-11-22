import { useEffect, useMemo, useRef, useState } from "react";
import { api, browserSpeak, playBase64Audio } from "./api";
import CameraStage from "./components/CameraStage.jsx";
import SignGlyph from "./components/SignGlyph.jsx";
import DatasetRecorder from "./components/DatasetRecorder.jsx";

const TABS = [
  { id: "signs", label: "TC1 Supported signs" },
  { id: "sign2speech", label: "TC2 Sign → speech" },
  { id: "speech2sign", label: "TC3 Speech → sign" },
  { id: "safety", label: "TC4 Safety & evidence" },
  { id: "dataset", label: "Dataset recorder" },
];

export default function App() {
  const [tab, setTab] = useState("signs");
  const [health, setHealth] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [consent, setConsent] = useState(null);
  const [form, setForm] = useState({
    subject_name: "",
    operator_name: "",
    purpose: "Hackathon demo of ISL Bridge. I consent to being filmed and recorded for this local demonstration.",
  });
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const cameraRef = useRef(null);
  const [recognition, setRecognition] = useState(null);
  const [speechOut, setSpeechOut] = useState(null);
  const [sttOut, setSttOut] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [runs, setRuns] = useState([]);
  const [events, setEvents] = useState([]);
  const [listening, setListening] = useState(false);
  const recRef = useRef(null);

  const consented = Boolean(consent?.granted);

  async function refresh() {
    const [h, s, c, r, e] = await Promise.all([
      api.health(),
      api.signs(),
      api.latestConsent(),
      api.runs(),
      api.events(),
    ]);
    setHealth(h);
    setCatalog(s);
    setConsent(c.consent || null);
    setRuns(r.runs);
    setEvents(e.events);
  }

  useEffect(() => {
    refresh().catch((err) => setError(err.message));
  }, []);


async function grantConsent(granted) {
  setError("");

  try {
    const record = await api.consent({ ...form, granted });

    if (!granted) {
      // Stop browser speech recognition if it is active.
      try {
        recRef.current?.stop();
      } catch {
        // It may already be stopped.
      }

      recRef.current = null;
      setListening(false);

      // Clear results that should not remain active after revocation.
      setRecognition(null);
      setSpeechOut(null);
      setSttOut(null);
      setTranscript("");
    }

    setConsent(record);
    await refresh();
  } catch (err) {
    setError(err.message);
  }
}





async function classify() {
  setError("");
  setBusy("recognize");
  setRecognition(null);
  setSpeechOut(null);

  try {
    // Drop leftover poses, then capture only this attempt.
    cameraRef.current?.clear?.();
    await new Promise((resolve) => setTimeout(resolve, 2500));

    const frames = cameraRef.current?.frames() || [];

    if (frames.length < 15) {
      throw new Error(
        "Not enough camera frames. Keep the camera enabled and try again."
      );
    }

    const result = await api.recognize(frames, consent.id);
    setRecognition(result);
  } catch (err) {
    setError(err.message || "Recognition failed. Please try again.");
  } finally {
    setBusy("");
  }
}




  async function speakRecognized() {
    if (recognition?.state !== "recognized") return;
    setError("");
    setBusy("speak");
    try {
      const payload = await api.speakSign(recognition.sign_id, consent.id);
      setSpeechOut(payload);
      await playPair(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function playPair(payload) {
    const en = payload.english.tts;
    const hi = payload.hindi.tts;
    if (en.audio_base64) await playBase64Audio(en.audio_base64, en.mime);
    else await browserSpeak(payload.english.text, "en");
    if (hi.audio_base64) await playBase64Audio(hi.audio_base64, hi.mime);
    else await browserSpeak(payload.hindi.text, "hi");
  }


function startBrowserStt() {
  const Speech =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

  if (!Speech) {
    setError(
      "Speech recognition is not supported. Please use Chrome."
    );
    return;
  }

  const rec = new Speech();
  rec.lang = "en-IN";
  rec.interimResults = true;
  rec.continuous = false;

  rec.onstart = () => {
    setError("");
    setListening(true);
  };

  rec.onresult = (ev) => {
    const text = Array.from(ev.results)
      .map((result) => result[0].transcript)
      .join(" ");

    setTranscript(text);
  };

  rec.onerror = (ev) => {
    setListening(false);
    setError(
      "Speech recognition error: " +
      ev.error +
      ". Check microphone permissions and internet connection."
    );
  };

  rec.onend = () => {
    setListening(false);
  };

  recRef.current = rec;

  try {
    setError("");
    setListening(true);
    rec.start();
  } catch (err) {
    setListening(false);
    setError("Could not start speech recognition: " + err.message);
  }
}


  async function matchTranscript(text) {
    setBusy("reverse");
    setError("");
    try {
      const result = await api.reverse(text, text);
      setSttOut({
        stt: { provider: "browser_or_corrected", text },
        reverse_route: result.reverse_route,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function recordAndStt() {
    if (!consented) return;
    setError("");
    setBusy("record");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      const stopped = new Promise((resolve) => {
        recorder.onstop = resolve;
      });
      recorder.start();
      await new Promise((r) => setTimeout(r, 3500));
      recorder.stop();
      await stopped;
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      const result = await api.stt(blob, consent.id, "en");
      setSttOut(result);
      setTranscript(result.stt?.text || "");
      if (!result.stt?.ok && !result.stt?.text) {
        startBrowserStt();
      }
    } catch (err) {
      setError(err.message);
      startBrowserStt();
    } finally {
      setBusy("");
    }
  }

  async function markRun(caseId, actual, status, evidence) {
    try {
      await api.saveRun({ case_id: caseId, actual, status, evidence });
      const r = await api.runs();
      setRuns(r.runs);
    } catch (err) {
      setError(err.message);
    }
  }

  const recChip = useMemo(() => {
    if (!recognition) return null;
    if (recognition.state === "UNRECOGNIZED") {
      return <span className="state-chip bad">UNRECOGNIZED</span>;
    }
    return <span className="state-chip good">{recognition.gloss}</span>;
  }, [recognition]);

  return (
    <>
      <header className="app">
        <div>
          <p className="kicker">VerbX / IvishAI partner challenge · PS 3</p>
          <h1>ISL Bridge</h1>
          <p>
            Documented <strong>Indian Sign Language (ISL)</strong> glosses, both ways with text and
            multilingual speech. VerbX does not recognize sign gestures — recognition is MediaPipe +
            our local model.
          </p>
        </div>
        <div className="badge-row">
          <span className="badge teal">ISL named</span>
          <span className="badge warn">UNRECOGNIZED state</span>
          <span className="badge">{health?.verbx_configured ? "VerbX key on server" : "VerbX fallback mode"}</span>
          <span className="badge ok">{consented ? "Consent on file" : "No consent"}</span>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((item) => (
          <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>
      <main>
        {error && <div className="banner alert">{error}</div>}

        {tab === "signs" && catalog && (
          <section className="panel">
            <p className="kicker">TC1</p>
            <h2>{catalog.sign_language}</h2>
            <p className="muted">{catalog.scope}</p>
            <div className="banner info">{catalog.data_origin}</div>
            <p>
              Supported count: <strong>{catalog.signs.length}</strong> · Reverse/speech languages:{" "}
              {catalog.languages.join(", ").toUpperCase()} · Unknown poses return{" "}
              <strong>{catalog.unrecognized_state}</strong>
            </p>
            <div className="grid cards">
              {catalog.signs.map((sign) => (
                <article className="sign-card" key={sign.id}>
                  <SignGlyph id={sign.id} />
                  <h3>
                    {sign.gloss}{" "}
                    <span className="muted">
                      {sign.en} / {sign.hi}
                    </span>
                  </h3>
                  <p className="how">{sign.how}</p>
                  <p className="muted">Hands: {sign.hands} · {sign.include_category}</p>
                </article>
              ))}
            </div>
            <div className="row" style={{ marginTop: "1rem" }}>
              <button
                className="btn"
                onClick={() =>
                  markRun(
                    "TC1",
                    `Published ${catalog.signs.length} named ISL glosses.`,
                    "pass",
                    { ids: catalog.signs.map((s) => s.id) }
                  )
                }
              >
                Save TC1 evidence
              </button>
            </div>
          </section>
        )}

        {tab === "sign2speech" && (
          <section className="grid two">
            <div className="panel">
              <p className="kicker">TC2 · Sign → English + Hindi speech</p>
              <h2>Hold a supported ISL sign</h2>
              <CameraStage ref={cameraRef} enabled={consented} statusText="Local recognizer, not VerbX" />
              <div className="row" style={{ marginTop: "0.8rem" }}>
                <button className="btn" disabled={!consented || busy} onClick={classify}>
                  {busy === "recognize" ? "Classifying…" : "Classify current pose"}
                </button>
                <button
                  className="btn secondary"
                  disabled={recognition?.state !== "recognized" || busy}
                  onClick={speakRecognized}
                >
                  {busy === "speak" ? "Speaking…" : "Approve & speak EN + HI"}
                </button>
              </div>
              <p className="muted">Speech is not sent until you approve. Random motion should stay UNRECOGNIZED.</p>
            </div>
            <div className="panel">
              <h3>Result {recChip}</h3>
              {recognition?.state === "UNRECOGNIZED" && (
                <div className="banner alert">
                  {recognition.reason} ({recognition.code}) Confidence {recognition.confidence} / threshold{" "}
                  {recognition.threshold}. No TTS.
                </div>
              )}
              {recognition?.state === "recognized" && (
                <div className="banner ok">
                  {catalog?.sign_language}: <strong>{recognition.gloss}</strong> ({recognition.en} / {recognition.hi})
                  · confidence {recognition.confidence}
                </div>
              )}
              {speechOut && (
                <>
                  <p>
                    English: {speechOut.english.text} ({speechOut.english.tts.provider})
                    <br />
                    Hindi: {speechOut.hindi.text} ({speechOut.hindi.tts.provider})
                  </p>
                  <button className="btn ghost" onClick={() => playPair(speechOut)}>
                    Play both again
                  </button>
                </>
              )}
              <pre className="log">{JSON.stringify(recognition || { hint: "Classify after consent." }, null, 2)}</pre>
              <div className="row">
                <button
                  className="btn"
                  disabled={!speechOut}
                  onClick={() =>
                    markRun("TC2", `Spoke ${recognition?.gloss} in English and Hindi.`, "pass", {
                      gloss: recognition?.gloss,
                      providers: [speechOut?.english?.tts?.provider, speechOut?.hindi?.tts?.provider],
                    })
                  }
                >
                  Save TC2 pass
                </button>
              </div>
            </div>
          </section>
        )}

        {tab === "speech2sign" && (
          <section className="grid two">
            <div className="panel">
              <p className="kicker">TC3 · Speech → text → supported ISL sign</p>
              <h2>Talk to a signer</h2>
              {!consented && <div className="banner alert">Microphone stays off until consent is recorded.</div>}
              <div className="row">
                <button className="btn" disabled={!consented || busy} onClick={recordAndStt}>
                  {busy === "record" ? "Recording 3.5s…" : "Record → VerbX STT"}
                </button>
                <button className="btn secondary" disabled={!consented} onClick={startBrowserStt}>
                  {listening ? "Listening…" : "Browser STT fallback"}
                </button>
              </div>
              <label htmlFor="transcript">Transcript (you may correct it)</label>
              <textarea id="transcript" rows={3} value={transcript} onChange={(e) => setTranscript(e.target.value)} />
              <button className="btn ghost" disabled={!transcript} onClick={() => matchTranscript(transcript)}>
                Map to supported sign
              </button>
            </div>
            <div className="panel">
              <h3>Reverse route</h3>
              {sttOut?.reverse_route?.matched ? (
                <div className="sign-card">
                  <SignGlyph id={sttOut.reverse_route.sign.id} />
                  <h3>{sttOut.reverse_route.sign.gloss}</h3>
                  <p>
                    {sttOut.reverse_route.sign.en} / {sttOut.reverse_route.sign.hi}
                  </p>
                  <p className="how">{sttOut.reverse_route.sign.how}</p>
                </div>
              ) : (
                <div className="banner alert">
                  {sttOut?.reverse_route?.reason || "No match yet. Unsupported phrases stay unsupported."}
                </div>
              )}
              <p className="muted">STT provider: {sttOut?.stt?.provider || "—"}</p>
              <pre className="log">{JSON.stringify(sttOut, null, 2)}</pre>
              <button
                className="btn"
                disabled={!sttOut}
                onClick={() =>
                  markRun(
                    "TC3",
                    sttOut.reverse_route?.matched
                      ? `Mapped speech to ${sttOut.reverse_route.sign.gloss}`
                      : `Unsupported phrase: ${sttOut.reverse_route?.state}`,
                    sttOut.reverse_route?.matched ? "pass" : "difference",
                    sttOut
                  )
                }
              >
                Save TC3 evidence
              </button>
            </div>
          </section>
        )}

        {tab === "dataset" && catalog && (
          <DatasetRecorder signs={catalog.signs} enabled={consented} />
        )}

        {tab === "safety" && (
          <section className="grid two">
            <div className="panel">
              <p className="kicker">TC4 · Consent and uncertainty</p>
              <h2>Filming consent</h2>
              <label>Person on camera</label>
              <input value={form.subject_name} onChange={(e) => setForm({ ...form, subject_name: e.target.value })} />
              <label>Operator / teammate</label>
              <input value={form.operator_name} onChange={(e) => setForm({ ...form, operator_name: e.target.value })} />
              <label>Purpose</label>
              <textarea rows={3} value={form.purpose} onChange={(e) => setForm({ ...form, purpose: e.target.value })} />
              <div className="row" style={{ marginTop: "0.8rem" }}>
                <button className="btn" disabled={!form.subject_name || !form.operator_name} onClick={() => grantConsent(true)}>
                  I consent to being filmed
                </button>
                <button className="btn warn" onClick={() => grantConsent(false)}>
                  Revoke / refuse
                </button>
              </div>
              <div className="banner info">
                VerbX APIs used: STT, TTS, translation. VerbX does not recognize sign gestures natively.
                Keys stay in <code>backend/.env</code>.
              </div>
              <h3>Unknown-sign check</h3>
              <p className="muted">
                After consent, go to Sign → speech, wave randomly, classify. The chip must read UNRECOGNIZED and
                Approve & speak must stay disabled.
              </p>
              <button
                className="btn"
                onClick={() =>
                  markRun(
                    "TC4",
                    consented
                      ? `Consent id ${consent.id}. Unrecognized demo: ${recognition?.state || "not yet run"}.`
                      : "Camera blocked without consent.",
                    !consented || recognition?.state === "UNRECOGNIZED" ? "pass" : "pending",
                    { consent, unrecognized: recognition?.state }
                  )
                }
              >
                Save TC4 evidence
              </button>
            </div>
            <div className="panel">
              <h3>Official test runs</h3>
              <table className="runs">
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Status</th>
                    <th>Actual</th>
                  </tr>
                </thead>
                <tbody>
                  {runs
  .filter((run) => ["TC1", "TC2", "TC3", "TC4"].includes(run.case_id))
  .map((run) => (
                    <tr key={run.case_id}>
                      <td>
                        <strong>{run.case_id}</strong>
                        <div className="muted">{run.title}</div>
                      </td>
                      <td>{run.status}</td>
                      <td>{run.actual || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <h3>Audit log</h3>
              <pre className="log">{JSON.stringify(events.slice(0, 12), null, 2)}</pre>
            </div>
          </section>
        )}
      </main>
    </>
  );
}
