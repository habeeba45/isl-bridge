import { useMemo, useRef, useState } from "react";
import CameraStage from "./CameraStage.jsx";

// Same capture window as "Classify current pose" in App.jsx, so the
// evaluation measures exactly what the live recognizer sees.
const CAPTURE_MS = 2500;
const GET_READY_MS = 1500;
const NEGATIVE = "NEGATIVE";

function roundFrames(frames) {
  // 4 decimals keeps sub-pixel precision at 640x480 and shrinks the file.
  return frames.map((f) => ({
    hands: (f.hands || []).map((hand) =>
      hand.map((p) => ({
        x: Math.round(p.x * 1e4) / 1e4,
        y: Math.round(p.y * 1e4) / 1e4,
        z: Math.round((p.z || 0) * 1e4) / 1e4,
      }))
    ),
  }));
}

export default function DatasetRecorder({ signs, enabled }) {
  const cameraRef = useRef(null);
  const [signer, setSigner] = useState("S1");
  const [lighting, setLighting] = useState("normal");
  const [label, setLabel] = useState(signs[0]?.id || NEGATIVE);
  const [samples, setSamples] = useState([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const labels = useMemo(() => [...signs.map((s) => s.id), NEGATIVE], [signs]);

  const counts = useMemo(() => {
    const c = {};
    for (const s of samples) {
      const key = `${s.signer}|${s.label}`;
      c[key] = (c[key] || 0) + 1;
    }
    return c;
  }, [samples]);

  const signers = useMemo(
    () => [...new Set(samples.map((s) => s.signer))].sort(),
    [samples]
  );

  async function record() {
    if (!enabled || busy) return;
    setBusy(true);
    try {
      setStatus("Get ready…");
      await new Promise((r) => setTimeout(r, GET_READY_MS));
      cameraRef.current?.clear?.();
      setStatus(`Recording ${label}…`);
      await new Promise((r) => setTimeout(r, CAPTURE_MS));
      const frames = cameraRef.current?.frames() || [];
      const withHand = frames.filter((f) => f.hands?.length).length;
      if (frames.length < 15) {
        setStatus("Too few camera frames — sample discarded. Try again.");
        return;
      }
      const sample = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        label,
        signer: signer.trim() || "S1",
        lighting,
        recorded_at: new Date().toISOString(),
        capture_ms: CAPTURE_MS,
        n_frames: frames.length,
        frames_with_hand: withHand,
        frames: roundFrames(frames),
      };
      setSamples((prev) => [...prev, sample]);
      setStatus(
        `Saved ${label} (${frames.length} frames, hand visible in ${withHand}).`
      );
    } finally {
      setBusy(false);
    }
  }

  function exportJsonl() {
    if (!samples.length) return;
    const body = samples.map((s) => JSON.stringify(s)).join("\n") + "\n";
    const blob = new Blob([body], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `isl_samples_${signers.join("-") || "S"}_${stamp}.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
    setStatus(`Exported ${samples.length} samples.`);
  }

  return (
    <section className="panel">
      <h2>Dataset recorder (for offline evaluation)</h2>
      <p className="muted">
        Records labelled 2.5-second hand-landmark clips — the same window that
        “Classify current pose” uses. Only landmark coordinates are kept, never
        video. Samples stay in this browser tab until you export them; nothing
        is uploaded. Use an anonymous signer code (S1, S2…), and record other
        people only with their consent. Use <strong>NEGATIVE</strong> for
        random movements and hand shapes that are not one of the 10 signs.
        Export often — reloading the page loses unexported samples.
      </p>
      <div className="grid two">
        <div>
          <CameraStage
            ref={cameraRef}
            enabled={enabled}
            statusText={status || "Dataset mode"}
          />
        </div>
        <div className="grid">
          <label>
            Signer code
            <input value={signer} onChange={(e) => setSigner(e.target.value)} />
          </label>
          <label>
            Lighting
            <select value={lighting} onChange={(e) => setLighting(e.target.value)}>
              <option value="normal">normal</option>
              <option value="dim">dim</option>
              <option value="bright">bright / window</option>
              <option value="backlit">backlit</option>
            </select>
          </label>
          <label>
            Label
            <select value={label} onChange={(e) => setLabel(e.target.value)}>
              {labels.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </label>
          <div className="row">
            <button className="btn" disabled={!enabled || busy} onClick={record}>
              {busy ? "Recording…" : "Record sample"}
            </button>
            <button
              className="btn ghost"
              disabled={!samples.length || busy}
              onClick={() => setSamples((prev) => prev.slice(0, -1))}
            >
              Undo last
            </button>
            <button className="btn secondary" disabled={!samples.length} onClick={exportJsonl}>
              Export JSONL ({samples.length})
            </button>
          </div>
          {!enabled && (
            <div className="banner info">Grant consent on the Safety tab first.</div>
          )}
          <table className="runs">
            <thead>
              <tr>
                <th>Label</th>
                {(signers.length ? signers : [signer]).map((s) => (
                  <th key={s}>{s}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((l) => (
                <tr key={l}>
                  <td>{l}</td>
                  {(signers.length ? signers : [signer]).map((s) => (
                    <td key={s}>{counts[`${s}|${l}`] || 0}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
