# ISL Bridge

**Partner Challenge:** VerbX / IvishAI (PS 3)

ISL Bridge connects a documented subset of **Indian Sign Language (ISL)** gestures with text and multilingual speech in both directions.

VerbX is used for **speech-to-text (STT), text-to-speech (TTS), and translation only**. It does not recognize sign gestures. Sign recognition uses **MediaPipe Hands landmarks and a local geometric classifier**. Unsupported or uncertain poses return `UNRECOGNIZED`.

The camera and microphone remain off until the user provides consent.

## What It Demonstrates

| Test Case | Demonstration |
|---|---|
| TC1 | Named ISL catalog containing 10 isolated glosses |
| TC2 | Supported sign → English and Hindi speech, with approval required before TTS |
| TC3 | Speech → editable transcript → matching ISL sign card, or an unsupported-phrase message |
| TC4 | Without consent, camera access is blocked; random or unsupported motion returns `UNRECOGNIZED` without speech |

**Languages:** English and Hindi
**Sign language:** Indian Sign Language (ISL), using a limited subset of isolated words rather than full ISL grammar.

## Features

- Local hand-landmark processing and geometric sign classification.
- A catalog of 10 supported isolated ISL glosses.
- English and Hindi text and speech support.
- Speech input with an editable transcript.
- Consent-controlled camera and microphone access.
- An explicit `UNRECOGNIZED` state for unsupported poses.
- Browser-based speech fallbacks when VerbX is not configured.

## Data Origin and Licensing

The glosses were scoped from documented ISL teaching practice and the INCLUDE lexicon:

Sridhar et al., ACM Multimedia 2020, [INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition](https://doi.org/10.1145/3394171.3413528).

The INCLUDE dataset is associated with the **CC BY-NC-SA 4.0** license. Review the dataset's license and attribution requirements before redistributing or using its material.

Live video is processed only after consent. This application does not bundle INCLUDE dataset videos. Do not distribute dataset material unless the applicable license permits it.

## Technologies Used

- **Frontend:** React and Vite
- **Backend:** Python and FastAPI
- **Hand landmarks:** MediaPipe Hands
- **Storage:** SQLite for consent records
- **Speech and translation:** VerbX when configured; browser and catalog fallbacks where supported

## Run Locally

### Prerequisites

Install Python, Node.js with npm, and Git.

### 1. Set Up the Backend

Open a terminal in the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

If you use VerbX, add your API key and any required configuration to `backend/.env`.

**Never commit your real API key or `.env` file.**

Start the backend on port `8001`:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Keep this terminal open.

### 2. Set Up the Frontend

Open a second terminal in the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the application in your browser:

http://127.0.0.1:5173

The Vite development proxy forwards `/api` requests to `http://127.0.0.1:8001`. Keep this setting in `frontend/vite.config.js` consistent with the backend port.

## Evaluation (recognition accuracy)

The repository includes tools to measure how well the recognizer works on real people, and to compare it with learned baselines.

### 1. Record labelled clips

Run the app locally, grant consent, and open the **Dataset recorder** tab.

- Choose a signer code (S1, S2, …), the lighting, and a label, then press **Record sample**. Each sample is a 2.5-second clip of MediaPipe hand landmarks, the same window that **Classify current pose** uses. No video is stored.
- Record every sign several times (20+ per signer is a good target). Also record **NEGATIVE** clips: random movements and hand shapes that are not one of the 10 signs.
- Record other people only with their consent, using anonymous codes.
- Press **Export JSONL** often and save the files in `backend/data/samples/`. This folder is git-ignored by default.

### 2. Run the evaluation

```powershell
cd backend
pip install -r requirements-eval.txt
python -m evaluation.evaluate data/samples/*.jsonl --title "ISL Bridge evaluation"
```

The script writes `runs/eval/<timestamp>/report.md`, `metrics.json` and confusion-matrix images. It reports:

- **Rule-based recognizer (deployed):** sign accuracy, wrong-sign rate, UNRECOGNIZED rate, false-accept rate on NEGATIVE clips, macro-F1, per-sign recall, the reasons for UNRECOGNIZED, and breakdowns by signer and by lighting.
- **Learned baselines** (logistic regression, k-NN and random forest) on wrist-normalized landmark features:
  - Leave-one-signer-out evaluation, which estimates performance on a new user.
  - Pooled k-fold evaluation, which is optimistic because the same people appear in training and test data.

`python -m evaluation.synthetic --out synthetic.jsonl` generates **synthetic stick-figure clips for testing the pipeline only**. Results on synthetic data are not accuracy results.

### Results

_Not yet measured. Results will be added here after recording real clips._

## Configure VerbX

1. Register using your Ziro email at [VerbX](https://one.verbxeco.com).
2. Eligible teams can redeem the `ZIRO26` offer once, subject to the provider's terms and eligibility. The stated credit period ends on **31 October 2026 at 23:59 IST**.
3. Store your API key in `backend/.env`.
4. If your VerbX dashboard documents different REST paths or authentication settings, configure the corresponding environment variables:
   - `VERBX_API_BASE`
   - `VERBX_TRANSLATE_PATH`
   - `VERBX_STT_PATH`
   - `VERBX_TTS_PATH`
   - `VERBX_AUTH_MODE`

Follow the current provider documentation for endpoint paths, authentication, and usage limits.

### Without a VerbX API Key

The application can use bilingual catalog strings and browser speech APIs as fallbacks where supported.

- Browser `speechSynthesis` can provide speech output.
- Browser Speech Recognition is available only in compatible browsers.
- Catalog-based translations are limited to the supported content.

The UI should identify fallback behavior clearly. Browser or catalog fallbacks must not be represented as VerbX results.

## Privacy and Safety Guardrails

- The application identifies the sign language as Indian Sign Language (ISL).
- Unsupported poses return `UNRECOGN