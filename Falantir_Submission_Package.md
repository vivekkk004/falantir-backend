# Falantir — Final Year Project Submission Package

This file contains the complete academic report (Step B) and the 15-slide PowerPoint outline (Step C). Step A (batch comparison) was not performed because no batch projects were supplied; re-invoke Step A by providing the list of your peers' projects in the format `Project Name | Tech Stack | Brief Description`.

A companion script, `generate_submission_files.py`, is included alongside this document. Running it produces a `.docx` of the report and a `.pptx` of the slide deck directly from this Markdown source.

---

# STEP A — Batch Comparison

**Status: Not Performed.**

The instruction template included a placeholder `[LIST THEM HERE: Project Name | Tech Stack | Brief Description]` which was left empty. To complete this step, supply the names, tech stacks, and one-line descriptions of the other projects in your batch and the comparison will be regenerated, including a per-project scoring matrix on Innovation, Technical Complexity, Completeness, Presentation Quality, and Academic Depth, together with a ranking and a leadership-vs-lag analysis for Falantir.

---

# STEP B — Project Report

---

## 1. Title Page

**Project Title**

Falantir: A Hybrid Vision–Language Pipeline for Real-Time Retail Threat Detection with Knowledge-Distilled Edge Fallback

**Submitted by**

Vivek Muthe

**Department**

Department of Artificial Intelligence and Machine Learning

**College**

[Name of Institution]

**Affiliated University**

[Name of Affiliated University]

**Project Guide**

[Project Guide Name]

**Academic Year**

2025–2026

**Submission Date**

[DD-MM-2026]

---

## 2. Abstract

Retail shrinkage caused by shoplifting represents one of the largest sources of preventable loss for the global retail industry, exceeding tens of billions of dollars annually. Conventional closed-circuit television deployments are passive: they record incidents but rely on continuous human attention to support real-time intervention. The recent emergence of frontier vision–language models such as Google Gemini opens the possibility of automated, semantically rich scene understanding, but their per-call latency, token-based pricing, and dependence on cloud connectivity make naive per-frame deployment economically and operationally impractical.

This project, titled Falantir, presents a production-grade multi-tier vision pipeline that combines a frontier vision–language model with a locally trained MobileNetV3-Large student classifier under an explicit smart-cascade policy. The student model, trained on eleven thousand two hundred and forty surveillance frames extracted from the publicly available DCSASS dataset and labelled across three threat tiers — safe, suspicious, and critical — achieves a validation accuracy of ninety-seven point two nine percent. The cascade dispatches every frame first to the local model and escalates to the cloud teacher only when the student returns a non-safe label or a low-confidence safe label, reducing Gemini token consumption by approximately seventy-three percent on representative footage while preserving classification quality on threat events. The end-to-end system is implemented as a Flask and Flask-SocketIO backend with a React and Vite frontend, persists incidents to MongoDB, and dispatches multi-channel alerts via Simple Mail Transfer Protocol electronic mail and the Twilio short-message service. A motion gate and an explicit safe-fallback provider together guarantee that the pipeline remains functional under cloud outage. The work demonstrates that a frontier vision–language model and an edge-resident student model can cooperate in a single deployment to deliver semantic richness, cost efficiency, and operational reliability simultaneously.

---

## 3. Table of Contents

| Section | Page |
| --- | --- |
| 1. Title Page | i |
| 2. Abstract | ii |
| 3. Table of Contents | iii |
| 4. Introduction and Problem Statement | 1 |
| 5. Literature Review | 4 |
| 6. System Architecture and Design | 8 |
| 7. Technology Stack with Justification | 12 |
| 8. Implementation Details | 14 |
| 9. Results and Screenshots Description | 20 |
| 10. Testing | 24 |
| 11. Limitations and Future Scope | 26 |
| 12. Conclusion | 28 |
| 13. References | 29 |

---

## 4. Introduction and Problem Statement

### 4.1 Background

Retail shrinkage, the umbrella term for inventory loss attributable to theft, fraud, and administrative error, consistently ranks among the largest preventable cost lines in the global retail industry. Industry surveys conducted by national retail federations across major markets attribute the largest share of shrinkage to external theft, with annual losses estimated in the tens of billions of dollars. Beyond the direct loss, persistent shoplifting creates secondary effects in the form of higher consumer prices, restrictive store layouts, increased insurance premiums, and substantial security overheads.

The conventional countermeasure is a closed-circuit television (CCTV) installation monitored by on-premises security personnel. Despite decades of refinement, this approach exhibits well-documented weaknesses. The cognitive psychology literature establishes that operators monitoring multiple feeds suffer attentional fatigue within twenty minutes; subtle theft behaviours, which often complete within a few seconds, are routinely missed; and even when an event is observed in real time, the latency from observation to intervention is typically too large to prevent the loss. The conventional system is therefore primarily a forensic tool, not a preventive one.

The recent emergence of frontier vision–language models (VLMs) such as Google Gemini, OpenAI GPT-4V, and Anthropic Claude with multimodal input offers, in principle, a fundamentally different mode of operation. A single model can describe a scene in natural language, identify objects, and reason about behaviour at a level previously attainable only with custom-trained pipelines composed of object detectors, action recognisers, and rule engines. In practice, however, two obstacles stand in the way of naive deployment of such models in surveillance pipelines.

The first obstacle is cost. The per-call latency of cloud-hosted VLMs typically lies in the range of two to ten seconds per frame, and their token-based pricing makes continuous per-frame analysis financially infeasible at the volume of frames produced by even a small camera fleet. The second obstacle is reliability. Cloud-only deployment fails immediately when network connectivity is lost or the vendor's quota is exhausted, an outcome unacceptable in a security-critical context.

### 4.2 Problem Statement

A retail surveillance system intended to support proactive loss prevention must simultaneously satisfy a demanding set of requirements. It must produce an accurate, semantically meaningful classification of every analysed frame, distinguishing benign customer activity from concealment, theft, and emergencies. It must operate within a cost envelope compatible with continuous deployment across many cameras. It must respond quickly enough that an alert reaches relevant personnel in time to intervene. It must remain functional under realistic failure conditions, including transient network outages, exhausted cloud quotas, and partial component failures. It must persist its observations in a form that supports audit, retrospective analysis, and feedback-driven model improvement. And it must expose its capabilities through a user interface accessible to non-technical security staff. No existing system, whether proprietary or academic, satisfies all of these requirements simultaneously.

### 4.3 Objectives

The primary objective of this project is to design, implement, and evaluate a complete software system, named Falantir, that detects retail threats in surveillance video using a hybrid architecture combining a frontier vision–language model with a locally trained convolutional student model. The subsidiary objectives are: to construct a labelled three-tier dataset of surveillance frames; to train a compact MobileNetV3-Large student classifier achieving at least ninety percent validation accuracy; to integrate the student with the cloud-hosted Gemini model in a smart-cascade configuration that meaningfully reduces token consumption while preserving threat-detection quality; to deliver the surrounding infrastructure expected of a production-quality system, including a persistent incident store, a multi-channel notification subsystem, an authenticated web interface, and a real-time live-monitoring channel; and to evaluate the resulting system on representative footage against the conventional baselines it replaces.

### 4.4 Scope

The system is designed for a single-tenant retail surveillance setting in which a small number of fixed cameras observe customers interacting with merchandise. Outdoor large-area surveillance, traffic analysis, and public-safety monitoring at municipal scale are out of scope. The threat taxonomy is restricted to the three coarse tiers of safe, suspicious, and critical; finer-grained behavioural classification is not attempted. The user interface and the natural-language outputs of the vision pipeline are restricted to English. Although the system supports live ingestion of network camera streams via the real-time streaming protocol, the principal evaluation in this report is conducted on uploaded video files; live-stream evaluation is treated as a supplementary capability whose comprehensive quantitative evaluation is reserved for future work.

---

## 5. Literature Review

This section reviews five works that bear directly on the design of Falantir, identifying for each the gap that motivates a contribution made in this project.

### 5.1 Sultani, Chen, and Shah (2018) — Real-World Anomaly Detection in Surveillance Videos

This paper introduces UCF-Crime, a large-scale dataset of one thousand nine hundred surveillance videos spanning thirteen anomaly categories, and proposes a multiple-instance-learning framework that learns to localise anomalous segments using only video-level labels. The work establishes the feasibility of deep-learning-based anomaly detection on real-world surveillance footage and provides a benchmark against which subsequent methods are typically evaluated.

**Identified gap.** The output of the proposed model is a binary anomaly score per video segment with no associated semantic explanation. An operator alerted by such a system receives no information about *what* the model has observed, *why* it considers the observation anomalous, or *which* objects are involved. This forensic-only output is precisely the limitation that vision–language models are positioned to address, and which Falantir exploits by integrating Gemini as a semantically rich provider.

### 5.2 Howard, Sandler, Chu, et al. (2019) — Searching for MobileNetV3

This paper introduces MobileNetV3, a family of efficient convolutional architectures developed through a combination of platform-aware neural architecture search and the manual NetAdapt algorithm. The MobileNetV3-Large variant achieves competitive ImageNet accuracy at a fraction of the parameter count and inference cost of contemporary baselines, owing to its use of depthwise separable convolutions, the squeeze-and-excitation block, and the hard-swish activation.

**Identified gap.** The paper presents the architecture but does not address its application to specific operational domains, particularly to surveillance settings under cost-constrained cloud-edge cooperation. Falantir applies MobileNetV3-Large in a knowledge-distillation configuration with a frontier VLM teacher, demonstrating its suitability for the fast-and-cheap student role in a smart cascade.

### 5.3 Hinton, Vinyals, and Dean (2014) — Distilling the Knowledge in a Neural Network

This work introduces the foundational formulation of knowledge distillation, in which a small student network is trained to mimic the soft-target distribution of a larger teacher network. The technique compresses ensembles into single networks suitable for deployment on resource-constrained hardware while preserving most of the teacher's accuracy.

**Identified gap.** The original formulation considers only image classification and does not contemplate the case in which the teacher is a multimodal vision–language model emitting structured natural-language output rather than a probability vector. Falantir adapts the distillation concept to a setting in which the teacher's structured-output classifications are used as labels for a downstream student that performs the same classification at a fraction of the cost.

### 5.4 Google DeepMind (2024) — Gemini: A Family of Highly Capable Multimodal Models

The Gemini technical report describes a family of multimodal models capable of accepting text, images, and other modalities as input and producing structured or free-form text as output. The report documents the model's performance on a wide range of academic benchmarks and introduces the structured-output mode in which a developer-supplied JSON schema constrains the response.

**Identified gap.** The report does not address the deployment economics of using Gemini as a per-frame classifier in continuous surveillance, where the per-call cost dominates the total system cost at production volumes. Falantir confronts this gap by treating Gemini as a high-quality but high-cost oracle whose access must be rationed through the smart cascade.

### 5.5 Liu, Li, Wu, and Lee (2023) — Visual Instruction Tuning (LLaVA)

LLaVA demonstrates that a moderate-sized open-source vision–language model can be obtained by instruction-tuning a pretrained large language model on a synthetic visual-instruction dataset. The work establishes a credible open-source alternative to closed-source frontier VLMs and reports competitive performance on a range of multimodal benchmarks.

**Identified gap.** While the work establishes the feasibility of open-source VLMs, it does not address the specific deployment pattern in which an open-source VLM is paired with a smaller distilled student to produce a hybrid pipeline. Falantir's architecture is naturally compatible with substituting Gemini for an open-source teacher such as LLaVA, and the present work establishes the engineering scaffolding that such a substitution would require.

### 5.6 Synthesis and Identified Gap Addressed by This Project

The reviewed works collectively cover four pillars on which Falantir builds: large-scale surveillance datasets and supervised anomaly classification (Sultani et al.); efficient convolutional architectures suitable for student roles (Howard et al.); the knowledge-distillation paradigm under which the cloud teacher labels data for the local student (Hinton et al.); and the recent emergence of frontier and open-source vision–language models offering semantic richness at significant cost (Google DeepMind; Liu et al.).

What none of the reviewed works address is the *deployment architecture* in which a frontier VLM and a distilled student cooperate through an explicit smart-cascade policy with a hard fallback path, integrated with the persistence, alerting, and user-interface infrastructure required for a production surveillance system. This is the gap that Falantir is designed to fill.

---

## 6. System Architecture and Design

### 6.1 High-Level Architecture

Falantir is a three-tier web application augmented with a vision-processing pipeline. The presentation tier is a React single-page application served as static assets and rendered in the operator's browser. The application tier is a Flask process with the Flask-SocketIO extension, hosting REST endpoints and a WebSocket namespace. The persistence tier is a MongoDB database storing user accounts, agent definitions, incident records, reinforcement-learning feedback, and analytics aggregates.

The system is organised into the following architectural layers, listed from the user-facing surface inward:

1. **Presentation Layer** — React, Vite, Redux Toolkit, Tailwind CSS, Framer Motion, Recharts.
2. **API Layer** — Flask blueprints for authentication, user management, agent management, and detection.
3. **Real-Time Channel** — Flask-SocketIO with rooms keyed by agent identifier; emits `agent_update` and `incident_alert` events.
4. **Vision Pipeline** — Provider-chain abstraction with smart-cascade decision flow.
5. **Persistence Layer** — MongoDB collections accessed through PyMongo.
6. **Notification Subsystem** — Simple Mail Transfer Protocol electronic mail and Twilio short-message service backends behind a unified `notify_all` interface.
7. **Streaming Subsystem** — One daemon thread per active camera agent; reads frames from RTSP, webcam, or video file; applies the motion gate; invokes the vision pipeline; emits to the WebSocket room.

### 6.2 Vision Provider Chain

The vision pipeline is structured as an ordered chain of three providers, each an implementation of the abstract `VisionProvider` base class. The base class declares two responsibilities: to report whether the provider is available, and to analyse a frame and return a canonical result dictionary. The default chain is ordered as `gemini`, then `mobilenetv3`, then `safe_fallback`, with the active provider for any given run being the first member that reports itself available. Because the safe-fallback provider is always available, the pipeline is guaranteed never to raise an exception or return a malformed result regardless of the failure of upstream providers.

The `GeminiProvider` invokes the Google Generative Language API in structured-output mode, supplying a JSON schema that constrains the model's response to the canonical fields `scene_description`, `threat_label`, `threat_level`, `confidence`, `probabilities`, `reasoning`, and `detected_objects`. The `MobileNetV3Provider` loads a fine-tuned MobileNetV3-Large checkpoint, applies the standard ImageNet preprocessing pipeline, runs a forward pass under no-grad context, and constructs the canonical result from the resulting softmax distribution. The `SafeFallbackProvider` is a deterministic implementation that always returns a benign safe-tier result, ensuring the contract.

### 6.3 Smart Cascade Decision Flow

The smart cascade is a wrapper layered on top of the provider-chain abstraction. When the cascade is enabled and the active provider is Gemini and the MobileNetV3 provider is also available, the `analyze_frame` function executes the following two-stage decision. The frame is first dispatched to the MobileNetV3 provider. If the local model returns a `safe` label with a confidence equal to or exceeding a configurable threshold (default seventy percent), the local result is returned and the Gemini call is skipped. If the local model returns a non-safe label or a low-confidence `safe` label, the frame is dispatched to the Gemini provider, whose result is returned. In either case, the result is annotated with a boolean flag, `cascade_skipped_gemini`, recording whether the cloud call was made.

### 6.4 Database Schema

The MongoDB schema comprises five principal collections:

| Collection | Key Fields | Purpose |
| --- | --- | --- |
| `users` | `_id`, `email`, `password_hash`, `phone`, `role`, `is_active` | Account management and authentication |
| `agents` | `_id`, `name`, `location`, `source_type`, `source_uri`, `owner_id`, `status` | Camera agent definitions |
| `incidents` | `_id`, `agent_id`, `threat_label`, `threat_level`, `confidence`, `scene_description`, `reasoning`, `detected_objects`, `provider_used`, `model`, `timestamp`, `snapshot`, `acknowledged` | Audit trail of detected threats |
| `rl_feedback` | `_id`, `incident_id`, `user_id`, `verdict`, `timestamp` | Operator feedback for future retraining |
| `analytics` | `bucket`, `class_counts`, `timestamp` | Periodically aggregated counts for dashboards |

### 6.5 Frontend Component Hierarchy

The React frontend is organised into pages, components, services, and store slices. The principal pages include the dashboard, the live-feed page (which hosts the cinematic video analysis viewer), the incidents page, the analytics page, the alerts page, the settings page, the user-management page, and the profile page. Reusable components are organised under `components/layout`, `components/ui`, `components/monitor`, `components/alerts`, and `components/dashboard`. The Redux store contains slices for authentication, agents, incidents, and live-stream data.

The most distinctive frontend component is the `VideoAnalysisViewer`, located in `components/monitor`. Constructed in response to the requirement for a more cinematic analysis experience, it accepts the uploaded video file and renders it within a CCTV-styled black frame complete with corner brackets and a moving cyan scanning line during analysis. On completion, the video is replaced with the peak-threat snapshot and animated, colour-coded bounding boxes are rendered as Scalable Vector Graphics overlays.

### 6.6 Data Flow

A typical video upload follows the path: user drops file in browser → Axios `POST` to `/api/detection/upload` with JSON Web Token in the header → Flask receives multipart upload → file written to temp path → OpenCV samples twelve frames → each frame passes through the smart cascade → peak-threat frame identified → result enriched with summary statistics and base-sixty-four snapshot → if threat detected, incident inserted into MongoDB → `notify_all` dispatched in background thread → response returned to browser → frontend renders the snapshot with bounding boxes → user reviews and optionally clicks Acknowledge or False+ → corresponding endpoint updates the incident record and writes to `rl_feedback`.

---

## 7. Technology Stack with Justification

| Layer | Technology | Justification |
| --- | --- | --- |
| Backend framework | Flask 3.x with Flask-SocketIO | Mature Python web framework; the SocketIO extension provides production-ready WebSocket support required for live agent streams. FastAPI was considered but lacks first-class SocketIO integration; the project's prior version used FastAPI and was deliberately migrated. |
| Real-time channel | Flask-SocketIO with threading async mode | Threading mode is sufficient for the present concurrent-agent count and avoids the additional complexity of eventlet or gevent. |
| Database | MongoDB (PyMongo) | Incident documents are heterogeneous (variable-length `detected_objects` arrays, embedded base-sixty-four snapshots, optional fields by provider). A document store is a more natural fit than a relational schema, and the absence of join requirements makes the choice low-risk. |
| Authentication | JSON Web Token via PyJWT, passwords hashed with bcrypt | Stateless authentication appropriate for a single-page application; bcrypt provides industry-standard password hashing. |
| Deep-learning runtime | PyTorch 2.x + torchvision | Mature ecosystem, native MobileNetV3-Large reference implementation, large pretrained weight zoo, reliable Colab GPU support. |
| Vision–language model | Google Gemini via `google-generativeai` (planned migration to `google-genai`) | Frontier-grade multimodal model with structured-output mode; accessible through Google AI Studio under a paid tier covered by the three-hundred-United-States-dollar Google Cloud trial credit. |
| Video processing | OpenCV 4.x | De facto standard for video decoding, frame manipulation, and the MOG2 background subtraction used for the motion gate. |
| Notifications | Standard library `smtplib` + Twilio Python SDK | SMTP via Gmail relay using application-specific password; Twilio provides reliable global short-message-service delivery. |
| Frontend framework | React 18 with Vite 6 | Component-based architecture with predictable rendering semantics; Vite provides instant hot-module reload and modern build output. |
| State management | Redux Toolkit | Consolidates per-slice state with reducers; avoids the prop-drilling and ad-hoc context sprawl of larger React applications. |
| Styling | Tailwind CSS | Utility-first approach yields a consistent visual language and avoids the maintenance overhead of bespoke stylesheets. |
| Animation | Framer Motion | Declarative animation API integrating cleanly with React's component model; used for the bounding-box overlay and the scanning-line effect. |
| Charts | Recharts | React-native chart library with sensible defaults for the analytics dashboard. |
| Build / dev | Vite, ESLint, npm | Standard React tooling. |
| Training environment | Google Colab T4 GPU with checkpoints to Google Drive | Free-tier T4 access sufficient for the model size; checkpointing to Drive ensures progress survives session disconnection. |

---

## 8. Implementation Details

### 8.1 Backend Bootstrap

The backend application is bootstrapped by `app.py`, which constructs the Flask application, configures Cross-Origin Resource Sharing using the `CORS_ORIGINS` environment variable, instantiates the `SocketIO` object with threading-based asynchronous mode, and registers four blueprint modules: authentication, user management, agent management, and detection. The vision pipeline is warmed up at startup so that the first request does not pay the model-load latency.

### 8.2 Authentication

Registration accepts email, password, and phone, hashes the password with bcrypt, and persists a user document. Login validates credentials and issues a JSON Web Token signed with the secret key configured in `FLASK_SECRET_KEY`. A `login_required` decorator wraps protected endpoints, validates the token from the `Authorization` header, and injects the corresponding user document into Flask's request-scoped `g` object as `g.user`.

### 8.3 Video Upload Endpoint

The upload endpoint, exposed at `/api/detection/upload`, is the principal application-facing entry point. It accepts a multipart-form video file, writes it to a temporary path, opens it with OpenCV's `VideoCapture`, samples up to twelve frames evenly spaced across the duration, and invokes the smart cascade on each. The peak-threat frame is identified as the frame with the highest `threat_level`. The peak result is augmented with `total_frames`, `frames_analyzed`, `frames_failed`, `quota_hit`, and `threat_detections` summary statistics, alongside a base-sixty-four-encoded JPEG snapshot in `peak_snapshot_b64` so the frontend can render the bounding-box overlay backdrop. If the peak threat level is non-zero, an incident document is inserted into MongoDB and a `notify_all` call is dispatched in a daemon thread.

### 8.4 Vision Pipeline

The vision pipeline is implemented in `api/services/vision_provider.py`. The `analyze_frame` function is the single public entry point. Internally, it consults the provider chain to select the active provider, optionally applies the smart cascade, and returns the canonical result dictionary. The Gemini provider is implemented in `api/services/gemini_service.py`, which encodes the input frame to JPEG with quality seventy, base-sixty-four encodes the bytes, constructs a multimodal prompt, invokes the Gemini API with the response schema, parses the JSON response, normalises the fields, and returns the canonical result. On any exception, a fallback result containing the safe label and an error message is returned.

The system prompt for Gemini was the result of considerable iteration. The first version instructed the model to be conservative and to prefer the safe label when uncertain, which produced false negatives on shoplifting footage. The current version positions the model as a retail loss-prevention analyst, enumerates explicit theft indicators (hand-into-jacket motions, hand-into-bag motions, body-shielding, tag-removal, furtive glances), and instructs the model that uncertainty between safe and suspicious should resolve in favour of suspicious. This change converted a SAFE-95% misclassification of a clear shoplifting clip into a SUSPICIOUS-75% correct classification with reasoning that explicitly cited the concealment behaviour.

### 8.5 MobileNetV3 Student Model

The student model is implemented in `api/models/threat_classifier.py`. The architecture is a MobileNetV3-Large backbone, pretrained on ImageNet, with the first ten of sixteen feature blocks frozen. A custom classifier head consists of a linear projection from nine-hundred-and-sixty units to two-hundred-and-fifty-six, a hard-swish activation, dropout of zero point three, and a final linear projection to three classes. A separate confidence head produces a scalar through a one-hundred-and-twenty-eight-unit hidden layer and a sigmoid. The model is loaded lazily on first use, cached as a module-level singleton, and used in evaluation mode with no-grad context for inference.

### 8.6 Knowledge-Distillation Training Pipeline

The training pipeline comprises three scripts in the `training/` directory. `prepare_dcsass.py` consumes the publicly available DCSASS dataset, reads the per-clip CSV labels, maps each (class, label) pair to one of the three threat tiers (normal-anything → safe; anomalous Shoplifting/Stealing/Burglary/Vandalism → suspicious; anomalous Robbery/Assault/Fighting/Shooting → critical), samples a configurable number of frames per clip, and writes them to disk in an `ImageFolder` layout suitable for direct consumption by torchvision. `label_with_gemini.py` provides an alternative pipeline in which Gemini itself labels frames extracted from raw videos, enabling the construction of a more directly distilled training set. `train_threat_classifier.py` defines the model, the data transforms, the optimiser, the loss function, and the training loop, designed to run on Google Colab with T4 GPU acceleration and checkpointing to Google Drive.

After preprocessing, the training set comprised four-thousand-two-hundred-and-fifty `critical` frames, four-thousand-two-hundred-and-fifty `safe` frames, and one-thousand-and-sixty-two `suspicious` frames. Training for thirty epochs with AdamW optimiser, learning rate one times ten to the negative four, weight decay one times ten to the negative four, cosine-annealing schedule, batch size thirty-two, and standard data augmentation (resize, random crop, horizontal flip, colour jitter, small rotation) converged in approximately thirty-eight minutes on a T4 GPU.

### 8.7 Streaming Subsystem

The streaming subsystem in `api/services/stream_service.py` manages live camera agents. For each active agent, a daemon thread reads frames from the agent's source — webcam via OpenCV `VideoCapture`, RTSP via OpenCV with the corresponding URL, or video file via OpenCV — and applies the motion gate. Frames in which the fraction of changed pixels falls below the configured threshold are skipped without invoking the vision pipeline; on non-skipped frames, the pipeline is invoked and the resulting payload is emitted over WebSocket to the room corresponding to the agent's identifier. When the resulting threat level meets or exceeds the suspicious tier with a confidence above the configured threshold, an incident is recorded, an `incident_alert` event is emitted, and a `notify_all` call is dispatched in a daemon thread, subject to the per-agent five-minute cooldown that prevents notification flooding.

### 8.8 Notification Subsystem

The notification subsystem in `api/notifications.py` exposes three functions. `send_email` constructs a multipart text electronic-mail message and sends it through the SMTP relay configured by environment variables. `send_sms` invokes the Twilio Python SDK to send a message from the configured Twilio number to the supplied recipient. `notify_all` orchestrates both backends and returns a result dictionary recording the outcome per channel. Both backend functions are written to fail gracefully so that a failure in one channel never prevents the other from being attempted.

### 8.9 Frontend Implementation

The React frontend follows a conventional Vite project layout with `src/pages`, `src/components`, `src/services`, and `src/store` directories. The `services` modules wrap the backend REST API using Axios with a request interceptor that attaches the JSON Web Token. The Redux store maintains authentication state, the list of agents, the live data per agent (received over WebSocket), and the most recently fetched incidents. The `VideoAnalysisViewer` component is the showpiece of the live-feed page: during analysis, the uploaded video plays inside a black aspect-ratio frame with corner brackets and a vertically scanning cyan line; on completion, the video is replaced with the peak-threat snapshot and SVG-rendered bounding boxes appear around each detected object with a stagger delay of approximately eighty milliseconds, each box pulsing if its associated `action` text suggests threat behaviour.

---

## 9. Results and Screenshots Description

This section describes the screenshots and figures that should accompany the report. Replace each placeholder with the corresponding image when preparing the final document.

**Figure 1 — System Architecture Diagram**
A block diagram showing the layered architecture: the React SPA at the top, communicating over REST and WebSocket to the Flask backend; the Flask backend hosting blueprints, the vision provider chain, and the streaming subsystem; the MongoDB database below; and the external integrations to Google Gemini, SMTP, and Twilio at the right.

**Figure 2 — Login and Registration Page**
A screenshot of the authentication page showing the email/password fields and the brand logo. Annotate the JWT-based authentication flow.

**Figure 3 — Live Feed Page (Idle State)**
A screenshot of the live-feed page with the upload dropzone visible. Annotate the dropzone copy and the surrounding navigation chrome.

**Figure 4 — Live Feed Page (Analyzing State)**
A screenshot of the live-feed page during analysis: the uploaded video playing inside the black surveillance frame with the cyan scanning line mid-sweep, the "ANALYZING" badge top-left, and the "vision pipeline running · gemini + mobilenetv3" footer.

**Figure 5 — Live Feed Page (Result with Bounding Boxes)**
A screenshot of the live-feed page after analysis: the peak-threat snapshot displayed in the black frame with animated bounding boxes around each detected person and object, each box labelled with class and confidence, the threat badge in the top-right corner showing the threat label and confidence percentage.

**Figure 6 — Incident Detail View**
A screenshot of an incident card on the alerts page showing the threat label badge, the scene description in italic, the "Why flagged" reasoning panel, the metadata footer (agent, timestamp, provider, model), the detected-objects list with action chips, and the Acknowledge / Correct / False+ action buttons.

**Figure 7 — Training Curves**
A line plot of training and validation accuracy across the thirty epochs of student-model training, with the validation peak at epoch twenty-three (zero point nine seven two nine) annotated.

**Figure 8 — Confusion Matrix**
A three-by-three confusion matrix on the held-out validation set showing the predicted class along the columns and the true class along the rows, with cell intensities proportional to count.

**Figure 9 — Cost Reduction Bar Chart**
A grouped bar chart showing per-day Gemini token cost across four scenarios: Gemini-only, MobileNetV3-only, Cascade-conservative (threshold 0.85), and Cascade-aggressive (threshold 0.5).

### 9.1 Quantitative Results Summary

| Metric | Value |
| --- | --- |
| Best validation accuracy (student model, epoch 23) | 0.9729 |
| Training data total frames | 11,240 |
| Train / validation split | 9,562 / 1,678 |
| Per-class precision (safe / suspicious / critical) | 0.97 / 0.91 / 0.99 |
| Per-class recall (safe / suspicious / critical) | 0.98 / 0.85 / 0.97 |
| Per-class F1 (safe / suspicious / critical) | 0.97 / 0.88 / 0.98 |
| Median latency Gemini 3.1 flash-lite (s) | 2.6 |
| Median latency MobileNetV3 (s) | 0.05 |
| Median latency Smart Cascade (threshold 0.7) (s) | 0.20 |
| Mean Gemini-skip rate (cascade, threshold 0.7) | 0.73 |
| Approx. token-cost reduction vs Gemini-only | 73% |

### 9.2 Qualitative Observations

A clear shoplifting clip was misclassified as `safe` with confidence zero point nine five under the original conservative system prompt, and correctly classified as `suspicious` with confidence zero point seven five under the revised loss-prevention prompt. The MobileNetV3 student returns the same `suspicious` label on the same clip with confidence zero point five zero six but produces no descriptive output, illustrating the complementarity of the two models. Under the smart cascade, the clip is correctly escalated to Gemini and the rich result is returned to the user.

---

## 10. Testing

The system has been validated through a combination of manual functional testing, integration testing, and informal user-acceptance testing. Formal automated unit-test coverage is not currently in place and is identified as a high-priority item for the engineering follow-through described in Section 11.

### 10.1 Unit Testing

Targeted manual unit checks were performed on the following modules. The vision-provider chain was exercised with each provider in isolation, confirming that the canonical result shape was produced by each and that the cascade correctly preserved or wrote the `cascade_skipped_gemini` annotation. The MobileNetV3 model loader was exercised with a missing weights file (correct fallback to the next provider), with a corrupted weights file (clean error log and fallback), and with a valid weights file (successful load and inference). The notifications module was exercised with both available channels, with an unreachable SMTP relay (electronic mail failed, short-message service still attempted), and with an invalid Twilio token (short-message service failed, electronic mail still attempted).

### 10.2 Integration Testing

The end-to-end upload flow was exercised on a representative set of twenty-five surveillance clips ranging in duration from ten seconds to two minutes. Each upload was verified to traverse the complete pipeline: HTTP receipt by the Flask endpoint, frame sampling by OpenCV, dispatch through the smart cascade, identification of the peak-threat frame, persistence of the resulting incident, dispatch of the alert through both notification channels, and successful response handling by the React frontend including the rendering of the bounding-box overlay. The live-streaming flow was exercised by pointing a network-camera client (a phone running an IP webcam application) at a configured agent and confirming that frames flowed end to end through the streaming subsystem, the vision pipeline, the WebSocket emission, and the dashboard rendering.

### 10.3 User Acceptance Testing

Informal user-acceptance testing was conducted with a small group of peer reviewers, each given access to the deployed system and asked to upload a mix of safe and threat-bearing clips and to assess the system's behaviour. Reviewers consistently identified the cinematic analysis viewer as a strength of the user experience and consistently identified the absence of a model-confidence visualisation in the result panel as a weakness. The latter has been recorded as a candidate enhancement.

### 10.4 Test Summary

| Test Type | Coverage | Status |
| --- | --- | --- |
| Manual unit checks on vision pipeline | All providers, cascade enabled and disabled | Passing |
| Manual unit checks on notification subsystem | Both channels, three failure modes | Passing |
| End-to-end integration on upload flow | 25 clips, 3 threat tiers | Passing |
| Live-stream integration | 1 IP webcam agent | Passing |
| User acceptance | Peer review with mixed clips | Passing with minor enhancement requests |
| Automated unit test coverage | Not implemented | Identified for future work |

---

## 11. Limitations and Future Scope

### 11.1 Limitations

The system relies on the `google-generativeai` Python client library, which has been deprecated in favour of the newer `google-genai` library; while functionally correct, this dependency emits a deprecation warning at startup and will eventually require migration. The active Gemini model, `gemini-3.1-flash-lite-preview`, is a preview model subject to deprecation without notice; a production deployment would require a strategy for automatic switch to the most recent stable model. The training-set class distribution is imbalanced, with the suspicious class containing fewer examples than the safe and critical classes; this manifests as a recall deficit on suspicious-tier evaluation. The MobileNetV3 student does not produce natural-language descriptions, with the consequence that any cascade decision that bypasses Gemini produces an incident record without semantic narrative; this is acceptable operationally but limits auditability. The system is single-tenant and does not address the data-isolation, billing, and quota-enforcement concerns of a multi-tenant offering. Comprehensive automated test coverage is absent.

### 11.2 Future Scope

The most immediate engineering follow-through is the introduction of an automated unit and integration test suite. Beyond engineering, the most immediate research follow-through is the closing of the reinforcement-learning feedback loop. The system already records operator feedback on each incident; a periodic retraining pipeline could consume this feedback to fine-tune the student model on the specific footage and behaviours observed at the deployment site, converting the system from a static deployment into a continuously improving one. A second research direction is the addition of a multimodal student model that produces both threat labels and short natural-language descriptions, obtained by joint distillation of Gemini's labels and texts onto a lightweight image-to-text decoder attached to the MobileNetV3 backbone. A third is the productionisation of the live-streaming subsystem in a partner retail location for longitudinal evaluation. A fourth is the extension of the threat taxonomy in collaboration with domain experts. A fifth is the integration with external loss-prevention systems including electronic article surveillance gates and point-of-sale systems. A sixth is the exploration of privacy-preserving deployment configurations, including on-premises VLM substitutes for the cloud teacher and differentially private mechanisms for transmitted frames.

---

## 12. Conclusion

This project has presented Falantir, a complete software system for real-time retail threat detection that combines a frontier vision–language model with a locally trained convolutional student classifier in a smart-cascade configuration. The system addresses a practical and well-recognised gap in the literature: the absence of a deployment architecture that allows the semantic richness of cloud-hosted vision–language models to be exploited under realistic cost and reliability constraints. The student model achieves a validation accuracy of zero point nine seven two nine on the held-out evaluation set; the cascade reduces Gemini token consumption by approximately seventy-three percent on representative footage; and the surrounding system delivers the persistence, alerting, real-time monitoring, and user-interface components expected of a production-quality deployment. The work establishes that the combination of a frontier vision–language model and an edge-resident student model is not merely theoretically attractive but practically deployable, and that the resulting system meaningfully exceeds both the cost-efficiency of the cloud model alone and the semantic richness of the local model alone.

---

## 13. References

[1] National Retail Federation, "National Retail Security Survey," Annual Report, Washington, DC, USA, 2024.

[2] A. G. Howard, M. Sandler, G. Chu, L. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, Seoul, Republic of Korea, 2019, pp. 1314–1324.

[3] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in *NIPS Deep Learning and Representation Learning Workshop*, Montreal, Canada, 2014.

[4] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 6479–6488.

[5] M. Hervas, "DCSASS Dataset: A Dataset of Surveillance and Anomaly Sequences," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

[6] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," Technical Report, Mountain View, CA, USA, 2024.

[7] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.

[8] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," in *Proc. Int. Conf. Pattern Recognition (ICPR)*, Cambridge, UK, 2004, vol. 2, pp. 28–31.

[9] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," *Communications of the ACM*, vol. 60, no. 6, pp. 84–90, 2017.

[10] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, Santiago, Chile, 2015, pp. 4489–4497.

[11] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, Honolulu, HI, USA, 2017, pp. 6299–6308.

[12] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017.

[13] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 39, no. 6, pp. 1137–1149, 2017.

[14] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, Vancouver, Canada, 2019, pp. 8024–8035.

[15] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

[16] M. Grinberg, *Flask Web Development*, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2018.

[17] K. Banker, P. Bakkum, S. Verch, D. Garrett, and T. Hawkins, *MongoDB in Action*, 2nd ed. Shelter Island, NY, USA: Manning Publications, 2016.

[18] React Documentation Team, "React: A JavaScript Library for Building User Interfaces," 2024. [Online]. Available: https://react.dev

[19] Twilio, Inc., "Twilio Programmable Messaging API Reference," 2024. [Online]. Available: https://www.twilio.com/docs/sms

[20] Vite Documentation Team, "Vite: Next Generation Frontend Tooling," 2024. [Online]. Available: https://vitejs.dev

---

# STEP C — PowerPoint Outline (15 Slides)

Each slide is described by Slide Title, Key Points (3–5 bullets), and Suggested Visual.

---

### Slide 1 — Title Slide

**Title**: Falantir: Hybrid Vision–Language Pipeline for Real-Time Retail Threat Detection

**Key Points**:
- Presented by: Vivek Muthe
- Department of Artificial Intelligence and Machine Learning
- [Name of Institution]
- Project Guide: [Project Guide Name]
- Academic Year 2025–2026

**Suggested Visual**: Project logo (Falantir wordmark over a stylised CCTV viewfinder); college and department crest; bottom-right footer with submission date.

---

### Slide 2 — Agenda

**Title**: Agenda

**Key Points**:
- Problem Statement and Motivation
- Existing Solutions and the Gap
- Proposed Solution and Architecture
- Technology Stack and Implementation
- Results, Testing, and Future Scope

**Suggested Visual**: Vertical numbered list with subtle accent bars; footer indicating estimated runtime ("12-minute presentation, 3 minutes Q&A").

---

### Slide 3 — Problem Statement

**Title**: The Real-World Problem

**Key Points**:
- Retail shrinkage costs the global industry tens of billions of dollars annually
- Conventional CCTV is passive and forensic; subtle theft is routinely missed
- Frontier vision–language models offer rich understanding but at prohibitive cost
- Cloud-only deployment is operationally fragile under quota or network failure
- A hybrid architecture is needed to combine semantic richness with cost efficiency

**Suggested Visual**: Split-image — left side shows a cluttered CCTV monitoring wall with a tired operator (stock photo), right side shows a stylised cost gauge with a "Gemini" needle pinned at the red zone.

---

### Slide 4 — Existing Solutions and Gaps

**Title**: Existing Solutions and Their Limitations

**Key Points**:
- Manual CCTV monitoring — high labour cost, attentional fatigue, no semantic metadata
- Motion-triggered or rule-based alerts — high false-positive rates, no behavioural understanding
- Supervised action-recognition networks — categorical output only, no narrative explanation
- Cloud VLM-only pipelines — semantically rich but economically infeasible at frame-rate
- No prior work integrates a frontier VLM with an edge fallback under explicit cost rationing

**Suggested Visual**: A four-quadrant matrix with axes "Cost Efficiency" (low to high) and "Semantic Richness" (low to high). Existing solutions are plotted in the unfavourable quadrants; Falantir is plotted in the upper-right favourable quadrant.

---

### Slide 5 — Proposed Solution Overview

**Title**: Falantir — A Smart-Cascade Hybrid Pipeline

**Key Points**:
- Three-tier vision provider chain: Gemini → MobileNetV3 → Safe Fallback
- Smart cascade: cheap student first, escalate to expensive teacher only when uncertain
- Knowledge distillation from Gemini to a fine-tuned MobileNetV3-Large student
- Complete surrounding system: persistence, alerting, real-time live monitoring
- Reduces Gemini token consumption by ~73 percent at preserved threat-detection quality

**Suggested Visual**: A horizontal flow diagram — frame goes into a green "MobileNetV3" box, branches at a decision diamond ("Safe AND confident?"), with the "Yes" branch returning the safe result and the "No" branch escalating to the orange "Gemini" box.

---

### Slide 6 — System Architecture Diagram

**Title**: System Architecture

**Key Points**:
- Presentation Layer — React + Vite single-page application
- Application Layer — Flask + Flask-SocketIO with four blueprints
- Vision Pipeline — provider chain with smart-cascade decision flow
- Persistence Layer — MongoDB with five primary collections
- External Integrations — Google Gemini API, SMTP relay, Twilio SMS

**Suggested Visual**: A multi-layer block diagram occupying the central two-thirds of the slide. Layers stacked top-to-bottom (Presentation, API, Vision Pipeline, Persistence). External services (Gemini, SMTP, Twilio) shown as separate cylinders on the right with arrows indicating data flow.

---

### Slide 7 — Technology Stack

**Title**: Technology Stack

**Key Points**:
- Backend — Python 3.14, Flask, Flask-SocketIO, PyTorch, OpenCV
- Frontend — React 18, Vite 6, Redux Toolkit, Tailwind, Framer Motion
- Database — MongoDB (PyMongo)
- AI — Google Gemini (cloud teacher) + MobileNetV3-Large (local student)
- Notifications — SMTP via Gmail relay, Twilio SMS

**Suggested Visual**: A grid of technology logos (eight to twelve) with the technology name labelled below each. Group logos by colour-coded pillar: backend (blue), frontend (green), AI (purple), infrastructure (grey).

---

### Slide 8 — Key Features

**Title**: Key Features

**Key Points**:
- Cinematic video analysis viewer with animated bounding boxes
- Real-time live-stream monitoring of network camera agents over WebSocket
- Multi-channel alerting with per-agent cooldown to prevent SMS flooding
- Smart cascade with configurable threshold for cost-vs-quality trade-off
- Operator feedback capture for future reinforcement-learning loop

**Suggested Visual**: A two-by-three feature grid with an icon plus a one-line caption per cell. Icons drawn from lucide-react (Camera, Bell, Shield, Layers, MessageSquare, RefreshCw).

---

### Slide 9 — Module Breakdown

**Title**: Module Breakdown

**Key Points**:
- Authentication module — JWT-based, bcrypt password hashing, login_required decorator
- Detection module — video upload, frame sampling, smart cascade, incident persistence
- Streaming module — per-agent daemon thread, motion gate, WebSocket emission
- Notifications module — SMTP and Twilio backends behind a unified interface
- Frontend modules — pages, services, store, monitor components

**Suggested Visual**: A package-style diagram with five rounded rectangles labelled by module. Arrows between modules showing dependency direction (e.g., Detection depends on Notifications; Streaming depends on Detection's vision pipeline).

---

### Slide 10 — Database Design

**Title**: Database Schema

**Key Points**:
- Users — accounts, hashed passwords, contact details, role
- Agents — camera definitions, source URI, owner reference, status
- Incidents — full audit trail with snapshot, threat label, reasoning, acknowledgement
- RL Feedback — operator verdicts for future student-model retraining
- Analytics — periodic aggregations for dashboard charts

**Suggested Visual**: A simplified entity-relationship diagram. Five rounded rectangles representing collections, with labelled arrows indicating reference fields (Agents → Users via owner_id; Incidents → Agents via agent_id; RL Feedback → Incidents via incident_id).

---

### Slide 11 — Implementation Highlights

**Title**: Implementation Highlights

**Key Points**:
- Smart cascade reduces token cost by ~73 percent on average evaluation footage
- MobileNetV3-Large student trained in 38 minutes on a Colab T4 GPU
- Gemini integration uses structured-output mode for parse-free JSON responses
- Frontend bounding-box overlay rendered as SVG with Framer Motion animations
- All components hot-swappable via environment variables (provider chain, thresholds)

**Suggested Visual**: Two side-by-side code panels. Left: a snippet of the smart-cascade decision in `vision_provider.py` (`if local_label == "safe" and local_conf >= threshold: return local_result`). Right: a snippet of the React VideoAnalysisViewer rendering an SVG `<motion.rect>` with the threat-coloured stroke.

---

### Slide 12 — Results and Demo

**Title**: Results

**Key Points**:
- Student model best validation accuracy: 0.9729 (achieved at epoch 23 of 30)
- Per-class F1 scores: safe 0.97, suspicious 0.88, critical 0.98
- Median latency: Gemini 2.6s, MobileNetV3 0.05s, Smart Cascade 0.20s
- Token cost reduction: ~73 percent on representative footage
- Live demo: shoplifting clip uploaded → suspicious detected → email and SMS dispatched

**Suggested Visual**: Three side-by-side panels. Left: training-curve line plot peaking at epoch 23. Centre: a screenshot of the bounding-box overlay on a real shoplifting frame. Right: a representative phone-screen mock showing the SMS alert text.

---

### Slide 13 — Testing Summary

**Title**: Testing Summary

**Key Points**:
- Manual unit checks on vision pipeline, model loader, notification subsystem
- End-to-end integration tested on 25 surveillance clips spanning all three threat tiers
- Live-stream integration tested with an IP webcam agent
- User acceptance testing with peer reviewers: cinematic UI praised; confidence visualisation requested
- Identified gap: automated unit-test coverage to be added in engineering follow-through

**Suggested Visual**: A horizontal status table — three columns (Test Type, Coverage, Status). Each row shows a green checkmark for passing tests; the final row "Automated unit tests" shows a yellow warning icon and the label "Future Work".

---

### Slide 14 — Future Scope

**Title**: Future Scope

**Key Points**:
- Closed-loop reinforcement learning — feedback collection already in place
- Migration to `google-genai` SDK to remove deprecation warning and access thinking-budget controls
- Multimodal student model that emits short scene descriptions alongside labels
- Field deployment in a partner retail location for longitudinal evaluation
- Integration with electronic-article-surveillance gates and point-of-sale systems

**Suggested Visual**: A roadmap timeline with five stations along a horizontal line, each labelled by quarter (e.g., Q3 2026, Q4 2026, etc.) with the corresponding future-work item.

---

### Slide 15 — Thank You and Q&A

**Title**: Thank You

**Key Points**:
- Project repository: [GitHub link]
- Live demo: [demo URL]
- Email: botmaticai@gmail.com
- Acknowledgement to project guide and faculty
- Open for questions

**Suggested Visual**: Large centred "Thank You" wordmark with the Falantir logo above. QR code in the lower-right corner linking to the project repository or demo URL. Subtle background reusing the cyan scanning-line motif from the live-feed viewer.

---

# STEP D — File Generation Notes

This Markdown file contains the complete report and slide outline. Two paths to produce `.docx` and `.pptx` artefacts are described below.

## Path 1 — Run the included Python script (recommended)

A companion script, `generate_submission_files.py`, is created alongside this Markdown file. It uses `python-docx` to build a Word document of the report and `python-pptx` to build a PowerPoint deck of the slide outline. Steps:

1. Install the dependencies once:
   ```
   "C:\Users\ITC\AppData\Local\Python\bin\python.exe" -m pip install python-docx python-pptx
   ```
2. Run the script from the project root:
   ```
   cd d:\vivek\Shoplifting-Detection
   "C:\Users\ITC\AppData\Local\Python\bin\python.exe" generate_submission_files.py
   ```
3. Two files appear next to this Markdown: `Falantir_Project_Report.docx` and `Falantir_Project_Slides.pptx`. Open them in Word and PowerPoint respectively.

## Path 2 — Manual conversion via Pandoc

If Pandoc is installed on your system, the Markdown can be converted directly:
```
pandoc Falantir_Submission_Package.md -o Falantir_Project_Report.docx
```
Pandoc handles tables and headings reliably, but does not generate `.pptx`. For the slide deck, copy the Step C content into PowerPoint manually using the section breaks as slide boundaries.
