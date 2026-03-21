# FALANTIR v2 — Autonomous AI Security Agent System

## Project Overview

Falantir is an autonomous AI security agent designed to protect small and medium businesses from retail theft, workplace violence, and suspicious activity. Unlike traditional CCTV systems that only record events, Falantir **perceives, decides, and acts** in real time.

**Version 1** (ShopGuard) was built as a prototype with video file uploads, YOLO-based shoplifting detection, and automated alerts via Twilio and Email.

**Version 2** (Falantir) evolves it into a production-ready multi-agent system with live camera support, three AI models running in parallel, real-time WebSocket dashboards, agent analytics, reinforcement learning feedback, and a mobile-ready PWA interface.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                              │
│   [Webcam]  [RTSP Camera]  [Video File Upload]                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              THREE-MODEL PARALLEL INFERENCE                      │
│                   (Python Threading)                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │   YOLOv8     │  │  Gemini API  │  │  Custom Threat     │     │
│  │              │  │  (2.0-flash) │  │  Classifier        │     │
│  │ Object       │  │              │  │  (MobileNetV3)     │     │
│  │ Detection    │  │ Scene        │  │                    │     │
│  │              │  │ Description  │  │ Threat Level       │     │
│  │ Bounding     │  │              │  │ safe/suspicious/   │     │
│  │ boxes,       │  │ Human-       │  │ critical +         │     │
│  │ labels,      │  │ readable     │  │ confidence score   │     │
│  │ confidence   │  │ sentence     │  │                    │     │
│  └──────────────┘  └──────────────┘  └────────────────────┘     │
│                           │                                      │
│                    Combined Result                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌─────────┐ ┌──────────────┐
     │   MongoDB    │ │ WebSocket│ │ Notifications│
     │              │ │ (Live)  │ │              │
     │ • incidents  │ │         │ │ • Email      │
     │ • agents     │ │ agent_  │ │ • SMS        │
     │ • analytics  │ │ update  │ │ • Voice Call │
     │ • rl_feedback│ │         │ │ (Twilio)     │
     │ • users      │ │ incident│ │              │
     │              │ │ _alert  │ │              │
     └──────────────┘ └────┬────┘ └──────────────┘
                           │
                           ▼
     ┌─────────────────────────────────────────────┐
     │           REACT FRONTEND (PWA)              │
     │                                             │
     │  Dashboard │ Live Feed │ Analytics │ Alerts │
     │  Settings  │ Workflows │ Users     │ Profile│
     │                                             │
     │  Desktop: Sidebar navigation                │
     │  Mobile: Bottom tab bar                     │
     └─────────────────────────────────────────────┘
```

---

## How the Three AI Models Work Together

Every video frame is processed by **three models running simultaneously in parallel** using Python threading. No single model creates a bottleneck — the combined result is available as fast as the slowest of the three.

| Model | Role | Source | Output |
|-------|------|--------|--------|
| **YOLOv8** | Object Detection | Ultralytics (custom trained) | Object labels, confidence, bounding boxes |
| **Gemini API** (gemini-2.0-flash) | Scene Description | Google AI Studio | Human-readable sentence describing the scene |
| **Custom Classifier** (MobileNetV3-Large) | Threat Classification | Trained by team | safe / suspicious / critical + confidence score |

### Combined Output Per Frame:
- List of detected objects from YOLO (with bounding boxes)
- Gemini natural language description of the scene
- Threat label from the custom classifier (safe/suspicious/critical)
- Confidence score (0–1)
- Probability breakdown across all three threat levels

### Fallback Behavior:
- If the custom threat classifier is not trained yet, YOLO detections boost the threat level automatically (shoplifting detection at >70% confidence → critical, ≤70% → suspicious)
- If Gemini API is not configured, scene descriptions are skipped
- The system gracefully degrades — it always works with whatever models are available

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite + Tailwind CSS | Dashboard, analytics, PWA |
| **State Management** | Redux Toolkit | Auth, agents, detection state |
| **Real-time** | Socket.IO (client + server) | Live WebSocket updates |
| **Charts** | Recharts | Bar charts, donut charts, analytics |
| **Backend** | Flask + Flask-SocketIO | REST API, WebSocket server |
| **Database** | MongoDB (PyMongo) | Incidents, agents, analytics, feedback |
| **Object Detection** | YOLOv8 (Ultralytics) | Detect objects in video frames |
| **Scene Analysis** | Gemini API (2.0-flash) | Human-readable scene descriptions |
| **Threat Model** | PyTorch + MobileNetV3-Large | Classify threat level with confidence |
| **Training** | Google Colab (T4 GPU) | Train custom classifier |
| **Model Storage** | HuggingFace Hub | Store and version model files |
| **Notifications** | Twilio + SMTP Email | SMS, voice call, and email alerts |
| **Video Input** | OpenCV (RTSP + webcam + file) | Read camera streams and videos |

---

## Backend Structure

```
Shoplifting-Detection/
├── app.py                          # Flask + SocketIO entry point (v2)
├── run_server.py                   # Uvicorn entry point (v1 legacy)
├── requirements_v2.txt             # v2 dependencies
├── .env.example.v2                 # Environment variables template
├── .gitignore
├── FALANTIR_PROJECT_OVERVIEW.md    # This file
│
├── api/
│   ├── main.py                     # FastAPI app (v1 legacy)
│   ├── auth.py                     # v1 auth (FastAPI)
│   ├── auth_v2.py                  # v2 auth (Flask) — JWT + bcrypt
│   ├── database.py                 # v1 database (Motor async)
│   ├── database_v2.py              # v2 database (PyMongo sync) — 5 collections
│   ├── detection_engine.py         # v1 detection engine (single camera)
│   ├── notifications.py            # Email/SMS/Voice call alerts
│   ├── schemas.py                  # Pydantic models (v1)
│   ├── models.py                   # MongoDB schema docs
│   │
│   ├── models/
│   │   └── threat_classifier.py    # MobileNetV3-Large dual-head model
│   │
│   ├── services/
│   │   ├── gemini_service.py       # Gemini API scene description
│   │   ├── inference_pipeline.py   # 3-model parallel inference
│   │   └── stream_service.py       # Multi-agent RTSP stream manager
│   │
│   └── routes/
│       ├── auth_routes.py          # v1 auth routes
│       ├── auth_routes_v2.py       # v2 auth (Flask Blueprint)
│       ├── user_routes.py          # v1 user routes
│       ├── user_routes_v2.py       # v2 user routes
│       ├── detection_routes.py     # v1 detection routes
│       ├── detection_routes_v2.py  # v2 detection + upload + incidents + feedback
│       └── agent_routes.py         # v2 agent CRUD + stream control
│
├── training/
│   ├── train_threat_classifier.py  # Colab training script
│   └── dataset_setup.py           # Dataset folder structure + frame extraction
│
├── configs/
│   └── shoplifting_wights.pt       # YOLOv8 trained weights (v1)
│
└── res/
    ├── inout1.mp4                  # Sample test video
    └── input2.mp4                  # Sample test video
```

---

## Frontend Structure

```
Shoplifting-Detection-frontend-/
├── index.html                      # PWA meta tags + service worker
├── package.json                    # v2.0.0 with recharts, socket.io-client
├── vite.config.js                  # Proxy /api → localhost:8000
├── tailwind.config.js              # Custom dark theme + animations
│
├── public/
│   ├── manifest.json               # PWA manifest
│   ├── sw.js                       # Service worker
│   └── icons/                      # PWA icons
│
└── src/
    ├── api/
    │   └── apiClient.js            # Fetch client with v2 {success,data,error} handling
    │
    ├── app/
    │   ├── store.js                # Redux store (auth, user, detection, agents)
    │   └── features/
    │       ├── authSlice.js        # Login/register/logout
    │       ├── userSlice.js        # Profile, all users
    │       ├── detectionSlice.js   # Detection engine state (v1 compat)
    │       └── agentSlice.js       # Agent CRUD + live WebSocket data
    │
    ├── hooks/
    │   ├── useAuth.js              # Auth state hook
    │   ├── useSocket.js            # WebSocket hook (join/leave agent rooms)
    │   ├── useDebounce.js          # Value debounce
    │   └── useRole.js              # Role-based permissions
    │
    ├── services/
    │   ├── authService.js          # Auth API calls
    │   ├── userService.js          # User API calls
    │   ├── detectionService.js     # Detection, upload, incidents, feedback, stats
    │   └── agentService.js         # Agent CRUD, stream URLs
    │
    ├── pages/
    │   ├── auth/
    │   │   ├── Login.jsx           # Split-screen login with branding
    │   │   └── Register.jsx        # Split-screen register
    │   ├── dashboard/
    │   │   └── Dashboard.jsx       # Multi-agent cards + live threat feed
    │   ├── monitor/
    │   │   └── Monitor.jsx         # Video upload + agent live feeds
    │   ├── alerts/
    │   │   └── Alerts.jsx          # Incident history + RL feedback buttons
    │   ├── analytics/
    │   │   └── Analytics.jsx       # Recharts bar/donut + agent table
    │   ├── workflow/
    │   │   └── WorkflowBuilder.jsx # Custom alert response rules
    │   ├── settings/
    │   │   └── Settings.jsx        # Agent management + model status
    │   ├── users/
    │   │   └── Users.jsx           # User management table
    │   └── profile/
    │       └── Profile.jsx         # User profile view/edit
    │
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.jsx         # Dark sidebar (desktop)
    │   │   ├── Navbar.jsx          # Top nav with live status
    │   │   ├── BottomTabBar.jsx    # Mobile bottom tabs
    │   │   ├── SidebarData.js      # Nav items config
    │   │   └── Footer.jsx          # Footer
    │   └── ui/
    │       ├── Button.jsx          # Variants: primary/secondary/danger/ghost/outline
    │       ├── Card.jsx            # White card with glass option
    │       ├── Input.jsx           # Form input with icon + label
    │       ├── Loader.jsx          # Spinner
    │       └── Modal.jsx           # Dialog modal
    │
    ├── routes/
    │   ├── AppRoutes.jsx           # All route definitions
    │   ├── PrivateRoute.jsx        # Auth guard + layout
    │   ├── PublicRoute.jsx         # Guest-only routes
    │   └── RoleBasedRoute.jsx      # Permission check
    │
    └── utils/
        ├── constants.js            # App name, threat colors, routes
        ├── helpers.js              # Date/string formatting
        ├── localStorage.js         # Token/user persistence
        └── permissions.js          # Role checks
```

---

## API Endpoints (v2)

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login, returns JWT |
| POST | `/api/auth/logout` | Logout |

### Users
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/users/me` | Get current user profile |
| PUT | `/api/users/me` | Update profile |
| GET | `/api/users/` | List all users (admin) |
| GET | `/api/users/:id` | Get user by ID |

### Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents/` | List all camera agents |
| POST | `/api/agents/` | Register new agent |
| DELETE | `/api/agents/:id` | Delete agent |
| POST | `/api/agents/:id/start` | Start live stream |
| POST | `/api/agents/:id/stop` | Stop live stream |
| GET | `/api/agents/:id/stream` | MJPEG video stream |
| GET | `/api/agents/:id/status` | Agent stream status |

### Detection & Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/detection/models` | Status of all 3 AI models |
| POST | `/api/detection/upload` | Upload video for analysis |
| GET | `/api/detection/incidents` | Paginated incident history |
| POST | `/api/detection/incidents/:id/acknowledge` | Acknowledge incident |
| POST | `/api/detection/feedback` | Submit RL feedback |
| GET | `/api/detection/stats` | Overall analytics stats |
| GET | `/api/detection/analytics/daily` | Daily incident counts |
| POST | `/api/detection/alert/manual` | Trigger manual alert |

### WebSocket Events
| Event | Direction | Description |
|-------|-----------|-------------|
| `agent_update` | Server → Client | Frame analysis result per agent |
| `incident_alert` | Server → Client | Critical threat detected |
| `agent_status_change` | Server → Client | Agent started/stopped |
| `join_agent` | Client → Server | Subscribe to agent updates |
| `leave_agent` | Client → Server | Unsubscribe from agent |

---

## MongoDB Collections

### `users`
```json
{
  "name": "string",
  "email": "string (unique)",
  "hashed_password": "string (bcrypt)",
  "phone": "string",
  "role": "user | admin",
  "is_active": true,
  "created_at": "datetime"
}
```

### `agents`
```json
{
  "name": "string",
  "location": "string",
  "camera_uri": "string (RTSP URL or camera index)",
  "status": "stopped | streaming",
  "created_at": "datetime"
}
```

### `incidents`
```json
{
  "agent_id": "string",
  "threat_label": "safe | suspicious | critical",
  "threat_level": "0 | 1 | 2",
  "confidence": "float (0-1)",
  "gemini_description": "string",
  "yolo_objects": "[{label, class_id, confidence, bbox}]",
  "snapshot": "base64 JPEG",
  "timestamp": "datetime",
  "acknowledged": false
}
```

### `analytics`
```json
{
  "agent_id": "string",
  "date": "YYYY-MM-DD",
  "counts": { "safe": 0, "suspicious": 0, "critical": 0, "total": 0 },
  "timestamp": "datetime"
}
```

### `rl_feedback`
```json
{
  "incident_id": "string",
  "user_id": "string",
  "verdict": "correct | false_positive",
  "correct_label": "string (optional)",
  "timestamp": "datetime"
}
```

---

## Key Features

### 1. Real-Time Multi-Agent Dashboard
- Grid of camera agent cards with live video streams
- Threat level color coding (green/amber/red) updating via WebSocket
- Live threat feed showing suspicious/critical detections as they happen
- Gemini AI descriptions and YOLO detected objects per agent

### 2. Video Upload & Analysis
- Drag-and-drop video upload
- Three AI models analyze every 5th frame in parallel
- Returns peak threat frame with Gemini description
- Results saved to database automatically

### 3. Alert History with RL Feedback
- Paginated incident list with snapshots
- "Correct" and "False Positive" buttons per incident
- Feedback stored for weekly model retraining
- Manual alert trigger (sends SMS/email/call)

### 4. Analytics Dashboard
- Recharts bar chart: daily incidents (7/30/90 day views)
- Donut chart: threat level distribution
- Per-agent performance table with avg confidence

### 5. Workflow Builder
- Custom alert response rules
- Trigger conditions: critical/suspicious/any threat
- Actions: SMS, email, voice call
- Location-specific filtering

### 6. Multi-Location Agent Management
- Register agents with name, location, camera URI
- Start/stop streams independently
- Real-time status monitoring
- Auto-reconnect for RTSP streams

### 7. PWA Support
- Installable on Android/iOS from browser
- Service worker for offline support
- Responsive layout with mobile bottom tab bar

### 8. Notification System
- Email alerts via SMTP
- SMS alerts via Twilio
- Voice call alerts via Twilio
- Sent automatically on critical detections

---

## How to Run

### Backend
```bash
cd Shoplifting-Detection
pip install -r requirements_v2.txt
cp .env.example.v2 .env    # Fill in your API keys
python app.py               # http://localhost:8000
```

### Frontend
```bash
cd Shoplifting-Detection-frontend-
npm install
npm run dev                 # http://localhost:5173
```

### Environment Variables Required
- `GEMINI_API_KEY` — Google Gemini API key
- `MONGO_URI` — MongoDB connection string
- `FLASK_SECRET_KEY` — JWT signing key
- `TWILIO_*` — Twilio credentials (optional, for SMS/calls)
- `SMTP_*` — Email credentials (optional, for email alerts)
- `HUGGINGFACE_*` — HuggingFace Hub (optional, for custom model)

---

## Development Phases

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| **1 — Custom Model** | Months 1–3 | Architecture done, training pending | MobileNetV3 classifier, Colab notebook, HuggingFace integration |
| **2 — Backend** | Months 3–5 | Complete | Flask API, WebSocket, 3-model pipeline, MongoDB, notifications |
| **3 — Frontend** | Months 5–8 | Complete | Dashboard, live feed, analytics, alerts, settings, PWA |
| **4 — Analytics & RL** | Months 8–10 | Feedback UI done, retraining pending | RL feedback loop, extended charts |
| **5 — Polish & Demo** | Months 10–12 | Pending | Multi-agent demo, load testing, final presentation |

---

## Team

| Role | Responsibilities |
|------|-----------------|
| AI / Model Engineer | Dataset, model architecture, training, HuggingFace upload |
| Backend Engineer | Flask API, WebSocket, inference pipeline, MongoDB, Twilio |
| Frontend Engineer | React dashboard, WebSocket integration, Recharts, PWA |
| Integration & DevOps | System integration, multi-agent demo, documentation |

---

*Falantir v2 — Autonomous AI Security Agent — Final Year AI Project*
