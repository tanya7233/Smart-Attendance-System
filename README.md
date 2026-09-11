<p align="center">
  <h1 align="center">📹 Smart Attendance System</h1>
  <p align="center">
    <b>Real-Time Face Recognition · Anti-Spoofing Liveness Detection · AI-Powered Chatbot</b>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.53+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini"></a>
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  </p>
</p>

---

An end-to-end, AI-powered attendance solution that leverages **OpenCV** for real-time facial recognition and **anti-spoofing liveness detection**. The system automates the check-in process through a sleek Streamlit dashboard and features an integrated Gemini-powered chatbot for natural language record querying.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Advanced Face Recognition** | High-accuracy matching using **LBPH** and **DNN-based embeddings** |
| 🛡️ **Anti-Spoofing Liveness** | Blink-frequency analysis to ensure a live person, not a photo |
| 🤖 **AI Attendance Assistant** | **RAG chatbot** powered by Gemini 2.5 Flash + FAISS for natural language queries |
| 📊 **Automated Analytics** | Daily/monthly reports with visualizations, exportable to **PDF, Excel, and CSV** |
| ⚙️ **Live Enrollment** | Register students directly via camera with automated dataset training |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Computer Vision** | OpenCV, DeepFace, Imutils |
| **Frontend** | Streamlit, Plotly |
| **LLM** | Google Gemini 2.5 Flash |
| **Database & Vector Search** | SQLite3, FAISS |
| **Data Processing** | Pandas, NumPy, Scikit-Learn |

---

## 📂 Project Structure
```
📦 Smart-Attendance-System
│
├── main.py                      # Entry point — initializes dirs & launches app
├── dashboard.py                 # Streamlit UI (live feed, reports, controls)
├── face_recognition_system.py   # Core OpenCV + liveness detection logic
├── chatbot.py                   # Gemini + FAISS RAG chatbot
├── database.py                  # SQLite handling & report generation
├── embedding_pipeline.py        # Face embeddings + ML (SVM) training
├── requirements.txt             # Python dependencies
│
└── data/                        # Auto-generated at runtime
    ├── attendance.db            # SQLite attendance database
    ├── models/                  # Trained LBPH & SVM models
    └── embeddings/              # Facial embedding vectors
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- A webcam
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository
```bash
git clone https://github.com/Zerodistraction-max/Smart-Attendance-with-Face-Liveness-Detection.git
cd Smart-Attendance-with-Face-Liveness-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory and add your credentials:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
ATTENDANCE_DB_PATH=data/attendance.db
```

### 4. Run the Application
```bash
python main.py
```

> The system will automatically initialize all required directories and launch the Streamlit dashboard in your browser.

---

## 🖥️ How It Works
```
┌──────────────┐     ┌────────────────────────┐     ┌──────────────────┐
│  Live Camera │────▶│  Liveness Detection    │────▶│ Face Recognition │
│     Feed     │     │  (Blink Frequency)     │     │  (LBPH / DNN)    │
└──────────────┘     └────────────────────────┘     └────────┬─────────┘
                                                             │
                              ┌──────────────────────────────▼──────┐
                              │          SQLite Database             │
                              │  (Attendance Logs + Student Records) │
                              └──────────────────────────────────────┘
                                             │
                    ┌────────────────────────┴───────────────────────┐
                    │                                                 │
           ┌────────▼────────┐                          ┌────────────▼──────────┐
           │  Analytics &    │                          │   Gemini RAG Chatbot  │
           │  Report Export  │                          │  (FAISS + Gemini 2.5) │
           │  (PDF/Excel/CSV)│                          └───────────────────────┘
           └─────────────────┘
```

1. **Live Feed** — The camera captures a real-time video stream.
2. **Liveness Check** — Blink detection confirms the subject is a live person.
3. **Recognition** — LBPH or DNN embeddings identify the student.
4. **Logging** — A timestamped record is saved to SQLite.
5. **Analytics** — Reports are auto-generated and available for export.
6. **Chatbot** — Ask questions in plain English like *"Who was absent on Monday?"*

---

## 🤖 AI Chatbot Examples

The Gemini-powered RAG assistant can answer natural language queries over your attendance database:

> 💬 *"Was Rahul present yesterday?"*
> 💬 *"How many students were absent last week?"*
> 💬 *"Show me the attendance summary for March."*
> 💬 *"Who has the lowest attendance this month?"*

---

## 📊 Reports & Analytics

- **Daily Summary** — Per-day attendance breakdown
- **Monthly Report** — Aggregated monthly statistics with charts
- **Export Formats** — PDF, Excel (`.xlsx`), and CSV
- **Visualizations** — Bar charts and trend graphs via Plotly

---

## 🔧 Configuration

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Your Gemini API key | *(required)* |
| `ATTENDANCE_DB_PATH` | Path to SQLite database | `data/attendance.db` |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m 'Add some feature'`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

Feel free to open an [issue](https://github.com/Zerodistraction-max/Smart-Attendance-with-Face-Liveness-Detection/issues) for bug reports, feature requests, or questions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Zerodistraction-max">Zerodistraction-max</a>
</p>
