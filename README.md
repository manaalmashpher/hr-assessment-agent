# Conversational SHL Assessment Recommender

This repository contains a conversational agent designed to recommend SHL Individual Test Solutions based on natural language dialogue. The agent helps hiring managers navigate the SHL product catalog by clarifying vague intents, honoring constraints, and providing grounded, hallucination-free recommendations.

## Features

- **Stateless API**: A FastAPI-powered `POST /chat` endpoint that processes the entire conversation history in a single request.
- **Hybrid Retrieval System**: Combines FAISS semantic search for vague capability queries with Regex-based explicit extraction for direct product comparisons.
- **Fast and Reliable LLM Generation**: Uses Groq (Llama-3.3-70b-versatile) for rapid inference, enforcing strict JSON output schemas.
- **Strict Scope Guarding**: Politely refuses off-topic, legal, or prompt-injection attempts and immediately redirects to assessment selection.
- **Graceful Rate-Limit Handling**: Bypasses internal SDK sleep delays to enforce a strict sub-30-second response time under all load conditions.

## Prerequisites

- Python 3.12+
- A valid Groq API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service exposes two endpoints:
- `GET /health` : Health check for readiness probes.
- `POST /chat` : Stateless conversational endpoint.

## Testing

A local replay harness is provided to simulate the automated evaluator. Ensure the Uvicorn server is running locally, then execute:

```bash
python test_local.py
```

This will run through 8 distinct conversational traces (e.g., Senior Leadership, vague queries, prompt injections) and assert strict schema compliance, URL validity, and conversational state transitions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
