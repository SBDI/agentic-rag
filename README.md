# Agentic RAG Agent

**Agentic RAG Agent** is a chat application that combines models with retrieval-augmented generation.
It allows users to ask questions based on custom knowledge bases, documents, and web data, retrieve context-aware answers, and maintain chat history across sessions.

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
uv venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```shell
uv pip install -r requirements.txt
```

### 3. Configure API Keys

Required:
```bash
export GROQ_API_KEY=your_groq_key_here
```

Optional (for additional models):
```bash
export OPENAI_API_KEY=your_openai_key_here
export ANTHROPIC_API_KEY=your_anthropic_key_here
export GOOGLE_API_KEY=your_google_key_here
```

### 4. Run PgVector

> Install [docker desktop](https://docs.docker.com/desktop/install/mac-install/) first.

- Run using a helper script

```shell
run_pgvector.sh
```

- OR run using the docker run command

```shell
docker run -d \
  -e POSTGRES_DB=ai \
  -e POSTGRES_USER=ai \
  -e POSTGRES_PASSWORD=ai \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5532:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

### 5. Run Agentic RAG App

```shell
streamlit run app.py
```

## 🔧 Customization

### Model Selection

The application supports multiple model providers:
- Groq (meta-llama/llama-4-scout-17b-16e-instruct, llama-3.3-70b-versatile) - Default
- OpenAI (o3-mini, gpt-4o)
- Anthropic (claude-3-5-sonnet)
- Google (gemini-2.0-flash-exp)

### How to Use
- Open [localhost:8501](http://localhost:8501) in your browser.
- Upload documents or provide URLs (websites, csv, txt, and PDFs) to build a knowledge base.
- Enter questions in the chat interface and get context-aware answers.
- The app can also answer question using duckduckgo search without any external documents added.

### Troubleshooting
- **Docker Connection Refused**: Ensure `pgvector`  containers are running (`docker ps`).
- **Groq API Errors**: Verify that the `GROQ_API_KEY` is set and valid.
- **OpenAI API Errors**: Verify that the `OPENAI_API_KEY` is set and valid.

## 📚 Documentation

For more detailed information:
- [Agno Documentation](https://docs.agno.com)
- [Streamlit Documentation](https://docs.streamlit.io)



