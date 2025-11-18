# 🚀 Agentic Startup Research Assistant

An **autonomous AI research companion** that transforms raw startup ideas into actionable insights.  
This system leverages **multi-agent orchestration**, **retrieval-augmented generation (RAG)**, and **strategic reasoning** to perform comprehensive **market, technical, and trend research** — all autonomously.

Built with **LangGraph**, **LangChain**, and **Gemini 2.5 Flash**, it dynamically coordinates intelligent agents to plan, research, summarize, and build structured reports.

---

## 🧩 Core Capabilities

### 🔍 1. Intent Understanding
Parses your startup idea or research question to extract domains, goals, and key research triggers.

### 🧠 2. Dynamic Task Planning
Automatically generates task blueprints and assigns them to specialized research agents.

### 🤖 3. Autonomous Multi-Agent Research
Uses domain-specific agents to gather, analyze, and summarize insights from sources such as **arXiv**, **market reports**, and **news data**.

### 📚 4. Retrieval-Augmented Generation (RAG)
Builds and queries a local vector store to retrieve the most relevant research chunks for contextual reasoning.

### 💡 5. Strategy and Insight Generation
Synthesizes findings into strategic recommendations, market positioning insights, and actionable suggestions.

### 🗂️ 6. Report Builder
Generates structured markdown and JSON reports summarizing research, strategies, and sources.

### 💬 7. Memory Chatbot Assistant
Includes an **interactive memory-based chatbot** powered by vector search + conversation memory.  
It allows you to:
- Continue discussions across multiple questions without losing context  
- Retrieve information from past sessions  
- Explore research results in a conversational manner — **GPT-style chat interface included**

---

## ⚙️ Prerequisites

- Python **3.10+**
- A **Google Gemini API key**
- (Optional) Git installed for version control

---

## 🧩 Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/dhanushpachabhatla/Agentic-Startup-Research-Assistant.git
cd Agentic-Startup-Research-Assistant
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
### 3. Install dependencies
Add your API key to .env
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
visit main/config.py to check used apis in this project, make sure to have them before you run anything
```bash
#reate a .env file in the root directory and add:
GOOGLE_API_KEY1="your_google_gemini_api_key_1"
GOOGLE_API_KEY2="your_google_gemini_api_key_2"
..
GOOGLE_API_KEY10="your_google_gemini_api_key_10"
TAVILY_API_KEY
NEWS_API_KEY
...
```

### 5. Running the Pipeline
```bash
#run the pipeline
python -m core.pipeline
```

After successful execution, the pipeline saves:
| File                                     | Description                               |
| ---------------------------------------- | ----------------------------------------- |
| `data/memory_store/agent_summaries.json` | Summaries from each agent                 |
| `data/raw_docs/raw_docs.json`            | Raw retrieved research data               |
| `final_report.md`                        | Combined summary or report (if generated) |

or 

### 6. Running the Project (with frontend)
```bash
streamlit run frontend/app_frontend.py
```







