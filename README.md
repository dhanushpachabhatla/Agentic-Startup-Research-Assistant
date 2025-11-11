# 🚀 Agentic Startup Research Assistant

An **AI-powered multi-agent research system** that autonomously performs market analysis, technical research, and insight summarization for startup ideas.  
It uses **LangGraph**, **LangChain**, and **Gemini 2.5 Flash** to orchestrate intelligent agents that plan, research, summarize, and build a final report.

---

## 🧠 Key Features

- 🧩 **Dynamic Orchestrator Pipeline** — executes multi-step plans via interconnected agents.  
- 🤖 **Research Agents** — specialized agents like `CompetitorScout`, `TechPaperMiner`, and `TrendScraper`.  
- 📄 **Automated Summarization** — powered by Google Gemini (via `langchain_google_genai`).  
- 🧱 **Modular Architecture** — plug in new agents, retrievers, or summarizers easily.  
- 💾 **Persistent Storage** — saves outputs such as raw documents and agent summaries.  
- ⚙️ **Configurable Planner** — dynamically generates and executes a research plan.

---

## ⚙️ Prerequisites

- Python **3.10+**
- A **Google Gemini API key**
- (Optional) Git installed for version control

---

## 🧩 Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Agentic-Startup-Research-Assistant.git
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
```bash
#reate a .env file in the root directory and add:
GOOGLE_API_KEY1="your_google_gemini_api_key_1"
GOOGLE_API_KEY2="your_google_gemini_api_key_2"
..
GOOGLE_API_KEY6="your_google_gemini_api_key_6"
```

### 5. Running the Project
```bash
#run the pipeline
python -m core.pipeline
```

### 5.Outputs
After successful execution, the pipeline saves:
| File                                     | Description                               |
| ---------------------------------------- | ----------------------------------------- |
| `data/memory_store/agent_summaries.json` | Summaries from each agent                 |
| `data/raw_docs/raw_docs.json`            | Raw retrieved research data               |
| `final_report.md`                        | Combined summary or report (if generated) |








