import json, os
from loguru import logger
from orchestrator import app   # import the compiled LangGraph workflow
from orchestrator import GraphState

EVAL_PATH = "data/memory_store/final_report.json"
SUMMARY_PATH = "data/memory_store/agent_summaries.json"
PLAN_PATH = "data/memory_store/plan.json"

def load_json(path: str):
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_outputs(plan, summaries, final_report):
    """Check for missing expected outputs per task."""
    missing_tasks = []
    for task in plan.get("tasks", []):
        expected = set(task.get("expected_outputs", []))
        if not expected:
            continue

        task_id = task["id"]
        found_outputs = set()

        # check summaries
        for s in summaries:
            if s.get("task_id") == task_id:
                found_outputs.add("summary")

        # optionally check final report
        if final_report and task["title"] in final_report:
            found_outputs.add("final_report")

        if not expected.issubset(found_outputs):
            missing_tasks.append(task)

    return missing_tasks

def rerun_missing_tasks(plan, missing_tasks, summaries, raw_docs):
    """Re-run specific incomplete tasks."""
    if not missing_tasks:
        logger.info("✅ All expected outputs present. Nothing to re-run.")
        return

    logger.warning(f"⚠️ Missing {len(missing_tasks)} tasks, re-running them...")

    # Rebuild partial state
    inputs = {
        "plan": plan,
        "completed_tasks": set(),
        "raw_documents": raw_docs,
        "agent_summaries": summaries,
        "final_report": ""
    }

    for task in missing_tasks:
        logger.info(f"🔁 Re-running Task {task['id']} - {task['title']}")
        inputs["plan"]["tasks"] = [task]  # isolate this task
        try:
            result_state = app.invoke(inputs, {"recursion_limit": 5})
            if result_state:
                summaries.extend(result_state.get("agent_summaries", []))
                raw_docs.extend(result_state.get("raw_documents", []))
        except Exception as e:
            logger.error(f"❌ Task {task['id']} failed on retry: {e}")

    # re-save after reruns
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Updated summaries after rerun saved to {SUMMARY_PATH}")

if __name__ == "__main__":
    logger.info("🚀 Running Evaluatory Agent...")
    plan = load_json(PLAN_PATH)
    summaries = load_json(SUMMARY_PATH) or []
    final_report = load_json(EVAL_PATH) or ""
    raw_docs = load_json("data/raw_docs/raw_docs.json") or []

    if not plan:
        logger.error("❌ Plan not found. Cannot evaluate.")
    else:
        missing = evaluate_outputs(plan, summaries, final_report)
        rerun_missing_tasks(plan, missing, summaries, raw_docs)
