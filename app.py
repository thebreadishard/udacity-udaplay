"""UdaPlay web chat UI.

A small Streamlit front end for the UdaPlay research agent. It mirrors the tools
and agent defined in Udaplay_02_solution_project.ipynb (the graded notebook stays
the source of truth) and adds a BPMN-style process-flow visualization so you can
see, per question, whether the agent:

  1. queried the internal vector DB (retrieve_game),
  2. judged the results sufficient or not (evaluate_retrieval), and
  3. fell back to a web search (game_web_search).

Run it with:
    streamlit run app.py

Prerequisite: run Udaplay_01_solution_project.ipynb once so the chromadb/ vector
database exists.
"""

# --- Optional SQLite compatibility (same shim as the notebooks) -------------
from importlib import import_module, util
import sys

if util.find_spec("pysqlite3") is not None:
    sys.modules["sqlite3"] = import_module("pysqlite3")

import json
import os
import re
import time
import uuid

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from lib.agents import Agent
from lib.llm import LLM
from lib.messages import SystemMessage, UserMessage
from lib.parsers import PydanticOutputParser
from lib.tooling import tool


# --- Live progress tracker --------------------------------------------------
class FlowTracker:
    """Tracks which step of the agent flow is running so the UI can update live.

    The tools (which run synchronously inside Agent.invoke) call into this
    tracker as they execute. The active turn registers a callback that redraws
    the process-flow diagram every time the state changes.
    """

    def __init__(self):
        self.callback = None
        # Minimum time (seconds) to hold each step's highlight. Streamlit
        # coalesces rapid writes to the same placeholder, so without a small
        # dwell the fast intermediate steps would be flushed away before the
        # browser ever paints them.
        self.dwell = 0.6
        self.reset()

    def reset(self):
        self.current = None
        self.did_retrieve = False
        self.did_evaluate = False
        self.did_web = False
        self.eval_useful = None
        self.done = False

    def state(self) -> dict:
        return {
            "current": self.current,
            "did_retrieve": self.did_retrieve,
            "did_evaluate": self.did_evaluate,
            "did_web": self.did_web,
            "eval_useful": self.eval_useful,
            "done": self.done,
        }

    def _render(self):
        if self.callback:
            self.callback(self.state())

    def begin(self):
        """Start a new turn at the entry point."""
        self.reset()
        self.current = "start"
        self._render()

    def mark(self, step: str):
        """A tool just started running."""
        self.current = step
        if step == "retrieve":
            self.did_retrieve = True
        elif step == "gateway":
            self.did_evaluate = True
        elif step == "websearch":
            self.did_web = True
        self._render()
        # Hold the highlight briefly so the browser paints this frame before
        # the (often very fast) tool work moves us on to the next state.
        if self.callback:
            time.sleep(self.dwell)

    def thinking(self):
        """A tool finished; the LLM is deciding what to do next.

        We intentionally keep ``current`` pointing at the step that just ran so
        its node stays highlighted (orange) through the LLM gap until the next
        step takes over. Clearing it here would make the highlight flicker for
        only a few hundred milliseconds and the client-side Graphviz renderer
        would usually coalesce it away before painting.
        """
        self._render()

    def set_eval(self, useful: bool):
        """Record the gateway decision (documents sufficient or not).

        Keeps the gateway node highlighted until the next step takes over.
        """
        self.eval_useful = useful
        self._render()

    def finish(self):
        """The turn is complete and the answer is ready."""
        self.current = "answer"
        self.done = True
        self._render()


TRACKER = FlowTracker()


# --- One-time setup (cached so it runs once per server process) -------------
@st.cache_resource(show_spinner="Starting UdaPlay agent...")
def build_agent() -> Agent:
    """Wire up the vector DB, tools, and agent. Mirrors the Part 02 notebook."""
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

    if not openai_api_key or not tavily_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY and TAVILY_API_KEY must be set in your .env file."
        )

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        api_base=openai_base_url,
    )
    chroma_client = chromadb.PersistentClient(path="chromadb")
    collection = chroma_client.get_collection(
        name="udaplay",
        embedding_function=embedding_fn,
    )
    tavily_client = TavilyClient(api_key=tavily_api_key)

    @tool
    def retrieve_game(query: str):
        """Semantic search: Finds most similar results in the vector DB

        args:
        - query: a question about game industry.
        """
        TRACKER.mark("retrieve")
        results = collection.query(query_texts=[query], n_results=5)
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        games = [
            {
                "Platform": metadata.get("Platform", "Unknown"),
                "Name": metadata.get("Name", "Unknown"),
                "YearOfRelease": metadata.get("YearOfRelease", "Unknown"),
                "Description": document,
            }
            for metadata, document in zip(metadatas, documents)
        ]
        TRACKER.thinking()
        return games

    class EvaluationReport(BaseModel):
        useful: bool = Field(
            description="Whether the retrieved documents are enough to answer the question"
        )
        description: str = Field(description="Detailed explanation of the evaluation result")

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: list) -> EvaluationReport:
        """Based on the user's question and on the list of retrieved documents,
        it will analyze the usability of the documents to respond to that question.

        args:
        - question: original question from user
        - retrieved_docs: retrieved documents most similar to the user query in the Vector Database
        """
        TRACKER.mark("gateway")
        llm = LLM(model="gpt-4o-mini", temperature=0.0)
        system_message = SystemMessage(
            content=(
                "Your task is to evaluate if the documents are enough to respond the query. "
                "Give a detailed explanation, so it's possible to take an action to accept it or not."
            )
        )
        user_message = UserMessage(
            content=f"Question: {question}\nRetrieved documents: {retrieved_docs}"
        )
        ai_message = llm.invoke(
            input=[system_message, user_message],
            response_format=EvaluationReport,
        )
        parser = PydanticOutputParser(model_class=EvaluationReport)
        report = parser.parse(ai_message)
        TRACKER.set_eval(report.useful)
        return report

    @tool
    def game_web_search(question: str):
        """Search the web for information about the game industry.

        args:
        - question: a question about game industry.
        """
        TRACKER.mark("websearch")
        search_results = tavily_client.search(
            question,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        results = [
            {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
            }
            for result in search_results.get("results", [])
        ]
        TRACKER.thinking()
        return {"answer": search_results.get("answer"), "results": results}

    return Agent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        tools=[retrieve_game, evaluate_retrieval, game_web_search],
        instructions=(
            "You are UdaPlay, a helpful research assistant for the video game industry. "
            "First use retrieve_game to search the internal vector database, then use "
            "evaluate_retrieval to check whether the results can answer the question. "
            "If they are not enough, use game_web_search to look the answer up on the web. "
            "Base your answers on the information returned by the tools and cite the game "
            "details (name, platform, year) when relevant."
        ),
    )


# --- Inspecting what the agent did in a single turn -------------------------
def extract_turn(run, query: str):
    """Pull the tool calls and evaluation decision out of the latest turn."""
    state = run.get_final_state() or {}
    messages = state.get("messages", [])

    # Everything after the last user message belongs to this turn.
    user_indices = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "user"]
    start = user_indices[-1] if user_indices else 0
    turn_messages = messages[start:]

    steps = []
    tool_names = []
    eval_useful = None

    for message in turn_messages:
        role = getattr(message, "role", None)
        if role == "assistant" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_names.append(name)
                steps.append({"kind": "call", "name": name, "args": args})
        elif role == "tool":
            name = getattr(message, "name", "")
            content = getattr(message, "content", "")
            if name == "evaluate_retrieval":
                if re.search(r'useful["\s:=]+true', content, re.IGNORECASE):
                    eval_useful = True
                elif re.search(r'useful["\s:=]+false', content, re.IGNORECASE):
                    eval_useful = False
            steps.append({"kind": "result", "name": name, "content": content})

    final_answer = ""
    for message in reversed(turn_messages):
        if getattr(message, "role", None) == "assistant" and getattr(message, "content", None):
            final_answer = message.content
            break

    return {
        "steps": steps,
        "tool_names": tool_names,
        "eval_useful": eval_useful,
        "answer": final_answer,
    }


# --- BPMN-style process flow (rendered client-side by st.graphviz_chart) -----
def build_flow_dot(state: dict) -> str:
    """Build a Graphviz DOT diagram of the agent flow.

    Coloring: green = completed step, orange = step currently running,
    gray = step not (yet) used.
    """
    current = state.get("current")
    done = state.get("done", False)
    did_retrieve = state.get("did_retrieve", False)
    did_evaluate = state.get("did_evaluate", False)
    did_web = state.get("did_web", False)
    eval_useful = state.get("eval_useful")

    GREEN_F, GREEN_L = "#c8e6c9", "#2e7d32"
    ORANGE_F, ORANGE_L = "#ffe0b2", "#e65100"
    IDLE_F, IDLE_L = "#f5f5f5", "#bdbdbd"

    def colors(node_id, visited):
        if current == node_id and not done:
            return ORANGE_F, ORANGE_L
        if visited:
            return GREEN_F, GREEN_L
        return IDLE_F, IDLE_L

    def node(node_id, label, shape, visited, rounded=False, extra=""):
        fill, line = colors(node_id, visited)
        style = "filled,rounded" if rounded else "filled"
        return (
            f'  {node_id} [label="{label}", shape={shape}, '
            f'style="{style}", fillcolor="{fill}", color="{line}", penwidth=2{extra}];'
        )

    def edge(src, dst, active, to_current=False, label=""):
        if to_current and not done:
            color, width = ORANGE_L, 2.5
        elif active:
            color, width = GREEN_L, 2.5
        else:
            color, width = IDLE_L, 1.0
        lbl = f', label="{label}", fontcolor="{color}"' if label else ""
        return f'  {src} -> {dst} [color="{color}", penwidth={width}{lbl}];'

    answered_from_db = done and did_evaluate and eval_useful is True and not did_web
    answered_directly = done and not did_retrieve

    lines = [
        "digraph UdaPlay {",
        "  rankdir=LR;",
        '  bgcolor="transparent";',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10];',
        node("start", "Start", "circle", True, extra=", width=0.5, fixedsize=true"),
        node("retrieve", "Query\\nVector DB", "box", did_retrieve, rounded=True),
        node("gateway", "Info\\nsufficient?", "diamond", did_evaluate),
        node("websearch", "Search\\nWeb", "box", did_web, rounded=True),
        node("answer", "Answer", "doublecircle", done, extra=", width=0.5, fixedsize=true"),
        edge("start", "retrieve", did_retrieve, to_current=(current == "retrieve")),
        edge("retrieve", "gateway", did_evaluate, to_current=(current == "gateway")),
        edge("gateway", "answer", answered_from_db, label="yes"),
        edge("gateway", "websearch", did_web, to_current=(current == "websearch"), label="no"),
        edge("websearch", "answer", done and did_web),
        edge("start", "answer", answered_directly, label="direct"),
        "}",
    ]
    return "\n".join(lines)


def turn_to_state(turn: dict) -> dict:
    """Convert a finished turn into the state dict build_flow_dot expects."""
    return {
        "current": None,
        "done": True,
        "did_retrieve": "retrieve_game" in turn["tool_names"],
        "did_evaluate": "evaluate_retrieval" in turn["tool_names"],
        "did_web": "game_web_search" in turn["tool_names"],
        "eval_useful": turn["eval_useful"],
    }


# --- Page ------------------------------------------------------------------
st.set_page_config(page_title="UdaPlay", page_icon="🎮", layout="centered")
st.title("🎮 UdaPlay")
st.caption("A research assistant for the video game industry — ask about games, release dates, platforms, and more.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "Each answer follows this flow:\n\n"
        "1. **Query Vector DB** — semantic search over the local game database.\n"
        "2. **Info sufficient?** — an LLM judges whether those results answer your question.\n"
        "3. **Search Web** — if not, it falls back to a live web search.\n\n"
        "**Orange** = step running now &nbsp;·&nbsp; **green** = completed &nbsp;·&nbsp; gray = not used.\n\n"
        "The *direct* Start→Answer path is taken only if the agent decides it can "
        "answer without any tools."
    )
    if st.button("Clear chat"):
        agent = build_agent()
        agent.reset_session(st.session_state.get("session_id"))
        st.session_state.history = []
        st.rerun()

try:
    agent = build_agent()
except Exception as exc:  # surface setup problems clearly in the UI
    st.error(f"Could not start the agent: {exc}")
    st.info("Make sure your .env is set and you've run Udaplay_01_solution_project.ipynb to build the vector DB.")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []


def render_steps(turn: dict):
    """Render the expandable list of tool calls and the gateway verdict."""
    with st.expander("Agent steps"):
        if not turn["steps"]:
            st.write("Answered directly, without using any tools.")
        for step in turn["steps"]:
            if step["kind"] == "call":
                args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items() if k != "retrieved_docs")
                st.markdown(f"- 🔧 **{step['name']}**({args})")
            elif step["name"] == "evaluate_retrieval":
                verdict = {True: "✅ sufficient", False: "❌ not enough"}.get(turn["eval_useful"], "—")
                st.markdown(f"  - verdict: {verdict}")


# Replay history
for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        if entry.get("turn"):
            # Flow first, then the answer, mirroring the live layout.
            st.graphviz_chart(build_flow_dot(turn_to_state(entry["turn"])))
            st.markdown(entry["content"])
            render_steps(entry["turn"])
        else:
            st.markdown(entry["content"])

# New question
if prompt := st.chat_input("Ask UdaPlay about a game..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Order matters: the status line and process flow render above the
        # answer, so the diagram sits between the question and the answer and
        # is clearly separate from the answer text. The answer is only filled
        # in once the flow has completed.
        status_placeholder = st.empty()
        flow_placeholder = st.empty()
        answer_placeholder = st.empty()
        steps_placeholder = st.empty()

        # Human-readable label for the step currently running
        status_labels = {
            "start": "🟠 Starting…",
            "retrieve": "🔎 Querying the vector database…",
            "gateway": "⚖️ Checking whether the results are enough…",
            "websearch": "🌐 Searching the web…",
        }

        def render_live(state):
            """Update the status line and flow diagram as the agent progresses."""
            if state.get("done"):
                status_placeholder.empty()
            else:
                status_placeholder.markdown(status_labels.get(state.get("current"), "💭 Thinking…"))
            flow_placeholder.graphviz_chart(build_flow_dot(state))

        TRACKER.callback = render_live
        TRACKER.begin()

        run = agent.invoke(prompt, session_id=st.session_state.session_id)
        turn = extract_turn(run, prompt)

        TRACKER.finish()
        TRACKER.callback = None

        answer_placeholder.markdown(turn["answer"] or "_(no answer)_")
        flow_placeholder.graphviz_chart(build_flow_dot(turn_to_state(turn)))
        with steps_placeholder.container():
            render_steps(turn)

    st.session_state.history.append(
        {"role": "assistant", "content": turn["answer"] or "_(no answer)_", "turn": turn}
    )
