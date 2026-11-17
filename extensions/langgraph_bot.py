"""Going further (course sessions 17-18): the same chatbot driven by a LangGraph graph.

    pip install langgraph
    python -m extensions.langgraph_bot

The graph has one node. It shows the bot's current message, pauses with interrupt() until a human answers,
and hands the answer to the engine. A checkpointer and a thread_id let the graph resume where it paused.
The engine still validates every answer and still makes the decision; LangGraph only orchestrates the turns.

Limit of this demo: the engine object lives in a process-level registry, not in the checkpointed state, so a
conversation survives a pause but not a process restart. A production version would keep the slot values in
the graph state and rebuild the engine from them.
"""
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from chatbot.engine import LoanChatbot

BOTS: dict[str, LoanChatbot] = {}          # thread_id -> engine


class State(TypedDict, total=False):
    messages: list          # what the bot last said (shown to the human at the pause)
    done: bool
    decision: str | None


def turn(state: State, config) -> State:
    tid = config["configurable"]["thread_id"]
    if tid not in BOTS:                                   # first visit: open the session exactly once
        BOTS[tid] = LoanChatbot(channel="langgraph")
        shown = BOTS[tid].start()
    else:
        shown = state.get("messages", [])
    bot = BOTS[tid]
    answer = interrupt({"messages": shown})               # <- human-in-the-loop pause; resumes with the answer
    out = bot.handle(answer)                              # rules validate; the model decides
    return {"messages": out, "done": bot.stage == "done", "decision": (bot.result or {}).get("decision")}


def build_graph():
    g = StateGraph(State)
    g.add_node("turn", turn)
    g.add_edge(START, "turn")
    g.add_conditional_edges("turn", lambda s: END if s.get("done") else "turn")
    return g.compile(checkpointer=InMemorySaver())        # interrupt() requires a checkpointer


def run_conversation(answers: list[str], thread_id: str = "demo-1") -> dict:
    """Feed a list of answers through the graph; returns the final state."""
    graph, cfg = build_graph(), {"configurable": {"thread_id": thread_id}}
    out = graph.invoke({}, cfg)                           # runs to the first interrupt
    for a in answers:
        if "__interrupt__" not in out:
            break
        out = graph.invoke(Command(resume=a), cfg)
    return out


if __name__ == "__main__":
    graph, cfg = build_graph(), {"configurable": {"thread_id": "cli"}}
    out = graph.invoke({}, cfg)
    while "__interrupt__" in out:
        for m in out["__interrupt__"][0].value["messages"]:
            print("\nBOT :", m["text"], ("  " + str(m["options"])) if m.get("options") else "")
        out = graph.invoke(Command(resume=input("YOU : ")), cfg)
    for m in out["messages"]:
        print("\nBOT :", m["text"])
