"""Terminal chat:  python -m chatbot.cli [--llm]"""
import sys

from .engine import LoanChatbot
from .llm import LLMClient


def main():
    bot = LoanChatbot(llm=LLMClient() if "--llm" in sys.argv else None, channel="cli")
    msgs = bot.start()
    while True:
        for m in msgs:
            print("\nBOT :", m["text"])
            if m.get("options"):
                print("      [" + " | ".join(f"{i + 1}. {o}" for i, o in enumerate(m["options"])) + "]")
        if bot.stage == "done":
            break
        try:
            msgs = bot.handle(input("YOU : "))
        except (EOFError, KeyboardInterrupt):
            break
    if bot.result:
        print(f"\n[session {bot.session_id} | decision latency {bot.result['decision_latency_ms']:.0f} ms]")


if __name__ == "__main__":
    main()
