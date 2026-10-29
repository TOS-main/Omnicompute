"""Saved conversations on the main device, so any controller can reopen an earlier chat.

Kept in chats.json next to the models folder. The open chat is the hub's shared
conversation; "New chat" keeps it in the list and starts an empty one.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_CHATS = 500


class ChatHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.chats: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.chats = {c["id"]: c for c in data.get("chats", []) if isinstance(c, dict) and c.get("id")}
        except (OSError, ValueError, KeyError):
            pass

    @staticmethod
    def title_of(messages: list[dict]) -> str:
        first = next((m.get("content", "") for m in messages if m.get("role") == "user" and m.get("content")), "")
        first = " ".join(first.split())
        return (first[:60] + "…") if len(first) > 60 else first or "New chat"

    def keep(self, chat_id: str, messages: list[dict]) -> None:
        """Save the open chat's messages (empty chats are not kept)."""
        if not messages:
            return
        now = time.time()
        c = self.chats.setdefault(chat_id, {"id": chat_id, "created": now, "title": ""})
        c["messages"] = [{k: v for k, v in m.items() if k != "live"} for m in messages]
        c["updated"] = now
        if not c.get("renamed"):
            c["title"] = self.title_of(messages)
        if len(self.chats) > MAX_CHATS:  # drop the oldest
            for old in sorted(self.chats.values(), key=lambda x: x["updated"])[: len(self.chats) - MAX_CHATS]:
                self.chats.pop(old["id"], None)
        self.save()

    def listing(self, q: str = "") -> list[dict]:
        """Every chat without its messages; q keeps the ones whose title or text contains it."""
        q = q.strip().lower()
        hit = lambda c: not q or q in c["title"].lower() or any(q in m.get("content", "").lower() for m in c.get("messages", []))  # noqa: E731
        return [{"id": c["id"], "title": c["title"], "created": c["created"], "updated": c["updated"],
                 "count": len(c.get("messages", []))} for c in self.chats.values() if hit(c)]

    def messages(self, chat_id: str) -> list[dict]:
        return list(self.chats[chat_id].get("messages", []))

    def rename(self, chat_id: str, title: str) -> None:
        c = self.chats[chat_id]
        c["title"], c["renamed"] = " ".join(title.split())[:80] or c["title"], True
        self.save()

    def delete(self, chat_id: str) -> None:
        self.chats.pop(chat_id, None)
        self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"chats": list(self.chats.values())}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # a read-only disk only loses the history
