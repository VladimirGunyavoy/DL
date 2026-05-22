#!/usr/bin/env python3
"""
get_context.py — роутинг контекста по типу задачи.

Использование:
    python llm/tools/get_context.py bug
    python llm/tools/get_context.py feature
    python llm/tools/get_context.py architecture
    python llm/tools/get_context.py start
    python llm/tools/get_context.py list
"""

import sys
import os

ROUTES = {
    "bug": [
        "llm/state/issues.md",
        "llm/context/code_snippets.md",
        "llm/state/current.md",
    ],
    "feature": [
        "llm/state/plan.md",
        "llm/context/architecture.md",
        "llm/context/code_snippets.md",
    ],
    "architecture": [
        "llm/context/architecture.md",
        "llm/history/decisions.md",
        "llm/context/dependencies.md",
    ],
    "start": [
        "llm/AGENT_START.md",
        "llm/state/current.md",
        "llm/state/session_handoff.md",
        "llm/state/plan.md",
    ],
    "end": [
        "llm/AGENT_END_ROUTINE.md",
        "llm/state/current.md",
        "llm/state/plan.md",
    ],
}

DESCRIPTIONS = {
    "bug":          "дебаггинг / фикс бага",
    "feature":      "добавление новой фичи",
    "architecture": "архитектурные вопросы / дизайн-решения",
    "start":        "начало новой сессии",
    "end":          "завершение сессии",
}


def print_context(task_type: str, base_dir: str) -> None:
    if task_type not in ROUTES:
        print(f"Неизвестный тип задачи: '{task_type}'")
        print(f"Доступные типы: {', '.join(ROUTES.keys())}")
        sys.exit(1)

    files = ROUTES[task_type]
    print(f"\n=== Контекст для задачи: {task_type} ({DESCRIPTIONS[task_type]}) ===\n")
    print("Файлы для чтения:")
    for f in files:
        full_path = os.path.join(base_dir, f)
        exists = "✓" if os.path.exists(full_path) else "✗ (не найден)"
        print(f"  {exists}  {f}")

    print("\n--- Содержимое файлов ---\n")
    for f in files:
        full_path = os.path.join(base_dir, f)
        print(f"\n{'='*60}")
        print(f"# {f}")
        print('='*60)
        if os.path.exists(full_path):
            with open(full_path, encoding="utf-8") as fh:
                print(fh.read())
        else:
            print("[файл не найден]")


def print_list() -> None:
    print("\nДоступные типы задач:")
    for k, v in DESCRIPTIONS.items():
        files = ", ".join(ROUTES[k])
        print(f"  {k:12s} — {v}")
        print(f"               файлы: {files}")
        print()


def find_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # tools/ → llm/ → project/
    return os.path.join(script_dir, "..", "..")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print_list()
        sys.exit(0)

    task = sys.argv[1].lower()
    root = find_project_root()
    print_context(task, root)
