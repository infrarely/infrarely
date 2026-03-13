"""
memory/structured.py
Persistent JSON-backed store for structured data,
assignments, notes, and calendar events.

Tools read/write through this layer — never through raw files.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import infrarely.core.app_config as config
from infrarely.observability import logger
from infrarely.observability.metrics import collector


def _load(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class StructuredMemory:
    """
    In-memory cache backed by JSON files.
    All writes are flushed immediately (simple consistency for local use).
    """

    def __init__(self):
        self._profiles: Dict = _load(config.PROFILE_FILE, {})
        self._courses: Dict = _load(config.COURSES_FILE, {})
        self._assignments: Dict = _load(config.ASSIGNMENTS_FILE, {})
        self._notes: Dict = _load(config.NOTES_FILE, {})
        self._calendar: Dict = _load(config.CALENDAR_FILE, {})
        logger.memory_log(
            "init",
            "structured",
            profiles=len(self._profiles),
            courses=sum(len(v) for v in self._courses.values()),
            assignments=sum(len(v) for v in self._assignments.values()),
        )

    # ── Student Profile ───────────────────────────────────────────────────
    def get_profile(self, student_id: str) -> Optional[Dict]:
        collector.record_memory_read()
        return self._profiles.get(student_id)

    def set_profile(self, student_id: str, profile: Dict):
        collector.record_memory_write()
        self._profiles[student_id] = profile
        _save(config.PROFILE_FILE, self._profiles)
        logger.memory_log("write", "structured", entity="profile", id=student_id)

    # ── Courses ────────────────────────────────────────────────────────────
    def get_courses(self, student_id: str) -> List[Dict]:
        collector.record_memory_read()
        return self._courses.get(student_id, [])

    def add_course(self, student_id: str, course: Dict):
        collector.record_memory_write()
        if student_id not in self._courses:
            self._courses[student_id] = []
        self._courses[student_id].append(course)
        _save(config.COURSES_FILE, self._courses)

    # ── Assignments ────────────────────────────────────────────────────────
    def get_assignments(
        self, student_id: str, course_id: Optional[str] = None
    ) -> List[Dict]:
        collector.record_memory_read()
        assignments = self._assignments.get(student_id, [])
        if course_id:
            assignments = [a for a in assignments if a.get("course_id") == course_id]
        return assignments

    def get_upcoming_assignments(self, student_id: str, days: int = 7) -> List[Dict]:
        from datetime import datetime, timedelta

        collector.record_memory_read()
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days)
        result = []
        for a in self._assignments.get(student_id, []):
            try:
                due = datetime.fromisoformat(a.get("due_date", ""))
                if now <= due <= cutoff:
                    result.append(a)
            except ValueError:
                pass
        return sorted(result, key=lambda x: x.get("due_date", ""))

    def add_assignment(self, student_id: str, assignment: Dict):
        collector.record_memory_write()
        if student_id not in self._assignments:
            self._assignments[student_id] = []
        self._assignments[student_id].append(assignment)
        _save(config.ASSIGNMENTS_FILE, self._assignments)

    def update_assignment_status(
        self, student_id: str, assignment_id: str, status: str
    ) -> bool:
        collector.record_memory_write()
        for a in self._assignments.get(student_id, []):
            if a.get("id") == assignment_id:
                a["status"] = status
                _save(config.ASSIGNMENTS_FILE, self._assignments)
                return True
        return False

    # ── Notes ─────────────────────────────────────────────────────────────
    def get_notes(self, student_id: str, course_id: Optional[str] = None) -> List[Dict]:
        collector.record_memory_read()
        notes = self._notes.get(student_id, [])
        if course_id:
            notes = [n for n in notes if n.get("course_id") == course_id]
        return notes

    def search_notes(self, student_id: str, query: str) -> List[Dict]:
        collector.record_memory_read()
        q = query.lower()
        return [
            n
            for n in self._notes.get(student_id, [])
            if q in n.get("title", "").lower() or q in n.get("content", "").lower()
        ]

    def add_note(self, student_id: str, note: Dict):
        collector.record_memory_write()
        if student_id not in self._notes:
            self._notes[student_id] = []
        self._notes[student_id].append(note)
        _save(config.NOTES_FILE, self._notes)

    # ── Calendar ──────────────────────────────────────────────────────────
    def get_events(
        self, student_id: str, date_prefix: Optional[str] = None
    ) -> List[Dict]:
        collector.record_memory_read()
        events = self._calendar.get(student_id, [])
        if date_prefix:
            events = [e for e in events if e.get("date", "").startswith(date_prefix)]
        return sorted(events, key=lambda x: x.get("date", ""))

    def add_event(self, student_id: str, event: Dict):
        collector.record_memory_write()
        if student_id not in self._calendar:
            self._calendar[student_id] = []
        self._calendar[student_id].append(event)
        _save(config.CALENDAR_FILE, self._calendar)
