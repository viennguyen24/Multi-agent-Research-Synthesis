from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

SESSION_TABLES = [
    "current_research_context",
    "current_plan",
    "current_draft",
    "current_critique",
    "current_issues",
    "supervisor_evals",
]

WM_SCHEMA_SQL = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[0]} (
    session_id  TEXT PRIMARY KEY,
    queries     TEXT,           -- JSON array of query strings
    results     TEXT,           -- JSON array of result dicts, parallel to queries
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[1]} (
    session_id       TEXT PRIMARY KEY,
    title            TEXT,
    guidelines       TEXT,   -- JSON dict
    success_criteria TEXT,   -- JSON array of strings
    introduction     TEXT,
    sections         TEXT,   -- JSON array of SectionBlock dicts
    conclusion       TEXT,
    updated_at       TEXT
);

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[2]} (
    session_id  TEXT PRIMARY KEY,
    plan_snap   TEXT,   -- snapshot of plan title/version at write time, for critic alignment
    document    TEXT,
    word_count  INTEGER,
    action      TEXT CHECK(action IN ('initial', 'critic_revision', 'replan_revision')),
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[3]} (
    session_id    TEXT PRIMARY KEY,
    summary       TEXT,
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[4]} (
    session_id       TEXT,
    issue_id         TEXT,    -- e.g. "ISS_001"
    severity         TEXT CHECK(severity IN ('critical', 'major', 'minor')),
    location         TEXT,
    type             TEXT,
    description      TEXT,
    PRIMARY KEY (session_id, issue_id)
);

CREATE TABLE IF NOT EXISTS {SESSION_TABLES[5]} (
    session_id  TEXT,
    version     INTEGER,
    decision    TEXT CHECK(decision IN ('accept', 'revise', 'replan')),
    reasoning   TEXT,
    feedback    TEXT,   -- includes full issue descriptions inline
    failures    TEXT,
    learnings   TEXT,
    created_at  TEXT,
    PRIMARY KEY (session_id, version)
);
"""

class WorkingMemorySession:
    """
    Single interface for all agent nodes to read/write working memory.
    
    Instantiate per-node-call with the session's db_path and session_id:
        wm = WorkingMemorySession(db_path=state["db_path"], session_id=state["session_id"])
    """

    def __init__(self, db_path: str, session_id: str) -> None:
        self.db_path = db_path
        self.session_id = session_id

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save_research_context(self, queries: list[str], results: list[dict]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO current_research_context (session_id, queries, results, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    queries=excluded.queries,
                    results=excluded.results,
                    updated_at=excluded.updated_at
                """,
                (self.session_id, json.dumps(queries), json.dumps(results), self._now()),
            )

    def get_current_research_context(self) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT queries, results, updated_at FROM current_research_context WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "queries": json.loads(row["queries"]) if row["queries"] else [],
            "results": json.loads(row["results"]) if row["results"] else [],
            "updated_at": row["updated_at"],
        }

    def save_plan(self, plan: dict) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO current_plan
                    (session_id, title, guidelines, success_criteria, introduction, sections, conclusion, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title,
                    guidelines=excluded.guidelines,
                    success_criteria=excluded.success_criteria,
                    introduction=excluded.introduction,
                    sections=excluded.sections,
                    conclusion=excluded.conclusion,
                    updated_at=excluded.updated_at
                """,
                (
                    self.session_id,
                    plan.get("title"),
                    json.dumps(plan.get("guidelines", {})),
                    json.dumps(plan.get("success_criteria", [])),
                    plan.get("introduction"),
                    json.dumps(plan.get("sections", [])),
                    plan.get("conclusion"),
                    self._now(),
                ),
            )

    def get_current_plan(self) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT title, guidelines, success_criteria, introduction, sections, conclusion, updated_at FROM current_plan WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "title": row["title"],
            "guidelines": json.loads(row["guidelines"]) if row["guidelines"] else {},
            "success_criteria": json.loads(row["success_criteria"]) if row["success_criteria"] else [],
            "introduction": row["introduction"],
            "sections": json.loads(row["sections"]) if row["sections"] else [],
            "conclusion": row["conclusion"],
            "updated_at": row["updated_at"],
        }

    def save_draft(self, document: str, plan_snap: str, action: str) -> None:
        """
        plan_snap: JSON string snapshot of the plan at write time, e.g.
                   json.dumps({"title": plan["title"], "updated_at": plan["updated_at"]})
                   Used by Critic to confirm which plan version this draft was written against.
        """
        word_count = len(document.split())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO current_draft (session_id, plan_snap, document, word_count, action, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    plan_snap=excluded.plan_snap,
                    document=excluded.document,
                    word_count=excluded.word_count,
                    action=excluded.action,
                    updated_at=excluded.updated_at
                """,
                (self.session_id, plan_snap, document, word_count, action, self._now()),
            )

    def get_current_draft(self) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT plan_snap, document, word_count, action, updated_at FROM current_draft WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_critique(self, summary: str, issues: list[dict]) -> list[str]:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO current_critique (session_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary=excluded.summary,
                    updated_at=excluded.updated_at
                """,
                (self.session_id, summary, self._now())
            )
            
            conn.execute("DELETE FROM current_issues WHERE session_id = ?", (self.session_id,))
            issue_ids = [issue["issue_id"] for issue in issues]
            if issues:
                issue_params = [
                    (
                        self.session_id,
                        issue.get("issue_id"),
                        issue.get("severity"),
                        issue.get("location"),
                        issue.get("type"),
                        issue.get("description"),
                    )
                    for issue in issues
                ]
                conn.executemany(
                    "INSERT INTO current_issues (session_id, issue_id, severity, location, type, description) VALUES (?, ?, ?, ?, ?, ?)",
                    issue_params,
                )
            return issue_ids

    def get_current_critique(self) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT summary, updated_at FROM current_critique WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_current_issues(self) -> list[dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT issue_id, severity, location, type, description FROM current_issues WHERE session_id = ? ORDER BY issue_id",
                (self.session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_issues_by_ids(self, issue_ids: list[str]) -> list[dict]:
        if not issue_ids:
            return []
        placeholders = ",".join("?" * len(issue_ids))
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT issue_id, severity, location, type, description
                FROM current_issues
                WHERE session_id = ? AND issue_id IN ({placeholders})
                ORDER BY issue_id
                """,
                [self.session_id, *issue_ids],
            ).fetchall()
        return [dict(row) for row in rows]

    def save_supervisor_eval(
        self,
        decision: str,
        reasoning: str,
        feedback: str,
        failures: str,
        learnings: str,
    ) -> int:
        with self._get_connection() as conn:
            version_row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM supervisor_evals WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            version = version_row[0]
            
            conn.execute(
                """
                INSERT INTO supervisor_evals
                    (session_id, version, decision, reasoning, feedback, failures, learnings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    version,
                    decision,
                    reasoning,
                    feedback,
                    failures,
                    learnings,
                    self._now(),
                ),
            )
            return version

    def get_current_supervisor_eval(self) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT version, decision, reasoning, feedback, failures, learnings, created_at
                FROM supervisor_evals
                WHERE session_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (self.session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_supervisor_evals(self) -> list[dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT version, decision, reasoning, feedback, failures, learnings, created_at
                FROM supervisor_evals
                WHERE session_id = ?
                ORDER BY version ASC
                """,
                (self.session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reset_session(self) -> None:
        with self._get_connection() as conn:
            for table in reversed(SESSION_TABLES):
                conn.execute(
                    f"DELETE FROM {table} WHERE session_id = ?",
                    (self.session_id,),
                )

    def session_exists(self) -> bool:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM current_plan WHERE session_id = ? LIMIT 1",
                (self.session_id,),
            ).fetchone()
        return row is not None