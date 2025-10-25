#!/usr/bin/env python3
"""
Ingests Summa Theologica articles from a plaintext file into a SQLite database.

For each article, this script:
- Parses structured sections (Objections, On the contrary, I answer that, Replies).
- Uses the OpenAI Python SDK to rephrase the article into a concise quiz question.
- Writes the question and responses (correct + incorrect with refutations) to SQLite.

Key behaviors:
- Stream-parse the large source file; process one article at a time.
- Resumable: skips articles that are already complete in the DB.
- Progress: prints per-article progress (skipped/processed and running totals).
- Guardrails: outputs avoid phrases like "According to Aquinas"; content assumes the article’s teaching is correct.

CLI examples:
  scripts/ingest_summa.py --source summa_theologica.txt --db lucelingo.db --max-articles 5
  scripts/ingest_summa.py --range 2:5       # only Q2..Q5
  scripts/ingest_summa.py --skip-openai     # fallback: minimal heuristic text

Environment:
- OPENAI_API_KEY required unless --skip-openai
- OPENAI_MODEL optional (default: gpt-4o-mini)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import typing as t
from dataclasses import dataclass, field

try:
    # OpenAI SDK (installed via shell.nix)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Article:
    part_number: int
    question_number: int
    article_number: int
    raw_title: str  # e.g., "Whether Man's Happiness Consists in Wealth?"
    objections: list[tuple[int, str]] = field(default_factory=list)  # (num, text)
    on_the_contrary: str | None = None
    i_answer_that: str | None = None
    replies: dict[int, str] = field(default_factory=dict)  # num -> reply text


# -----------------------------
# Utilities
# -----------------------------


PART_CODE_TO_INT = {
    "I": 1,
    "I-II": 2,
    "II-II": 3,
    "III": 4,
    "Suppl.": 5,
}


def load_env_from_dotenv(path: str = ".env") -> None:
    """Minimal .env loader to set environment variables if present.
    Lines like KEY=VALUE. Ignores comments and export statements.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("export "):
                    s = s[len("export ") :]
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Non-fatal; continue with existing environment
        pass


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def map_part_code_to_int(code: str) -> int:
    code = code.strip()
    if code in PART_CODE_TO_INT:
        return PART_CODE_TO_INT[code]
    # Fallbacks for common variants
    variants = {
        "I-II": 2,
        "II-II": 3,
        "I-II.": 2,
        "II-II.": 3,
        "III.": 4,
        "SUPPL.": 5,
    }
    return variants.get(code.upper(), 0)


# -----------------------------
# Parser
# -----------------------------


class SummaParser:
    """Streaming parser for the Summa text file focusing on articles.

    Relies on bracket footers like: FIRST ARTICLE [I-II, Q. 2, Art. 1]
    and section markers: Objection N:, Obj. N:, On the contrary, I answer that,
    Reply Obj. N:, Reply to Objection N.
    """

    RE_ARTICLE_HEADER = re.compile(
        r"^(?:[A-Z]+ ARTICLE)\s*\[(?P<part>[^,]+),\s*Q\.?\s*(?P<q>\d+),\s*Art\.?\s*(?P<a>\d+)\]",
        re.IGNORECASE,
    )
    RE_QUESTION_HEADER = re.compile(r"^QUESTION\s+(?P<q>\d+)", re.IGNORECASE)
    RE_SECTION_OBJECTION = re.compile(
        r"^(?:Objection|Obj\.)\s*(?P<num>\d+)\s*:?", re.IGNORECASE
    )
    RE_SECTION_ON_THE_CONTRARY = re.compile(r"^_?On the contrary,?_?", re.IGNORECASE)
    RE_SECTION_I_ANSWER_THAT = re.compile(r"^_?I answer that,?_?", re.IGNORECASE)
    RE_SECTION_REPLY = re.compile(
        r"^(?:Reply\s*(?:to\s*Objection)?|Reply\s*Obj\.)\s*(?P<num>\d+)\s*:?",
        re.IGNORECASE,
    )
    RE_RULE = re.compile(r"^_{3,}\s*$")  # separators like ________________________

    def __init__(self, fp: t.TextIO):
        self.fp = fp
        self.current_article: Article | None = None
        self.current_capture: tuple[str, int | None] | None = None  # (section, number?)

    def _flush_capture(self, buf: list[str]) -> str:
        text = "\n".join(buf).strip()
        return text

    def _switch_section(self, section: str, num: int | None, buf: list[str]):
        # Store accumulated text into the previous section
        if self.current_article and self.current_capture and buf:
            sec, n = self.current_capture
            chunk = self._flush_capture(buf)
            if sec == "objection" and n is not None:
                self.current_article.objections.append((n, chunk))
            elif sec == "on_the_contrary":
                self.current_article.on_the_contrary = chunk
            elif sec == "i_answer_that":
                self.current_article.i_answer_that = chunk
            elif sec == "reply" and n is not None:
                self.current_article.replies[n] = chunk
        buf.clear()
        self.current_capture = (section, num)

    def _start_article(self, part: int, q: int, a: int, title: str):
        # Emit previous article if present
        if self.current_article is not None:
            yield self.current_article
        self.current_article = Article(
            part_number=part,
            question_number=q,
            article_number=a,
            raw_title=normalize_whitespace(title),
        )
        self.current_capture = None

    def _finalize(self, buf: list[str]):
        if self.current_article is not None:
            # Flush any pending capture
            if self.current_capture and buf:
                sec, n = self.current_capture
                chunk = self._flush_capture(buf)
                if sec == "objection" and n is not None:
                    self.current_article.objections.append((n, chunk))
                elif sec == "on_the_contrary":
                    self.current_article.on_the_contrary = chunk
                elif sec == "i_answer_that":
                    self.current_article.i_answer_that = chunk
                elif sec == "reply" and n is not None:
                    self.current_article.replies[n] = chunk
            art = self.current_article
            self.current_article = None
            self.current_capture = None
            return art
        return None

    def iter_articles(self) -> t.Iterator[Article]:
        buf: list[str] = []
        current_title: list[str] = []
        in_title_block = False
        title_started = False
        for raw in self.fp:
            line = raw.rstrip("\n\r")

            # Match start of an article header (with bracket footnote)
            m = self.RE_ARTICLE_HEADER.match(line.strip())
            if m:
                # Title expected a few lines below header; we will read next non-empty as title
                part_code = m.group("part").strip()
                qnum = int(m.group("q"))
                anum = int(m.group("a"))
                part_num = map_part_code_to_int(part_code)
                # Reset title accumulator
                current_title = []
                in_title_block = True
                title_started = False
                # Before switching, we must flush previous section
                yield from self._start_article(part_num, qnum, anum, title="")
                # Next lines will accumulate the title until a blank
                self.current_capture = None
                buf.clear()
                continue

            if in_title_block:
                if line.strip() == "":
                    if not title_started:
                        # skip leading blanks between header and the actual title line
                        continue
                    # assign title and end title block
                    title_text = normalize_whitespace(" ".join(current_title))
                    if self.current_article:
                        self.current_article.raw_title = title_text
                    in_title_block = False
                    title_started = False
                else:
                    current_title.append(line.strip())
                    title_started = True
                continue

            # Reset capture on visual rule separators (article boundary or section breaks)
            if self.RE_RULE.match(line):
                pass

            s = line.strip()
            if not s and not buf:
                # skip leading empty lines
                continue

            # Section switches
            m = self.RE_SECTION_OBJECTION.match(s)
            if m and self.current_article is not None:
                num = int(m.group("num"))
                # commit previous section
                self._switch_section("objection", num, buf)
                # remove the heading from the line content
                remainder = s[m.end() :].lstrip(" -:\t")
                if remainder:
                    buf.append(remainder)
                continue

            if (
                self.RE_SECTION_ON_THE_CONTRARY.match(s)
                and self.current_article is not None
            ):
                self._switch_section("on_the_contrary", None, buf)
                remainder = self.RE_SECTION_ON_THE_CONTRARY.sub("", s).lstrip(" ,:-\t")
                if remainder:
                    buf.append(remainder)
                continue

            if (
                self.RE_SECTION_I_ANSWER_THAT.match(s)
                and self.current_article is not None
            ):
                self._switch_section("i_answer_that", None, buf)
                remainder = self.RE_SECTION_I_ANSWER_THAT.sub("", s).lstrip(" ,:-\t")
                if remainder:
                    buf.append(remainder)
                continue

            m = self.RE_SECTION_REPLY.match(s)
            if m and self.current_article is not None:
                num = int(m.group("num"))
                self._switch_section("reply", num, buf)
                remainder = s[m.end() :].lstrip(" -:\t")
                if remainder:
                    buf.append(remainder)
                continue

            # If we find a QUESTION header after being in an article, it likely signals end of previous
            if self.RE_QUESTION_HEADER.match(s):
                continue

            # If we're inside a capture, append text
            if self.current_capture:
                buf.append(line)

        # finalize last article at EOF
        last = self._finalize(buf)
        if last is not None:
            yield last


# -----------------------------
# SQLite storage
# -----------------------------


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Question (
  id INTEGER PRIMARY KEY,
  question_text TEXT NOT NULL,
  part_number INTEGER NOT NULL,
  question_number INTEGER NOT NULL,
  article_number INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (part_number, question_number, article_number)
);

CREATE TABLE IF NOT EXISTS CorrectResponse (
  id INTEGER PRIMARY KEY,
  question_id INTEGER NOT NULL,
  response_text TEXT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES Question(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS IncorrectResponse (
  id INTEGER PRIMARY KEY,
  question_id INTEGER NOT NULL,
  response_text TEXT NOT NULL,
  refutation_text TEXT,
  FOREIGN KEY (question_id) REFERENCES Question(id) ON DELETE CASCADE
);
"""


class DB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.executescript(SCHEMA_SQL)

    def upsert_question(self, *, part: int, q: int, a: int, question_text: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO Question (question_text, part_number, question_number, article_number)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(part_number, question_number, article_number)
            DO UPDATE SET question_text=excluded.question_text, updated_at=CURRENT_TIMESTAMP
            """,
            (question_text, part, q, a),
        )
        # Fetch id explicitly (lastrowid not reliable for updates)
        cur.execute(
            "SELECT id FROM Question WHERE part_number=? AND question_number=? AND article_number=?",
            (part, q, a),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to retrieve upserted Question id")
        return int(row[0])

    def replace_responses(
        self,
        *,
        question_id: int,
        correct_text: str,
        incorrect: list[tuple[str, str | None]],  # (option, refutation)
    ) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM CorrectResponse WHERE question_id=?", (question_id,))
        cur.execute("DELETE FROM IncorrectResponse WHERE question_id=?", (question_id,))
        cur.execute(
            "INSERT INTO CorrectResponse (question_id, response_text) VALUES (?, ?)",
            (question_id, correct_text),
        )
        cur.executemany(
            "INSERT INTO IncorrectResponse (question_id, response_text, refutation_text) VALUES (?, ?, ?)",
            [(question_id, opt, ref) for (opt, ref) in incorrect],
        )
        self.conn.commit()

    def get_question_id(self, *, part: int, q: int, a: int) -> int | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id FROM Question WHERE part_number=? AND question_number=? AND article_number=?",
            (part, q, a),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

    def is_question_complete(self, question_id: int) -> bool:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM CorrectResponse WHERE question_id=?", (question_id,)
        )
        has_correct = int(cur.fetchone()[0]) > 0
        cur.execute(
            "SELECT COUNT(*) FROM IncorrectResponse WHERE question_id=?", (question_id,)
        )
        has_incorrect = int(cur.fetchone()[0]) > 0
        return has_correct and has_incorrect

    def close(self):
        self.conn.close()


# -----------------------------
# OpenAI client helpers
# -----------------------------


def call_openai_chat_json(
    *,
    client: "OpenAI",
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.1,
    retries: int = 5,
    backoff_base: float = 1.5,
) -> dict:
    """Call OpenAI Chat Completions with JSON-object response, with simple retries."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:  # includes rate limits, network hiccups
            last_err = e
            # Exponential-ish backoff
            sleep_s = backoff_base ** (attempt - 1)
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {retries} attempts: {last_err}")


def build_prompt(article: Article) -> tuple[str, str]:
    system = (
        "You transform scholastic articles into faithful, concise quiz items for a Catholic "
        "learning app. Return STRICT JSON only. Avoid any references to sources or authorities."
    )
    # Prepare user prompt with content summaries
    parts = []
    parts.append("Task: Rephrase this scholastic article into a quiz item.")
    parts.append("Requirements:")
    parts.append("- Produce a single-sentence question text (neutral, concise).")
    parts.append(
        "- Correct answer: 1–2 sentences summarizing 'I answer that', stated as true without attribution."
    )
    parts.append("- Incorrect options: summarize each Objection in 1 sentence.")
    parts.append(
        "- For each incorrect option, give a short refutation based on the matching Reply."
    )
    parts.append(
        "- If a matching Reply is missing, write a brief general refutation from 'I answer that'."
    )
    parts.append("- Be doctrinally precise and avoid straw-men, but keep it brief.")
    parts.append(
        "- Do NOT include phrases like 'According to Aquinas', 'Aquinas says', or any reference to Aquinas or other authorities."
    )
    parts.append(
        "- Write as if the article's teaching is correct; avoid hedging or disputing the conclusion."
    )
    parts.append("")
    parts.append(
        "Return JSON with keys: question_text, correct_answer, incorrect_options[]."
    )
    parts.append("Each incorrect_options item: { option, refutation }.")
    parts.append("")
    parts.append(f"Article title: {article.raw_title}")
    if article.on_the_contrary:
        otc = normalize_whitespace(article.on_the_contrary)
        parts.append(f"On the contrary: {otc}")
    if article.i_answer_that:
        iat = normalize_whitespace(article.i_answer_that)
        if len(iat) > 4000:
            iat = iat[:4000] + "..."
        parts.append(f"I answer that: {iat}")
    if article.objections:
        obs = []
        for n, txt in sorted(article.objections, key=lambda x: x[0]):
            obs.append(f"{n}. {normalize_whitespace(txt)}")
        parts.append("Objections:\n" + "\n".join(obs))
    if article.replies:
        reps = []
        for n in sorted(article.replies):
            reps.append(f"{n}. {normalize_whitespace(article.replies[n])}")
        parts.append("Replies:\n" + "\n".join(reps))

    user = "\n".join(parts)
    return system, user


def heuristic_fallback(article: Article) -> dict:
    # Minimal transformation without OpenAI: use raw title as question; correct from 'I answer that'
    q = article.raw_title
    correct = (
        "" if not article.i_answer_that else normalize_whitespace(article.i_answer_that)
    )
    incorrect = []
    for n, ttxt in sorted(article.objections, key=lambda x: x[0]):
        ref = article.replies.get(n) if article.replies else None
        incorrect.append(
            {
                "option": normalize_whitespace(ttxt),
                "refutation": None if ref is None else normalize_whitespace(ref),
            }
        )
    return {
        "question_text": q,
        "correct_answer": correct,
        "incorrect_options": incorrect,
    }


def scrub_authority_references(text: str) -> str:
    """Remove references like 'According to Aquinas' or 'Aquinas says'. Conservative cleanup.

    Intentionally simple; avoids adding new content, only removes disallowed phrases.
    """
    if not text:
        return text
    # Patterns to remove (case-insensitive)
    patterns = [
        r"\bAccording to\s+(?:St\.?\s*)?(?:Thomas\s+)?Aquinas\b[:,]?\s*",
        r"\bAs\s+(?:St\.?\s*)?(?:Thomas\s+)?Aquinas\s+(?:teaches|says|writes)\b[:,]?\s*",
        r"\bAquinas\s+says(?:\s+that)?\b[:,]?\s*",
        r"\bSt\.?\s*Thomas\s+Aquinas\b",
        r"\bThomas\s+Aquinas\b",
        r"\bAquinas\b",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    # Normalize whitespace after removals
    out = re.sub(r"\s+", " ", out).strip()
    # Remove awkward leading punctuation left behind
    out = re.sub(r"^[,;:\-\s]+", "", out)
    return out


def process_article(
    article: Article,
    db: DB,
    *,
    client: "OpenAI" | None,
    rate_limit_delay: float,
    model: str,
) -> None:
    system, user = build_prompt(article)
    if client is not None:
        data = call_openai_chat_json(
            client=client,
            model=model,
            system_prompt=system,
            user_prompt=user,
            max_tokens=600,
            temperature=0.1,
        )
        # Gentle pacing to respect rate limits (also have retry/backoff)
        time.sleep(rate_limit_delay)
    else:
        data = heuristic_fallback(article)

    question_text = data.get("question_text") or article.raw_title
    correct_answer = data.get("correct_answer") or (
        normalize_whitespace(article.i_answer_that or "")
    )
    incorrect_options = data.get("incorrect_options") or []
    # Build incorrect tuples
    incorrect: list[tuple[str, str | None]] = []
    for item in incorrect_options:
        opt = normalize_whitespace(str(item.get("option", "")).strip())
        ref = item.get("refutation")
        ref = None if ref is None else normalize_whitespace(str(ref))
        if opt:
            incorrect.append((opt, ref))

    # If there were no objections at all, fabricate at least one benign distractor based on OTC
    if not incorrect and article.on_the_contrary:
        incorrect.append((normalize_whitespace(article.on_the_contrary), ""))

    # Final scrub to ensure no authority references appear in outputs
    question_text = scrub_authority_references(question_text)
    correct_answer = scrub_authority_references(correct_answer)
    incorrect = [
        (
            scrub_authority_references(opt),
            None if ref is None else scrub_authority_references(ref),
        )
        for (opt, ref) in incorrect
    ]

    # Persist
    qid = db.upsert_question(
        part=article.part_number,
        q=article.question_number,
        a=article.article_number,
        question_text=question_text,
    )
    db.replace_responses(
        question_id=qid, correct_text=correct_answer, incorrect=incorrect
    )


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Ingest Summa articles into SQLite as quiz questions"
    )
    ap.add_argument(
        "--source", default="summa_theologica.txt", help="Path to Summa text file"
    )
    ap.add_argument(
        "--db", default="lucelingo.db", help="Path to SQLite DB to create/use"
    )
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument(
        "--max-articles", type=int, default=None, help="Process only first N articles"
    )
    ap.add_argument(
        "--range",
        default=None,
        help="Restrict by question number range inclusive: e.g. 2:5 (Q2..Q5). Leave empty for all.",
    )
    ap.add_argument(
        "--calls-per-minute",
        type=int,
        default=int(os.environ.get("RATE_LIMIT_CALLS", "15")),
    )
    ap.add_argument(
        "--skip-openai",
        action="store_true",
        help="Do not call OpenAI; use heuristic text",
    )
    return ap.parse_args(argv)


def in_range(qnum: int, r: tuple[int | None, int | None] | None) -> bool:
    if r is None:
        return True
    lo, hi = r
    if lo is not None and qnum < lo:
        return False
    if hi is not None and qnum > hi:
        return False
    return True


def main(argv: list[str]) -> int:
    load_env_from_dotenv()
    args = parse_args(argv)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.skip_openai and not api_key:
        print(
            "ERROR: OPENAI_API_KEY not set and --skip-openai not used", file=sys.stderr
        )
        return 2

    # Instantiate OpenAI SDK client
    client = None
    if not args.skip_openai:
        if OpenAI is None:
            print(
                "ERROR: openai SDK not available; install and try again or use --skip-openai",
                file=sys.stderr,
            )
            return 2
        client = OpenAI()

    db = DB(args.db)

    qrange: tuple[int | None, int | None] | None = None
    if args.range:
        try:
            lo_s, hi_s = args.range.split(":", 1)
            lo = int(lo_s) if lo_s.strip() else None
            hi = int(hi_s) if hi_s.strip() else None
            qrange = (lo, hi)
        except Exception:
            print(
                "Invalid --range format; expected like 2:5 (or :10 or 5:)",
                file=sys.stderr,
            )
            return 2

    calls_per_minute = max(1, args.calls_per_minute)
    delay = 60.0 / float(calls_per_minute)

    processed = 0
    skipped = 0
    seen = 0
    started = time.time()
    with open(args.source, "r", encoding="utf-8", errors="ignore") as fp:
        parser = SummaParser(fp)
        for art in parser.iter_articles():
            if not in_range(art.question_number, qrange):
                continue
            seen += 1
            ident = f"P{art.part_number} Q{art.question_number} A{art.article_number}"
            # Resume behavior: skip if already complete
            existing_id = db.get_question_id(
                part=art.part_number, q=art.question_number, a=art.article_number
            )
            if existing_id is not None and db.is_question_complete(existing_id):
                skipped += 1
                print(
                    f"{ident}: skip (already complete). Totals: processed={processed}, skipped={skipped}"
                )
                continue
            # Otherwise, process or reprocess
            try:
                print(f"{ident}: generating...", flush=True)
                process_article(
                    art, db, client=client, rate_limit_delay=delay, model=args.model
                )
                processed += 1
                print(
                    f"{ident}: done. Totals: processed={processed}, skipped={skipped}"
                )
            except Exception as e:
                print(f"{ident}: ERROR: {e}", file=sys.stderr)
            if args.max_articles and processed >= args.max_articles:
                break

    elapsed = time.time() - started
    print(
        f"Finished. Seen={seen}, processed={processed}, skipped={skipped} in {elapsed:.1f}s; DB at {args.db}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
