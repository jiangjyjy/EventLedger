from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from carve.datasets.nq_openqa import NQOpenQACase
from carve.schemas import Event, Trace
from carve.verifiers.openqa import OpenQAExactMatchVerifier


class NQOpenQAClient(Protocol):
    def complete(self, role: str, prompt: str, seed: int) -> str:
        ...


def _format_contexts(case: NQOpenQACase, indices: list[int], max_chars: int) -> str:
    return "\n\n".join(
        f"[{index}] {case.contexts[index - 1][0]}\n{case.contexts[index - 1][1][:max_chars]}"
        for index in indices
    )


def _route_indices(available: int) -> tuple[list[int], list[int]]:
    """Keep top evidence shared while routing the remaining evidence differently."""
    indices = list(range(1, available + 1))
    if not indices:
        return [], []
    reader_a = [indices[0], *indices[1::2]]
    reader_b = [indices[0], *indices[2::2]]
    if len(indices) > 1 and len(reader_b) == 1:
        reader_b.append(indices[1])
    return reader_a, reader_b


def _parse_reader_output(content: str, available_indices: list[int]) -> tuple[str, int | None, float | None]:
    answer = content.strip()
    citation: int | None = None
    confidence: float | None = None
    for line in content.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized = key.strip().lower()
        if normalized == "answer":
            answer = value.strip()
        elif normalized == "evidence":
            match = re.search(r"\d+", value)
            if match:
                citation = int(match.group())
        elif normalized == "confidence":
            try:
                confidence = float(value.strip())
            except ValueError:
                pass
    if citation not in available_indices:
        citation = None
    return answer, citation, confidence


def _evidence_check(answer: str, citation: int | None, case: NQOpenQACase) -> dict[str, Any]:
    if citation is None or citation < 1 or citation > len(case.contexts):
        return {"citation_exists": False, "answer_supported": False, "citation": citation}
    passage = case.contexts[citation - 1][1].lower()
    terms = [term for term in re.findall(r"[a-z0-9]+", answer.lower()) if len(term) > 1]
    supported = bool(terms) and all(term in passage for term in terms)
    return {"citation_exists": True, "answer_supported": supported, "citation": citation}


def _answer_shape_guidance(question: str) -> str:
    """Constrain extraction to the semantic granularity requested by the question."""
    lowered = question.lower().strip()
    if lowered.startswith(("when ", "what year ", "in what year ")):
        return "If the passage provides a day-month-year, return the full date rather than only its year; return only the requested year when the question explicitly asks for a year. Do not append a noun or event description."
    if lowered.startswith(("how many ", "what number ", "how much ")):
        return "Return only the number or measured quantity, including a unit only when needed to identify it."
    if lowered.startswith(("who ", "which person ")):
        return "Return only the person's name who directly satisfies the question; do not substitute an author, songwriter, band member, or related person mentioned nearby."
    if lowered.startswith(("where ", "which place ")):
        return "Return only the place name, without an explanation or surrounding sentence."
    if lowered.startswith(("what is the name", "what was the name", "which book", "which film", "which song")):
        return "Return only the requested title or name, preserving edition/version qualifiers and numbered editions when they are part of the answer. Never shorten an answer to a bare number if the passage attaches a required qualifier."
    return "Return the shortest exact phrase that answers the question; do not add attributes that were not asked for."


class NQOpenQARunner:
    def __init__(self, client: NQOpenQAClient, *, verifier: OpenQAExactMatchVerifier | None = None, top_k: int = 8, max_context_chars: int = 900, reader_context_mode: str = "split"):
        if reader_context_mode not in {"split", "full_top_k"}:
            raise ValueError(f"unsupported reader context mode: {reader_context_mode}")
        self.client = client
        self.verifier = verifier or OpenQAExactMatchVerifier()
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.reader_context_mode = reader_context_mode

    def run(self, case: NQOpenQACase, *, seed: int = 0, model: str = "unknown", split: str = "smoke") -> Trace:
        trace_id = f"{case.task_id}-nq-dpr-{seed}"
        events: list[Event] = []
        retrieved_count = min(self.top_k, len(case.contexts))
        retrieved_indices = list(range(1, retrieved_count + 1))
        if self.reader_context_mode == "full_top_k":
            reader_a_indices, reader_b_indices = retrieved_indices, retrieved_indices
        else:
            reader_a_indices, reader_b_indices = _route_indices(retrieved_count)

        def add(event_id: str, event_type: str, role: str, content: str, parents: list[str], *, telemetry: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None, event_model: str | None = None) -> None:
            events.append(Event(event_id, trace_id, case.task_id, len(events), event_type, role, f"{role}-1", content, parents, model=event_model, tokens_in=int((telemetry or {}).get("input_tokens", 0)), tokens_out=int((telemetry or {}).get("output_tokens", 0)), latency_ms=float((telemetry or {}).get("wall_clock_latency_ms", 0.0)), metadata={"telemetry": telemetry or {}, **(metadata or {})}))

        retrieved_evidence = _format_contexts(case, retrieved_indices, self.max_context_chars)
        add("e1", "tool", "dpr_retrieval", retrieved_evidence, [], metadata={"evidence_ids": list(case.evidence_ids), "top_k": retrieved_count, "context_indices": retrieved_indices})
        router_payload = {"reader_a_indices": reader_a_indices, "reader_b_indices": reader_b_indices}
        add("e2", "assign", "evidence_router", json.dumps(router_payload, sort_keys=True), ["e1"], metadata=router_payload)

        def call(role: str, prompt: str, event_id: str, parents: list[str]) -> str:
            started = time.perf_counter()
            content = str(self.client.complete(role, prompt, seed + len(events))).strip()
            reported = self.client.last_completion_telemetry() if hasattr(self.client, "last_completion_telemetry") else {}
            telemetry = {
                "api_calls": int(reported.get("api_calls", 1)),
                "api_request_attempts": int(reported.get("api_request_attempts", reported.get("api_calls", 1))),
                "input_tokens": int(reported.get("input_tokens", len(prompt.split()))),
                "output_tokens": int(reported.get("output_tokens", len(content.split()))),
                "wall_clock_latency_ms": float(reported.get("wall_clock_latency_ms", (time.perf_counter() - started) * 1000)),
            }
            add(event_id, "revise" if role.startswith("reader") else "aggregate", role, content, parents, telemetry=telemetry, event_model=model)
            return content

        def reader_prompt(indices: list[int], strategy: str) -> str:
            evidence = _format_contexts(case, indices, self.max_context_chars)
            shape = _answer_shape_guidance(case.question)
            return f"You are a {strategy}. Use only the supplied evidence. Extract the shortest exact contiguous answer span that directly answers the question; do not paraphrase, generalize, hedge, or output Unknown when an answer can be copied from a passage. {shape} Before answering, identify what entity, attribute, date granularity, or qualifier the question requests. Preserve every qualifier needed to distinguish the requested answer from nearby related facts. Cite the single passage that contains the answer.\n\nQuestion:\n{case.question}\n\nEvidence:\n{evidence}\n\nReturn exactly:\nAnswer: <short exact answer span>\nEvidence: <passage index>\nConfidence: <0-1>"

        raw_a = call("reader_a", reader_prompt(reader_a_indices, "precision evidence extractor"), "e3", ["e2"])
        answer_a, citation_a, confidence_a = _parse_reader_output(raw_a, reader_a_indices)
        events[-1].metadata.update({"context_indices": reader_a_indices, "answer": answer_a, "citation": citation_a, "confidence": confidence_a})
        raw_b = call("reader_b", reader_prompt(reader_b_indices, "independent verifier"), "e4", ["e2"])
        answer_b, citation_b, confidence_b = _parse_reader_output(raw_b, reader_b_indices)
        events[-1].metadata.update({"context_indices": reader_b_indices, "answer": answer_b, "citation": citation_b, "confidence": confidence_b})

        check_a = _evidence_check(answer_a, citation_a, case)
        add("e5", "tool", "evidence_check_a", json.dumps(check_a, sort_keys=True), ["e3"], metadata=check_a)
        check_b = _evidence_check(answer_b, citation_b, case)
        add("e6", "tool", "evidence_check_b", json.dumps(check_b, sort_keys=True), ["e4"], metadata=check_b)

        selector_prompt = (
            f"Question: {case.question}\n\nCandidate A: {answer_a}\nCitation A: {citation_a}\n"
            f"A support: {check_a['answer_supported']}\nExcerpt A: {_format_contexts(case, [citation_a] if citation_a else [], self.max_context_chars)}\n\n"
            f"Candidate B: {answer_b}\nCitation B: {citation_b}\nB support: {check_b['answer_supported']}\n"
            f"Excerpt B: {_format_contexts(case, [citation_b] if citation_b else [], self.max_context_chars)}\n\n"
            f"Answer-shape rule: {_answer_shape_guidance(case.question)}\n"
            "Choose the candidate that answers the exact question at the requested granularity. Prefer an exact contiguous span over a paraphrase. Reject answers that add an unasked attribute or drop a required qualifier (for example, reject '581' when the question asks for '581 second edition', and reject a year-only answer when the passage provides the requested full date). For who-questions, verify that the selected person is the one asked about, not a nearby author or collaborator. Return exactly candidate_a, candidate_b, or abstain."
        )
        choice = call("selector", selector_prompt, "e7", ["e3", "e4", "e5", "e6"]).lower().splitlines()[0]
        if choice not in {"candidate_a", "candidate_b", "abstain"}:
            choice = "abstain"
        supported = {
            "candidate_a": bool(check_a["answer_supported"]) and bool(answer_a.strip()) and answer_a.strip().lower() != "unknown",
            "candidate_b": bool(check_b["answer_supported"]) and bool(answer_b.strip()) and answer_b.strip().lower() != "unknown",
        }
        if choice == "abstain" and sum(supported.values()) == 1:
            choice = next(name for name, is_supported in supported.items() if is_supported)
        final = answer_a if choice == "candidate_a" else answer_b if choice == "candidate_b" else ""
        add("e8", "aggregate", "final_resolver", final, ["e7"], metadata={"choice": choice})
        result = self.verifier.verify(final, case.answers)
        add("e9", "tool", "answer_alias_verifier", json.dumps(result.details, sort_keys=True), ["e8"], metadata={"verifier_score": result.score, "verifier_success": result.success})
        telemetry = [event.metadata.get("telemetry", {}) for event in events]
        return Trace(trace_id, case.task_id, "natural_questions_open_dpr_dev", split, events, final, verifier_score=result.score, success=result.success, manifest={"workflow": "nq_openqa_dpr_dag_v2", "telemetry": {"api_calls": sum(int(item.get("api_calls", 0)) for item in telemetry), "input_tokens": sum(int(item.get("input_tokens", 0)) for item in telemetry), "output_tokens": sum(int(item.get("output_tokens", 0)) for item in telemetry)}, "evidence_ids": list(case.evidence_ids)})
