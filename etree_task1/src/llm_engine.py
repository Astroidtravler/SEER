import logging
import math
import re
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, args):
        self.args = args
        self.api_url = f"{args.llm_url.rstrip('/')}/v1/chat/completions"
        self.model_name = args.llm_model
        self.temperature = getattr(args, "temperature", 0.7)
        self.timeout_s = getattr(args, "llm_timeout", 30)
        self.max_retry = getattr(args, "llm_max_retry", 2)
        self.headers = {"Content-Type": "application/json"}
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


    def reset_counters(self) -> None:
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_counters(self) -> Dict[str, int]:
        return {
            "llm_calls": int(self.request_count),
            "llm_prompt_tokens": int(self.prompt_tokens),
            "llm_completion_tokens": int(self.completion_tokens),
            "request_count": int(self.request_count),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
        }

    def _post_json(self, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        last_err = None
        for i in range(self.max_retry + 1):
            try:
                r = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                self.request_count += 1
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                return data
            except Exception as e:
                last_err = e
                if i < self.max_retry:
                    time.sleep(0.5 * (2 ** i))
        raise RuntimeError(f"LLM request failed after retry: {last_err}")

    def generate_actions(self, context, n_candidates=3):
        prompt = (
            "You are a strict logical reasoning system. "
            f"Propose {n_candidates} valid deductions.\n\n"
            "Rules:\n"
            "1) Each deduction must combine two or more facts.\n"
            "2) Do not repeat facts verbatim.\n"
            "3) Output format: <id1> & <id2> -> <new_conclusion>\n\n"
            f"Facts:\n{context}\n\n"
            "Output one deduction per line."
        )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 256,
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            data = self._post_json(payload, timeout=self.timeout_s)
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            logprobs_content = choice.get("logprobs", {}).get("content", [])
            return self._parse_generation_output(content, logprobs_content)
        except Exception as e:
            logger.error("LLM API Generation Error: %s", e)
            return []


    def evaluate_state_raw(self, node):
        """Return continuous reward in [-1, 1] from LLM judge score."""
        if not node.logical_parent_ids:
            return 1.0

        premises_text = []
        for pid in node.logical_parent_ids:
            if pid in node.id2sent:
                premises_text.append(f"[{pid}] {node.id2sent[pid]}")
            else:
                for text, mapped_id in node.used_premises.items():
                    if mapped_id == pid:
                        premises_text.append(f"[{pid}] {text}")
                        break

        conclusion_id = node.latest_conclusion_id
        conclusion_text = node.id2sent.get(conclusion_id, "") if conclusion_id else ""
        premises_str = "\n".join(premises_text)

        prompt = (
            "You are an expert logician. Return a real score in [-1, 1] for whether conclusion follows from premises.\n\n"
            f"Premises:\n{premises_str}\n\n"
            f"Conclusion:\n{conclusion_text}\n\n"
            "Output exactly one float number."
        )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
        }

        try:
            data = self._post_json(payload, timeout=10)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            val = self._extract_score(content)
            return max(-1.0, min(1.0, val))
        except Exception as e:
            logger.error("LLM API Raw Evaluation Error: %s", e)
            return -1.0

    def evaluate_state(self, node):
        if not node.logical_parent_ids:
            return self.args.reward_value

        premises_text = []
        for pid in node.logical_parent_ids:
            if pid in node.id2sent:
                premises_text.append(f"[{pid}] {node.id2sent[pid]}")
            else:
                for text, mapped_id in node.used_premises.items():
                    if mapped_id == pid:
                        premises_text.append(f"[{pid}] {text}")
                        break

        conclusion_id = node.latest_conclusion_id
        conclusion_text = node.id2sent.get(conclusion_id, "") if conclusion_id else ""
        premises_str = "\n".join(premises_text)

        prompt = (
            "You are an expert logician. Judge whether the conclusion strictly follows.\n\n"
            f"Premises:\n{premises_str}\n\n"
            f"Conclusion:\n{conclusion_text}\n\n"
            "Output exactly one number: 1.0 / -0.5 / -1.0"
        )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 8,
        }

        try:
            data = self._post_json(payload, timeout=10)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            val = self._extract_score(content)
            if val >= 0.9:
                return self.args.reward_value
            if -0.6 < val < 0:
                return self.args.redundant_value
            return self.args.penalize_value
        except Exception as e:
            logger.error("LLM API Evaluation Error: %s", e)
            return self.args.penalize_value

    def _extract_score(self, content: str) -> float:
        m = re.search(r"-?\d+(?:\.\d+)?", content)
        if not m:
            return -1.0
        try:
            return float(m.group(0))
        except Exception:
            return -1.0

    def _parse_generation_output(self, content, logprobs_content):
        candidates = []
        pattern = re.compile(r"^\s*((?:sent\d+|int\d+|hypothesis)(?:\s*&\s*(?:sent\d+|int\d+|hypothesis))+?)\s*->\s*(.+?)\s*$")
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        line_logprobs = self._split_logprobs_by_line(logprobs_content)
        priors = []
        parsed = []

        for idx, line in enumerate(lines):
            match = pattern.match(line)
            if not match:
                continue
            lhs = match.group(1).strip()
            conclusion = match.group(2).strip().strip(' .*"\'')
            pre_ids = [pid.strip() for pid in lhs.split("&")]
            parsed.append((line, pre_ids, conclusion))
            avg_lp = np_mean(line_logprobs[idx]) if idx < len(line_logprobs) else -5.0
            priors.append(math.exp(max(-20.0, min(0.0, avg_lp))))

        if not parsed:
            return []

        z = sum(priors) if sum(priors) > 0 else 1.0
        priors = [p / z for p in priors]

        for (line, pre_ids, conclusion), p in zip(parsed, priors):
            action_str = f"reason: {' & '.join(pre_ids)} -> {conclusion}"
            candidates.append(
                {
                    "action_str": action_str,
                    "pre_ids": pre_ids,
                    "conclusion": conclusion,
                    "probability": p,
                    "raw_line": line,
                }
            )
        return candidates

    def _split_logprobs_by_line(self, logprobs_content: List[Dict[str, Any]]) -> List[List[float]]:
        if not logprobs_content:
            return []
        rows: List[List[float]] = [[]]
        for t in logprobs_content:
            token = t.get("token", "")
            lp = t.get("logprob", None)
            if lp is not None:
                rows[-1].append(lp)
            if "\n" in token:
                rows.append([])
        return [r for r in rows if r]


def np_mean(vals: List[float]) -> float:
    if not vals:
        return -5.0
    return float(sum(vals) / len(vals))
