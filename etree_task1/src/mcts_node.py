import copy
import hashlib
import json
import logging
from typing import Dict, List, Optional, Set

from utils import normalize, sort_key

logger = logging.getLogger(__name__)


class Action:
    retrieve = "retrieve"
    reason = "reason"
    end = "end"


class MCTSNode:
    """State + graph statistics node for LLM-MCTS Graph-MDP."""

    def __init__(self, data_item=None, parent: Optional["MCTSNode"] = None, action: Optional[str] = None):
        # 1) reasoning context
        self.data_id: Optional[str] = None
        self.Q = ""
        self.A = ""
        self.H = ""
        self.num_int = 0
        self.used_premises: Dict[str, str] = {}
        self.sent2id: Dict[str, str] = {}
        self.id2sent: Dict[str, str] = {}
        self.proof_str: List[str] = []
        self.latest_conclusion_id: Optional[str] = None

        # 2) topology / mcts stats
        self.chronological_parent = parent
        self.action_taken = action
        self.logical_parent_ids: List[str] = []
        self.children: Dict[str, "MCTSNode"] = {}
        self.parents: Set["MCTSNode"] = set()

        self.visits = 0
        self.total_value = 0.0
        self.r_t = 0.0
        self.V_t = 0.0
        self.prior_p = 0.0
        self.node_hash: Optional[str] = None
        self.last_bellman_residual: Optional[float] = None

        # priors attached to incoming edges (for weighted parent aggregation)
        self.incoming_prior: Dict[str, float] = {}

        if parent is not None:
            self.parents.add(parent)

        if data_item is not None:
            self._init_from_data(data_item)
        elif parent is not None:
            self._copy_from_parent(parent)

    def add_parent(self, parent: "MCTSNode", edge_prior: float = 0.0) -> None:
        self.parents.add(parent)
        ph = parent.node_hash or parent.state_hash()
        self.incoming_prior[ph] = max(self.incoming_prior.get(ph, 0.0), float(edge_prior))

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def accumulate(self, value: float, inc_visit: bool = True) -> None:
        if inc_visit:
            self.visits += 1
        self.total_value += float(value)
        # keep V_t as running mean estimate even in value-only propagation
        denom = self.visits if self.visits > 0 else 1
        self.V_t = self.total_value / denom

    def set_bootstrap_value(self, value: float) -> None:
        self.V_t = float(value)

    def _init_from_data(self, data_item):
        self.data_id = data_item["id"]
        self.H = normalize(data_item["hypothesis"])
        self.Q = normalize(data_item.get("question", ""))
        self.A = normalize(data_item.get("answer", ""))

        self.id2sent = copy.deepcopy(data_item["meta"]["triples"])
        for k, v in self.id2sent.items():
            self.sent2id[v] = k

    def _copy_from_parent(self, parent_node):
        self.data_id = parent_node.data_id
        self.Q = parent_node.Q
        self.A = parent_node.A
        self.H = parent_node.H
        self.num_int = parent_node.num_int

        self.used_premises = copy.deepcopy(parent_node.used_premises)
        self.sent2id = copy.deepcopy(parent_node.sent2id)
        self.id2sent = copy.deepcopy(parent_node.id2sent)
        self.proof_str = copy.deepcopy(parent_node.proof_str)
        self.latest_conclusion_id = parent_node.latest_conclusion_id

    def validate_action(self, pre_ids: List[str], new_conclusion: str):
        if not pre_ids or len(pre_ids) < 2:
            return False, "too_few_premises", ""
        missing = [pid for pid in pre_ids if pid not in self.id2sent]
        if missing:
            logger.debug("Skip invalid action due to missing premises: %s", missing)
            return False, "missing_premises", ""
        clean_conclusion = normalize(new_conclusion)
        if not clean_conclusion:
            return False, "empty_conclusion", ""
        if clean_conclusion in self.sent2id:
            return False, "duplicate_conclusion", clean_conclusion
        return True, "ok", clean_conclusion

    def apply_action(self, pre_ids: List[str], new_conclusion: str) -> bool:
        ok, _, clean_conclusion = self.validate_action(pre_ids, new_conclusion)
        if not ok:
            return False

        self.logical_parent_ids = list(pre_ids)

        for pid in pre_ids:
            sent_text = self.id2sent.pop(pid)
            self.sent2id.pop(sent_text, None)
            self.used_premises[sent_text] = pid

        self.num_int += 1
        new_int_id = f"int{self.num_int}"
        self.id2sent[new_int_id] = clean_conclusion
        self.sent2id[clean_conclusion] = new_int_id
        self.latest_conclusion_id = new_int_id

        step_str = f"{' & '.join(pre_ids)} -> {new_int_id}: {clean_conclusion}"
        self.proof_str.append(step_str)
        return True

    def get_context_for_llm(self) -> str:
        available_facts = [f"[{k}] {v}" for k, v in sorted(self.id2sent.items(), key=sort_key)]
        return "\n".join(available_facts)

    def state_hash(self) -> str:
        canonical = self.canonical_state()
        payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        self.node_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return self.node_hash

    def canonical_state(self) -> Dict:
        """Canonical state representation for Graph-MDP equivalence relation.

        Two states are equivalent iff their canonical_state() are identical.
        """
        return {
            "hypothesis": self.H,
            "facts": sorted(self.id2sent.items(), key=lambda x: x[0]),
            "used": sorted(self.used_premises.items(), key=lambda x: x[1]),
        }

    def is_terminal(self) -> bool:
        action_done = self.action_taken is not None and Action.end in str(self.action_taken).lower()
        reached_hypothesis = self.H and any(normalize(v) == self.H for v in self.id2sent.values())
        exhausted = len(self.id2sent) < 2
        return action_done or reached_hypothesis or exhausted
