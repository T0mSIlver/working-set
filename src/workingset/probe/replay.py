"""A session that replays a recorded agent session (`ws test --replay`).

Drop-in for `session.Session` inside `user_loop`: the same `next_turn`,
`commit` and image attributes, plus `think_s` and `output_tokens`, which a
synthetic session draws from the workload and a replayed one reads off the
trace (see `workingset.agent_trace`).

What is replayed, request by request:

  * the prompt size. The first request of a session is the shared byte-stable
    prefix (as much of it as the prompt holds) plus fresh text, so the server
    caches the prefix across sessions as it does a real system prompt. Each
    later request is the previous prompt, the server's own reply and fresh
    text up to the recorded size, so it extends the cached sequence. A prompt
    that shrank in the recording (compaction, a branch) or would pass the
    context cap is rebuilt from the prefix: cold beyond it, as it was.
  * the reply length: `max_tokens` is the recorded output, and `ignore_eos`
    (on by default) makes the server generate all of it.
  * the wait before each request: the recorded tool time or human pause,
    capped at `replay_max_gap_s`. A gap where the session sat unused (a
    retry after a failed request, a restart, a session left open: see
    `agent_trace.INACTIVE`) follows `replay_gap_policy` instead; by default
    it is replaced by a think gap drawn from the trace.

What is not: the generation time, which is whatever the server takes now
(the loop is closed, as it is for a real agent), and forced misses, since a
replay goes cold only where the recording did. Each simulated session starts
at a random request of a random recorded session, so a short window sees
sessions at every stage rather than all of them at their first, small prompt;
when its recording ends it moves on to another one, from its start.
"""
from __future__ import annotations

import functools
import random

from ..agent_trace import (DEFAULT_GAP_POLICY, AgentTrace, load,
                           replay_gap, think_pool)
from .session import make_text, nonce_bits


@functools.lru_cache(maxsize=4)
def load_trace(path: str) -> AgentTrace:
    return load(path)


_POOLS: dict[int, list[float]] = {}


def _pool(trace: AgentTrace) -> list[float]:
    key = id(trace)
    if key not in _POOLS:
        _POOLS[key] = think_pool(trace)
    return _POOLS[key]


class ReplaySession:
    def __init__(self, trace: AgentTrace, uid: int, prefix_text: str,
                 prefix_tokens: int, cpt: float, cap_tokens: int,
                 max_gap_s: float, rng: random.Random,
                 text_rng: random.Random,
                 gap_policy: str = DEFAULT_GAP_POLICY):
        self.trace, self.uid, self.is_sub = trace, uid, False
        self.prefix_text, self.prefix_tokens = prefix_text, prefix_tokens
        self.cpt, self.cap, self.max_gap_s = cpt, cap_tokens, max_gap_s
        self.rng, self.text_rng = rng, text_rng
        self.gap_policy, self.pool = gap_policy, _pool(trace)
        self.n_turn = 0
        self.turn_images: list = []
        self.n_new_images = 0
        self._text = ""
        self._prev: tuple[int, int] | None = None   # (prompt, output) sent
        self._fresh = True                            # next turn starts over
        # a random request of a random recording: weighted by requests, so a
        # long session is entered more often, as it is more often running
        flat = [(i, j) for i, s in enumerate(trace.sessions)
                for j in range(len(s))]
        self._s, self._pos = flat[rng.randrange(len(flat))]

    @property
    def _req(self):
        return self.trace.sessions[self._s][self._pos]

    def _chars(self, tokens: float) -> int:
        return int(tokens * self.cpt)

    def next_turn(self, force_miss: bool | None = None) -> tuple[str, str]:
        r = self._req
        want = min(r.prompt_tokens, self.cap)
        prev = self._prev
        if self._fresh or prev is None or want < prev[0] or \
                r.prompt_tokens > self.cap:
            kind = "first" if (self.n_turn == 0 or self._pos == 0) else "miss"
            n_prefix = min(self.prefix_tokens, want)
            self._text = (self.prefix_text[:self._chars(n_prefix)] + "\n"
                          + make_text(self.text_rng, max(want - n_prefix, 0),
                                      self.cpt))
            self._fresh = False
        else:
            kind = "hit"
            grow = want - prev[0] - prev[1]
            if grow > 0:
                self._text += "\n" + make_text(self.text_rng, grow, self.cpt)
        return self._text, kind

    def commit(self, reply: str) -> None:
        r = self._req
        self._text += reply
        self._prev = (r.prompt_tokens, r.output_tokens)
        self.n_turn += 1
        self._pos += 1
        if self._pos >= len(self.trace.sessions[self._s]):
            self._s = self.rng.randrange(len(self.trace.sessions))
            self._pos, self._prev, self._fresh = 0, None, True

    def think_s(self, wl) -> float:
        gap = replay_gap(self._req, self.gap_policy, self.max_gap_s,
                         self.rng, self.pool)
        if gap is None:
            # a new recording starts: the pause before it was not recorded,
            # so draw a think gap of the trace (around the workload's mean
            # when it holds none)
            gap = (self.pool[self.rng.randrange(len(self.pool))] if self.pool
                   else self.rng.expovariate(1.0 / wl.think_time_s))
        return min(gap, self.max_gap_s)

    def output_tokens(self, wl) -> int:
        return max(1, self._req.output_tokens)

    def intended_prompt_tokens(self, prompt: str) -> int:
        return int(len(prompt) / self.cpt)


def make_replay_session(wl, opts, prefixes, uid: int,
                        seed: int | None = None) -> ReplaySession:
    """Seeded like `make_session`: (seed, uid) fixes which recording and where
    in it; the run nonce moves only the bytes."""
    seed = opts.seed if seed is None else seed
    bits = nonce_bits(getattr(opts, "run_nonce", ""))
    return ReplaySession(
        trace=load_trace(opts.replay), uid=uid, prefix_text=prefixes.user,
        prefix_tokens=wl.system_prefix_tokens, cpt=opts.chars_per_token,
        cap_tokens=opts.context_cap_tokens, max_gap_s=opts.replay_max_gap_s,
        rng=random.Random(f"replay {seed} {uid}"),
        gap_policy=getattr(opts, "replay_gap_policy", DEFAULT_GAP_POLICY),
        text_rng=random.Random(f"replay-text {seed} {uid} {bits:016x}"))
