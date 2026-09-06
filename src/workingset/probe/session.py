"""Byte-stable prefixes and per-session context generation.

Ported verbatim from scripts/validate_deployment.py (`make_text`,
`draw_session_tokens`, and the prompt assembly inside `user_loop`). The three
properties that make the workload the one the model prices:

  * ONE byte-identical prefix per request class, shared by every session in
    the process — that is what vLLM's prefix cache dedups. Subagents get their
    own lean prefix unless `sub_shares_prefix` points them at the user one.
  * a per-session context drawn from the model's log-normal (median/sigma),
    clipped to [prefix, cap] exactly as `Workload.sample` clips.
  * a warm turn appends `warm_turn_tokens` of new text plus the previous
    reply, so the next turn extends the cached run the way a real agentic
    session does. A miss prepends a random salt AHEAD of the prefix, making
    the whole request unmatchable — the model's cache miss.

Token counts are chars/`chars_per_token` approximations; the achieved
`prompt_tokens` are read back from `usage` and reported against the intent.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# seeds fixed so the prefixes are byte-stable across processes as well as
# across sessions: two runs of the same config warm the SAME cache entry
USER_PREFIX_SEED = 0xC0FFEE
SUB_PREFIX_SEED = 0x5AB4EF1C

_VOCAB = ("the build failed on stage three because a header moved and the cache "
          "key did not include it so the runner reused a stale object then the "
          "linker saw two symbols with one name and gave up while the test "
          "suite kept polling a socket that nothing owned anymore").split()


def make_text(rng: random.Random, tokens: float, cpt: float) -> str:
    """~`tokens` worth of plain prose (cpt chars per token, measured later)."""
    budget = max(int(tokens * cpt), 1)
    out, n = [], 0
    while n < budget:
        w = _VOCAB[rng.randrange(len(_VOCAB))]
        out.append(w)
        n += len(w) + 1
    return " ".join(out)[:budget]


def draw_session_tokens(rng: random.Random, median: float, sigma: float,
                        prefix: float, cap: float) -> int:
    """One session's full prompt length: log-normal(median, sigma), clipped to
    [prefix, cap] exactly as Workload.sample clips (a prompt always contains
    at least its shared prefix, never more than the cap)."""
    full = rng.lognormvariate(math.log(median), sigma)
    return int(min(max(full, prefix), cap))


@dataclass(frozen=True)
class Prefixes:
    """The two byte-stable blocks a run shares across every session."""
    user: str
    sub: str


def build_prefixes(wl, cpt: float) -> Prefixes:
    """`wl` is a workingset.config.WorkloadCfg (or anything with the same
    two attributes). Same seeds and same generator as the harness's
    `state["system_prefix"]` / `state["sub_prefix"]`."""
    return Prefixes(
        user=make_text(random.Random(USER_PREFIX_SEED),
                       wl.system_prefix_tokens, cpt),
        sub=make_text(random.Random(SUB_PREFIX_SEED),
                      wl.subagent_prefix_tokens, cpt),
    )


def sub_prefix_floor(wl) -> int:
    """The clip floor a SUBAGENT session's draw uses: its own lean prefix,
    unless sub_shares_prefix points it at the user prefix. Displaying both
    classes at the user floor overstated the subagent p50 by ~87% under the
    reference config, which is why this is one function and not two literals."""
    return wl.system_prefix_tokens if wl.sub_shares_prefix \
        else wl.subagent_prefix_tokens


@dataclass
class Session:
    """One simulated user (assumption 1 of the planner): a context, a running
    history, and the prompt for the next turn."""
    uid: int
    is_sub: bool
    prefix_text: str
    prefix_tokens: int
    ctx: str
    rng: random.Random
    cpt: float
    warm_turn_tokens: int
    miss_rate: float
    n_turn: int = 0
    history: str = ""

    def next_turn(self, force_miss: bool | None = None) -> tuple[str, str]:
        """Return (prompt, kind). kind is "first" (session establishment),
        "hit" or "miss". A first turn is never a miss: there is nothing
        cached to invalidate."""
        first = self.n_turn == 0
        if force_miss is None:
            is_miss = (not first) and self.rng.random() < self.miss_rate
        else:
            is_miss = bool(force_miss) and not first
        # a salt ahead of the prefix makes the WHOLE request unmatchable — the
        # model's cache miss (full re-prefill), applied to this turn only
        salt = f"[miss-salt {self.rng.getrandbits(64):016x}] " if is_miss else ""
        turn_text = "" if first else "\n" + make_text(
            self.rng, self.warm_turn_tokens, self.cpt)
        prompt = salt + self.prefix_text + "\n" + self.ctx + self.history + turn_text
        self._pending_turn_text = turn_text
        return prompt, ("first" if first else ("miss" if is_miss else "hit"))

    def commit(self, reply: str) -> None:
        """The response joins the context: the next warm turn extends the
        cached sequence exactly the way a real agentic session does."""
        self.history += getattr(self, "_pending_turn_text", "") + reply
        self._pending_turn_text = ""
        self.n_turn += 1

    def intended_prompt_tokens(self, prompt: str) -> int:
        return int(len(prompt) / self.cpt)


def make_session(wl, opts, prefixes: Prefixes, uid: int, is_sub: bool,
                 seed: int | None = None) -> Session:
    """Build one session exactly as `user_loop` did: class-specific prefix,
    class-specific log-normal draw, unique context on top of the shared block.

    The RNG is seeded `(seed << 20) ^ uid`, so a given (seed, uid) always
    produces the same session — the determinism the tests pin.
    """
    seed = opts.seed if seed is None else seed
    rng = random.Random((seed << 20) ^ uid)
    cpt = opts.chars_per_token
    median = wl.subagent_median_tokens if is_sub else wl.user_prompt_median_tokens
    sigma = wl.subagent_sigma if is_sub else wl.user_prompt_sigma
    # each request class carries ITS OWN byte-stable prefix, exactly as
    # Workload.sample prices it
    own_sub = is_sub and not wl.sub_shares_prefix
    prefix_tok = wl.subagent_prefix_tokens if own_sub else wl.system_prefix_tokens
    prefix_txt = prefixes.sub if own_sub else prefixes.user
    full = draw_session_tokens(rng, median, sigma, prefix_tok,
                               opts.context_cap_tokens)
    ctx = make_text(rng, max(full - prefix_tok, 0), cpt)
    return Session(uid=uid, is_sub=is_sub, prefix_text=prefix_txt,
                   prefix_tokens=prefix_tok, ctx=ctx, rng=rng, cpt=cpt,
                   warm_turn_tokens=wl.warm_turn_tokens,
                   miss_rate=wl.miss_rate)


def sampler_selfcheck(wl, opts) -> tuple[list[dict], bool]:
    """The harness's dry-run sampler self-check, as data.

    The sampled raw median and log-sd must reproduce the configured
    (median, sigma) — this is the distribution warm capacity and the prefill
    tail are priced on, so the probe must draw it faithfully. Returns
    (rows, ok).
    """
    from .stats import pct
    rng = random.Random(opts.seed)
    rows, ok_all = [], True
    for cls, med, sig, floor in (
            ("user", wl.user_prompt_median_tokens, wl.user_prompt_sigma,
             wl.system_prefix_tokens),
            ("subagent", wl.subagent_median_tokens, wl.subagent_sigma,
             sub_prefix_floor(wl))):
        raw = [rng.lognormvariate(math.log(med), sig) for _ in range(20_000)]
        smp_med = pct(raw, 50)
        logs = [math.log(x) for x in raw]
        mu = sum(logs) / len(logs)
        smp_sig = math.sqrt(sum((x - mu) ** 2 for x in logs) / (len(logs) - 1))
        clipped = [min(max(x, floor), opts.context_cap_tokens) for x in raw]
        ok = (abs(smp_med / med - 1) < 0.03 and abs(smp_sig / sig - 1) < 0.03)
        ok_all &= ok
        rows.append({"class": cls, "median_cfg": med, "median_smp": smp_med,
                     "sigma_cfg": sig, "sigma_smp": smp_sig,
                     "p5": pct(clipped, 5), "p50": pct(clipped, 50),
                     "p95": pct(clipped, 95),
                     "mean": sum(clipped) / len(clipped), "ok": ok})
    return rows, ok_all
