"""Style matching between the harmful contrast set and a harmless pool.

The harmless contrast prompts must look like the harmful ones on surface
features that are not the harmful/harmless distinction itself; otherwise a
difference-of-means direction partly encodes length or sentence mood. This
module provides:

* a tokenizer-agnostic length function (regex tokens by default, or any
  callable such as a Hugging Face tokenizer's ``encode``),
* an imperative detector based on a fixed first-word verb list
  (``IMPERATIVE_VERBS``, frozen here so the check is reproducible),
* greedy nearest-length matching with seeded tie-breaks, optionally within
  the same imperative/non-imperative stratum, under one tokenizer or jointly
  under several (sum of the absolute length gaps),
* a seeded swap refinement (``refine_by_swaps``) that, after greedy matching,
  replaces matched pool items with unmatched ones until the token-length SMD
  is within a target under every tokenizer, or no improving swap exists,
* summary statistics: token-length standardized mean difference (SMD), the SMD
  of log token length (report only; lengths are right-skewed) and
  imperative-share difference.

Pure Python plus ``random``; no global side effects.
"""

from __future__ import annotations

import bisect
import math
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

# Frozen first-word verb list used for the imperative-share check. Changing
# this list changes the reported statistic, so it is versioned.
IMPERATIVE_VERBS_VERSION = "v1-2026-10-07"
IMPERATIVE_VERBS: frozenset[str] = frozenset(
    """
    add analyze analyse answer arrange assess build calculate categorize
    categorise change check choose classify combine compare compile compose
    construct convert count craft create debug define demonstrate describe
    design detail determine develop devise discuss draft draw edit elaborate
    enumerate estimate evaluate explain explore extract find fix format
    generate give guide help identify illustrate imagine implement improve
    instruct interpret introduce invent list locate make map name note offer
    organize organise outline paraphrase pick plan predict prepare present
    produce program propose provide publish rank rate recommend rewrite
    rephrase research review revise run select share show simulate sketch
    solve sort state suggest summarize summarise teach tell translate use
    walk write
    """.split()
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[A-Za-z']+")


def regex_token_length(text: str) -> int:
    """Number of regex tokens (word runs and single punctuation marks)."""
    return len(_TOKEN_RE.findall(text))


def make_length_fn(tokenizer: object | None = None) -> Callable[[str], int]:
    """Return a length function.

    ``tokenizer`` may be None (regex tokens), a callable ``str -> int``, or an
    object with an ``encode`` method (Hugging Face tokenizer; special tokens
    are not added).
    """
    if tokenizer is None:
        return regex_token_length
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        def _hf_len(text: str) -> int:
            try:
                return len(encode(text, add_special_tokens=False))
            except TypeError:
                return len(encode(text))
        return _hf_len
    if callable(tokenizer):
        return tokenizer  # type: ignore[return-value]
    raise TypeError("tokenizer must be None, a callable, or expose encode()")


def first_word(text: str) -> str:
    """Lower-cased first alphabetic word of ``text`` ('' if none)."""
    m = _WORD_RE.search(text.strip())
    return m.group(0).lower().strip("'") if m else ""


def is_imperative(text: str, verbs: frozenset[str] = IMPERATIVE_VERBS) -> bool:
    """True if the first word is in the frozen verb list."""
    return first_word(text) in verbs


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs: Sequence[float]) -> float:
    """Sample variance (ddof=1); 0.0 for fewer than two values."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def smd(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardized mean difference (mean_a - mean_b) / pooled SD.

    Pooled SD = sqrt((var_a + var_b) / 2). Returns 0.0 when both groups are
    constant and equal, and +/-inf when they are constant and differ.
    """
    if not a or not b:
        return float("nan")
    diff = mean(a) - mean(b)
    sd = math.sqrt((variance(a) + variance(b)) / 2.0)
    if sd == 0.0:
        return 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    return diff / sd


def log_lengths(xs: Sequence[float]) -> list[float]:
    """Natural log of each length, floored at 1 token (so empty strings do not give -inf)."""
    return [math.log(max(float(x), 1.0)) for x in xs]


def imperative_share(texts: Sequence[str], verbs: frozenset[str] = IMPERATIVE_VERBS) -> float:
    if not texts:
        return float("nan")
    return sum(is_imperative(t, verbs) for t in texts) / len(texts)


@dataclass
class StyleReport:
    n_a: int
    n_b: int
    mean_len_a: float
    mean_len_b: float
    smd_len: float
    abs_smd_len: float
    smd_log_len: float
    imperative_share_a: float
    imperative_share_b: float
    imperative_share_diff: float
    verbs_version: str = IMPERATIVE_VERBS_VERSION

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def style_report(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
    length_fn: Callable[[str], int] = regex_token_length,
    verbs: frozenset[str] = IMPERATIVE_VERBS,
) -> StyleReport:
    """Length SMD, log-length SMD and imperative-share difference between two text groups (a minus b)."""
    la = [float(length_fn(t)) for t in texts_a]
    lb = [float(length_fn(t)) for t in texts_b]
    s = smd(la, lb)
    ia = imperative_share(texts_a, verbs)
    ib = imperative_share(texts_b, verbs)
    return StyleReport(
        n_a=len(la),
        n_b=len(lb),
        mean_len_a=mean(la),
        mean_len_b=mean(lb),
        smd_len=s,
        abs_smd_len=abs(s),
        smd_log_len=smd(log_lengths(la), log_lengths(lb)),
        imperative_share_a=ia,
        imperative_share_b=ib,
        imperative_share_diff=ia - ib,
    )


@dataclass
class MatchResult:
    """Output of :func:`greedy_match`.

    ``pairs[i] = (target_index, pool_index)``, in the order targets were
    processed. ``stratum_fallbacks`` counts targets for which no pool item in
    the same imperative stratum remained and the whole pool was used.
    """

    pairs: list[tuple[int, int]]
    stratum_fallbacks: int = 0
    abs_len_gaps: list[int] = field(default_factory=list)


def greedy_match(
    targets: Sequence[str],
    pool: Sequence[str],
    seed: int,
    length_fn: Callable[[str], int] = regex_token_length,
    match_imperative: bool = True,
    verbs: frozenset[str] = IMPERATIVE_VERBS,
    joint_length_fns: Sequence[Callable[[str], int]] = (),
) -> MatchResult:
    """Greedy one-to-one nearest-length matching without replacement.

    Targets are processed in a seeded random order. For each target the pool
    item(s) with the smallest absolute token-length gap are found; ties are
    broken uniformly at random with the same seeded RNG. With
    ``match_imperative`` the search is restricted to pool items whose
    imperative flag equals the target's, falling back to the full remaining
    pool when that stratum is empty.

    ``joint_length_fns`` (optional) are further length functions (other
    models' tokenizers). The gap is then the sum over ``length_fn`` and every
    joint function of the absolute length gaps, so one matched set is close in
    length under all of them. With no joint functions the result is exactly the
    single-tokenizer matching.
    """
    if len(pool) < len(targets):
        raise ValueError(f"pool too small: {len(pool)} < {len(targets)} targets")
    rng = random.Random(seed)
    fns = [length_fn, *joint_length_fns]
    t_len = [tuple(f(t) for f in fns) for t in targets]
    p_len = [tuple(f(p) for f in fns) for p in pool]
    t_imp = [is_imperative(t, verbs) for t in targets]
    p_imp = [is_imperative(p, verbs) for p in pool]

    # Remaining pool indices bucketed by (imperative flag, length vector) for fast lookup.
    buckets: dict[bool, dict[tuple[int, ...], list[int]]] = {True: {}, False: {}}
    for j, (imp, ln) in enumerate(zip(p_imp, p_len)):
        buckets[imp].setdefault(ln, []).append(j)
    counts = {True: sum(p_imp), False: len(pool) - sum(p_imp)}

    order = list(range(len(targets)))
    rng.shuffle(order)
    result = MatchResult(pairs=[])
    for i in order:
        if match_imperative:
            strata = [t_imp[i]] if counts[t_imp[i]] > 0 else [True, False]
            if counts[t_imp[i]] == 0:
                result.stratum_fallbacks += 1
        else:
            strata = [True, False]
        best_gap: int | None = None
        cands: list[tuple[bool, tuple[int, ...], int]] = []  # (stratum, length vector, position)
        ti = t_len[i]
        for s in strata:
            for ln, idxs in buckets[s].items():
                if not idxs:
                    continue
                gap = sum(abs(a - b) for a, b in zip(ln, ti))
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    cands = [(s, ln, k) for k in range(len(idxs))]
                elif gap == best_gap:
                    cands.extend((s, ln, k) for k in range(len(idxs)))
        if best_gap is None:
            raise RuntimeError("pool exhausted during matching")
        # Deterministic candidate order before the seeded pick.
        cands.sort(key=lambda c: (c[0], c[1], buckets[c[0]][c[1]][c[2]]))
        s, ln, k = cands[rng.randrange(len(cands))]
        j = buckets[s][ln].pop(k)
        counts[s] -= 1
        result.pairs.append((i, j))
        result.abs_len_gaps.append(best_gap)
    return result


# --------------------------------------------------------------------------
# Swap refinement after greedy matching
# --------------------------------------------------------------------------

_KEY_DIGITS = 12  # objective values equal to this many decimals count as ties (broken by the seeded RNG)


def _int_moments(xs: Sequence[int]) -> tuple[int, int, int]:
    return len(xs), sum(xs), sum(x * x for x in xs)


def _smd_from_moments(n_a: int, s1_a: int, s2_a: int, n_b: int, s1_b: int, s2_b: int) -> float:
    """``smd`` computed from integer counts, sums and sums of squares (exact moments, so reproducible)."""
    if n_a < 1 or n_b < 1:
        return float("nan")
    var_a = (s2_a * n_a - s1_a * s1_a) / (n_a * (n_a - 1)) if n_a >= 2 else 0.0
    var_b = (s2_b * n_b - s1_b * s1_b) / (n_b * (n_b - 1)) if n_b >= 2 else 0.0
    diff = s1_a / n_a - s1_b / n_b
    sd = math.sqrt((var_a + var_b) / 2.0)
    if sd == 0.0:
        return 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    return diff / sd


def _objective(smds: Sequence[float]) -> tuple[float, float]:
    """(max |SMD|, sum |SMD|) over tokenizers, rounded; non-finite values count as +inf. Lower is better."""
    a = [abs(x) if math.isfinite(x) else math.inf for x in smds]
    return (round(max(a), _KEY_DIGITS), round(sum(a), _KEY_DIGITS))


@dataclass
class RefineResult:
    """Output of :func:`refine_by_swaps`.

    ``chosen[i]`` is the pool index matched to target ``i`` after refinement.
    ``swaps`` lists ``(target position, removed pool index, added pool index)``
    in the order applied. ``stop_reason`` is ``met`` (every tokenizer's
    |SMD| <= ``target``), ``no_improving_swap`` or ``max_swaps``.
    """

    chosen: list[int]
    target: float
    met: bool
    stop_reason: str
    initial_smd: list[float]
    final_smd: list[float]
    n_swaps: int = 0
    n_swaps_cross_stratum: int = 0
    swaps: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def final_max_abs_smd(self) -> float:
        return max(abs(x) for x in self.final_smd) if self.final_smd else float("nan")


def refine_by_swaps(
    target_lens: Sequence[Sequence[int]],
    pool_lens: Sequence[Sequence[int]],
    chosen: Sequence[int],
    available: Iterable[int],
    seed: int,
    target: float,
    pool_strata: Sequence[object] | None = None,
    allow_cross_stratum: bool = True,
    max_swaps: int | None = None,
) -> RefineResult:
    """Seeded swap refinement of a matched set toward |SMD_t| <= ``target`` under every tokenizer t.

    ``target_lens[t][i]`` and ``pool_lens[t][j]`` are the token lengths of target
    ``i`` and pool item ``j`` under tokenizer ``t``. ``chosen`` holds the pool
    indices currently matched (one per target); ``available`` the pool indices
    that may be swapped in (already style-filtered and not used anywhere else).
    The objective is ``(max_t |SMD_t|, sum_t |SMD_t|)``, compared
    lexicographically.

    Each step ranks the matched items by the objective of the set with that
    item removed (best first; ties in a seeded random order), and for the first
    ranked item whose best replacement strictly improves the objective, swaps it
    for that replacement (ties among equally good replacements broken with the
    seeded RNG). It stops when every |SMD_t| <= ``target``, when no swap
    improves the objective, or after ``max_swaps`` swaps (default
    ``10 * len(chosen)``). Each accepted swap strictly lowers the objective, so
    it terminates; the result is a function of the inputs and ``seed`` only.

    With ``pool_strata`` (for example the imperative flag), swaps first keep the
    removed item's stratum; when that phase stalls and ``allow_cross_stratum``
    is true, a second phase allows any stratum (counted in
    ``n_swaps_cross_stratum``).
    """
    k_tok = len(target_lens)
    if k_tok == 0 or len(pool_lens) != k_tok:
        raise ValueError("target_lens and pool_lens need one entry per tokenizer (at least one)")
    chosen = list(chosen)
    n = len(chosen)
    if any(len(t) != n for t in target_lens):
        raise ValueError("every target_lens entry must have one length per chosen item")
    if len(set(chosen)) != n:
        raise ValueError("chosen pool indices must be distinct")
    avail = sorted(set(available))
    if set(avail) & set(chosen):
        raise ValueError("available pool indices overlap the chosen ones")
    if max_swaps is None:
        max_swaps = 10 * n
    rng = random.Random(seed)
    strata = pool_strata if pool_strata is not None else None

    def vec(j: int) -> tuple[int, ...]:
        return tuple(int(pool_lens[t][j]) for t in range(k_tok))

    def stratum(j: int) -> object:
        return strata[j] if strata is not None else None

    harm = [_int_moments([int(x) for x in target_lens[t]]) for t in range(k_tok)]
    s1 = [sum(int(pool_lens[t][j]) for j in chosen) for t in range(k_tok)]
    s2 = [sum(int(pool_lens[t][j]) ** 2 for j in chosen) for t in range(k_tok)]

    def smds(a1: Sequence[int], a2: Sequence[int], nb: int) -> list[float]:
        return [_smd_from_moments(*harm[t], nb, a1[t], a2[t]) for t in range(k_tok)]

    # unmatched pool items bucketed by (stratum, length vector); indices kept sorted for determinism
    buckets: dict[tuple[object, tuple[int, ...]], list[int]] = {}
    for j in avail:
        buckets.setdefault((stratum(j), vec(j)), []).append(j)

    cur = smds(s1, s2, n)
    initial = list(cur)
    cur_key = _objective(cur)
    result = RefineResult(chosen=chosen, target=float(target), met=False, stop_reason="",
                          initial_smd=initial, final_smd=list(cur))
    phases = ["within", "cross"] if (strata is not None and allow_cross_stratum) else (
        ["within"] if strata is not None else ["any"])

    for phase in phases:
        while True:
            if cur_key[0] <= target:
                result.met, result.stop_reason = True, "met"
                break
            if result.n_swaps >= max_swaps:
                result.stop_reason = "max_swaps"
                break
            # rank removals: objective of the set without each matched item
            positions = list(range(n))
            rng.shuffle(positions)
            removal_keys = {}
            for p in positions:
                v = vec(chosen[p])
                removal_keys[p] = _objective(smds([s1[t] - v[t] for t in range(k_tok)],
                                                  [s2[t] - v[t] * v[t] for t in range(k_tok)], n - 1))
            positions.sort(key=lambda p: removal_keys[p])  # stable: ties keep the seeded order
            accepted = False
            for p in positions:
                x = chosen[p]
                xv, xs = vec(x), stratum(x)
                best_key: tuple[float, float] | None = None
                best_cells: list[tuple[object, tuple[int, ...]]] = []
                for cell, idxs in buckets.items():
                    if not idxs or (phase == "within" and cell[0] != xs):
                        continue
                    yv = cell[1]
                    key = _objective(smds([s1[t] - xv[t] + yv[t] for t in range(k_tok)],
                                          [s2[t] - xv[t] * xv[t] + yv[t] * yv[t] for t in range(k_tok)], n))
                    if best_key is None or key < best_key:
                        best_key, best_cells = key, [cell]
                    elif key == best_key:
                        best_cells.append(cell)
                if best_key is None or not best_key < cur_key:
                    continue
                cands = sorted(j for c in best_cells for j in buckets[c])
                y = cands[rng.randrange(len(cands))]
                ycell = (stratum(y), vec(y))
                buckets[ycell].remove(y)
                bisect.insort(buckets.setdefault((xs, xv), []), x)
                chosen[p] = y
                yv = ycell[1]
                for t in range(k_tok):
                    s1[t] += yv[t] - xv[t]
                    s2[t] += yv[t] * yv[t] - xv[t] * xv[t]
                cur = smds(s1, s2, n)
                cur_key = best_key
                result.n_swaps += 1
                if ycell[0] != xs:
                    result.n_swaps_cross_stratum += 1
                result.swaps.append((p, x, y))
                accepted = True
                break
            if not accepted:
                result.stop_reason = "no_improving_swap"
                break
        if result.met or result.stop_reason == "max_swaps":
            break
    result.final_smd = list(cur)
    return result
