"""Exact rational checks for the v3 candidate task-family seeds.

This is a research diagnostic, not a benchmark owner. It verifies only the
distributional claims made in docs/task-family-seeds-v3.md; it does not sample,
does not update beliefs, and does not implement the hard-do owner.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product

BINARY = (0, 1)


def _marginal(joint: dict[tuple[int, ...], F], keep: tuple[int, ...]) -> dict[tuple[int, ...], F]:
    out: dict[tuple[int, ...], F] = {}
    for assignment, mass in joint.items():
        key = tuple(assignment[i] for i in keep)
        out[key] = out.get(key, F(0)) + mass
    return out


def _tv(p: dict[tuple[int, ...], F], q: dict[tuple[int, ...], F]) -> F:
    keys = set(p) | set(q)
    return sum(abs(p.get(key, F(0)) - q.get(key, F(0))) for key in keys) / 2


def _check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}: {detail}")


def _tq_cb_joint(
    kind: str, sign: int, e: F, target: str | None = None, value: int | None = None
) -> dict[tuple[int, int, int], F]:
    """Assignment order (X, Y, S). Kind F: X->Y, S->Y. Kind R: Y->X, S->X."""
    out: dict[tuple[int, int, int], F] = {}
    for x, y, s in product(BINARY, repeat=3):
        if target == "X" and x != value:
            out[(x, y, s)] = F(0)
            continue
        if target == "Y" and y != value:
            out[(x, y, s)] = F(0)
            continue
        if target == "S" and s != value:
            out[(x, y, s)] = F(0)
            continue

        mass = F(1)
        if kind == "F":
            if target != "X":
                mass *= F(1, 2)
            if target != "S":
                mass *= F(1, 2)
            if target != "Y":
                if s == 1:
                    base = F(1, 2) + sign * e if x == 1 else F(1, 2) - sign * e
                else:
                    base = F(1, 2) - sign * e if x == 1 else F(1, 2) + sign * e
                mass *= base if y == 1 else 1 - base
        else:
            if target != "Y":
                mass *= F(1, 2)
            if target != "S":
                mass *= F(1, 2)
            if target != "X":
                if s == 1:
                    base = F(1, 2) + sign * e if y == 1 else F(1, 2) - sign * e
                else:
                    base = F(1, 2) - sign * e if y == 1 else F(1, 2) + sign * e
                mass *= base if x == 1 else 1 - base
        out[(x, y, s)] = mass
    return out


def verify_tq_cb(e: F) -> None:
    worlds = {
        key: _tq_cb_joint(kind, sign, e)
        for kind, sign, key in (("F", 1, "F+"), ("F", -1, "F-"), ("R", 1, "R+"), ("R", -1, "R-"))
    }
    _check(
        "TQ-CB",
        worlds["F+"] == worlds["R+"] and worlds["F-"] == worlds["R-"],
        f"observational pair equality for e={e}",
    )
    do_x1 = {
        key: _tq_cb_joint(kind, sign, e, "X", 1)
        for kind, sign, key in (("F", 1, "F+"), ("F", -1, "F-"), ("R", 1, "R+"), ("R", -1, "R-"))
    }
    half = {(0,): F(1, 2), (1,): F(1, 2)}
    _check(
        "TQ-CB",
        all(_marginal(do_x1[key], (1,)) == half for key in do_x1),
        f"do(X=1) Y-only is Bern(1/2) in all worlds for e={e}",
    )
    _check(
        "TQ-CB",
        all(_marginal(do_x1[key], (2,)) == half for key in do_x1),
        f"do(X=1) S-only is Bern(1/2) in all worlds for e={e}",
    )
    _check("TQ-CB", _tv(do_x1["F+"], do_x1["F-"]) == 2 * e, "TV(F+,F-) under do(X=1) == 2e")
    _check("TQ-CB", _tv(do_x1["F+"], do_x1["R+"]) == e, "TV(F+,R+) under do(X=1) == e")
    _check("TQ-CB", _tv(do_x1["F+"], do_x1["R-"]) == e, "TV(F+,R-) under do(X=1) == e")


def _path_parents(edges: tuple[tuple[int, int], ...], n: int) -> dict[int, list[int]]:
    parents: dict[int, list[int]] = {i: [] for i in range(1, n + 1)}
    for parent, child in edges:
        parents[child].append(parent)
    return parents


def _path_joint(
    edges: tuple[tuple[int, int], ...],
    n: int,
    e: F,
    target: int | None = None,
    value: int | None = None,
) -> dict[tuple[int, ...], F]:
    parents = _path_parents(edges, n)
    out: dict[tuple[int, ...], F] = {}
    for assignment in product(BINARY, repeat=n):
        mass = F(1)
        for node, node_value in enumerate(assignment, start=1):
            if target == node:
                if node_value != value:
                    mass = F(0)
                    break
                continue
            if not parents[node]:
                mass *= F(1, 2)
            else:
                parent_value = assignment[parents[node][0] - 1]
                base = F(1, 2) + e if parent_value == 1 else F(1, 2) - e
                mass *= base if node_value == 1 else 1 - base
        out[assignment] = mass
    return out


def verify_ad_mec3(e: F) -> None:
    edges = {
        # Assignment order is (X, M, Y).
        "C": ((1, 2), (2, 3)),  # X -> M -> Y
        "F": ((2, 1), (2, 3)),  # X <- M -> Y
        "C'": ((3, 2), (2, 1)),  # Y -> M -> X
    }
    worlds = {name: _path_joint(edge_list, 3, e) for name, edge_list in edges.items()}
    _check(
        "AD-MEC3",
        worlds["C"] == worlds["F"] == worlds["C'"],
        f"observational equality for e={e}",
    )
    means = {
        name: _marginal(_path_joint(edge_list, 3, e, 1, 1), (1,)).get((1,), F(0))
        for name, edge_list in edges.items()
    }
    _check(
        "AD-MEC3",
        means["C"] == F(1, 2) + e and means["F"] == means["C'"] == F(1, 2),
        f"do(X=1) M separates C from F/C' for e={e}",
    )
    means_y = {
        name: _marginal(_path_joint(edge_list, 3, e, 3, 1), (1,)).get((1,), F(0))
        for name, edge_list in edges.items()
    }
    _check(
        "AD-MEC3",
        means_y["C'"] == F(1, 2) + e and means_y["F"] == F(1, 2),
        f"do(Y=1) M separates C' from F for e={e}",
    )


def verify_ad_path4(e: F) -> None:
    edges = {
        "O1": ((1, 2), (2, 3), (3, 4)),
        "O2": ((2, 1), (2, 3), (3, 4)),
        "O3": ((2, 1), (3, 2), (3, 4)),
        "O4": ((2, 1), (3, 2), (4, 3)),
    }
    worlds = {name: _path_joint(edge_list, 4, e) for name, edge_list in edges.items()}
    _check(
        "AD-PATH4",
        all(world == worlds["O1"] for world in worlds.values()),
        f"observational equality for e={e}",
    )
    m1 = {
        name: _marginal(_path_joint(edge_list, 4, e, 1, 1), (1,)).get((1,), F(0))
        for name, edge_list in edges.items()
    }
    _check(
        "AD-PATH4",
        m1["O1"] == F(1, 2) + e and all(m1[name] == F(1, 2) for name in ("O2", "O3", "O4")),
        f"node 1 separates O1 for e={e}",
    )
    m2 = {
        name: _marginal(_path_joint(edge_list, 4, e, 2, 1), (2,)).get((1,), F(0))
        for name, edge_list in edges.items()
    }
    _check(
        "AD-PATH4",
        m2["O2"] == F(1, 2) + e and all(m2[name] == F(1, 2) for name in ("O3", "O4")),
        f"node 2 separates O2 from O3/O4 for e={e}",
    )
    m3 = {
        name: _marginal(_path_joint(edge_list, 4, e, 4, 1), (2,)).get((1,), F(0))
        for name, edge_list in edges.items()
    }
    _check(
        "AD-PATH4",
        m3["O4"] == F(1, 2) + e and m3["O3"] == F(1, 2),
        f"node 4 separates O4 from O3 for e={e}",
    )


def _tq_s_joint(
    world: str, q: F, target_t: bool = False, value: int | None = None
) -> dict[tuple[int, int, int], F]:
    """TQ-S / CD-S worlds. Assignment order (T, S, Y)."""
    out: dict[tuple[int, int, int], F] = {}
    for t, s, y in product(BINARY, repeat=3):
        if target_t:
            if t != value:
                out[(t, s, y)] = F(0)
                continue
        else:
            mass_t = F(1, 2)
        mass_s = F(1, 2)
        if not target_t:
            mass = mass_t * mass_s
        else:
            mass = mass_s

        if world == "W0":
            base = F(1, 2) + q / 2 if t == 1 else F(1, 2) - q / 2
        elif world == "W1":
            base = F(1, 2) - q / 2 if t == 1 else F(1, 2) + q / 2
        elif world == "W2":
            if t == 1:
                base = F(1, 2)
            else:
                base = F(1, 2) - q if s == 0 else F(1, 2) + q
        else:
            if t == 1:
                base = F(1, 2)
            else:
                base = F(1, 2) + q if s == 0 else F(1, 2) - q
        mass *= base if y == 1 else 1 - base
        out[(t, s, y)] = mass
    return out


def verify_tq_s(q: F) -> None:
    worlds = {name: _tq_s_joint(name, q) for name in ("W0", "W1", "W2", "W3")}
    do_t1 = {name: _tq_s_joint(name, q, True, 1) for name in worlds}
    m1 = {name: _marginal(do_t1[name], (2,)).get((1,), F(0)) for name in worlds}
    _check(
        "TQ-S",
        m1["W0"] == F(1, 2) + q / 2 and m1["W1"] == F(1, 2) - q / 2,
        f"do(T=1) Y separates W0/W1 for q={q}",
    )
    _check(
        "TQ-S",
        m1["W2"] == m1["W3"] == F(1, 2),
        f"do(T=1) Y keeps W2/W3 ambiguous for q={q}",
    )
    _check("TQ-S", _tv(do_t1["W0"], do_t1["W2"]) == q / 2, "stage-1 TV == q/2")
    do_t0 = {name: _tq_s_joint(name, q, True, 0) for name in worlds}
    _check("TQ-S", _tv(do_t0["W2"], do_t0["W3"]) == 2 * q, "stage-2 TV(W2,W3) == 2q")


def verify_ad_auto_pairwise_separation(n: int, e: F) -> None:
    """AD-AUTO L0/L1 feasibility: every one-edge world pair is separable by one action."""
    edges = [(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
    actions = [
        (target, value, measure)
        for target in range(1, n + 1)
        for value in BINARY
        for measure in range(1, n + 1)
        if measure != target
    ]
    min_max_tv = min(
        max(
            _tv(
                _marginal(_path_joint((e1,), n, e, target, value), (measure - 1,)),
                _marginal(_path_joint((e2,), n, e, target, value), (measure - 1,)),
            )
            for target, value, measure in actions
        )
        for idx, e1 in enumerate(edges)
        for e2 in edges[idx + 1 :]
    )
    _check("AD-AUTO", min_max_tv == e, f"N={n}: min over world pairs of max-action TV == e={e}")


def main() -> None:
    for e in (F(2, 5), F(1, 5), F(1, 20)):
        verify_tq_cb(e)
    for e in (F(2, 5), F(1, 5), F(1, 20)):
        verify_ad_mec3(e)
    for e in (F(2, 5), F(1, 5), F(1, 20)):
        verify_ad_path4(e)
    for q in (F(1, 2), F(1, 4), F(1, 20)):
        verify_tq_s(q)
    for e in (F(1, 4),):
        verify_ad_auto_pairwise_separation(3, e)
        verify_ad_auto_pairwise_separation(4, e)
    print("ALL V3/V4 SEED MATH CHECKS PASSED")


if __name__ == "__main__":
    main()
