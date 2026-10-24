#!/usr/bin/env python3
"""Naive Backtracking for 3-Colorability — Online Judge submission framework.

Your task: read an undirected graph from stdin and decide whether it can be
colored with 3 colors so that no edge joins two vertices of the same color.
Print the answer (and the first valid coloring found) to stdout.

Only fill in the TODO function below. The I/O parsing and main() are already
provided — do NOT modify them.

Run locally:  python3 three_color.py < input.txt
"""

import sys
from typing import List, Optional, Set, Tuple

NUM_COLORS = 3
UNCOLORED = -1  # sentinel meaning "this vertex has no color yet"


# TODO: Decide whether the graph is 3-colorable using NAIVE BACKTRACKING.
#
# Parameters:
#   n   -- number of vertices; vertices are numbered 0 .. n-1
#   adj -- adjacency list; adj[v] is the set of neighbors of vertex v
#          (the input never contains duplicate edges or self-loops:
#          self-loops are intercepted by main() before this function runs)
#
# Return a tuple (coloring, nodes):
#   coloring -- a list of length n with coloring[v] in {0, 1, 2} giving a
#               valid 3-coloring, or None if no 3-coloring exists
#   nodes    -- the total number of times your recursive function is entered
#
# The search rules are FULLY FIXED so that every correct program prints the
# exact same answer:
#   1. Color the vertices in the fixed order 0, 1, 2, ..., n-1. Write a
#      recursive function dfs(v) meaning "assign a color to vertex v
#      (vertices 0 .. v-1 are already colored)".
#   2. Counting nodes: every entry into dfs counts as ONE node — the very
#      first call dfs(0) counts, and the final successful call dfs(n)
#      (meaning "all vertices colored") also counts. Increment your counter
#      as the first statement of dfs.
#   3. Inside dfs(v): if v == n, every vertex is colored — you are done
#      (success). Otherwise try colors 0, 1, 2 IN THIS ORDER.
#   4. A color c is allowed for v if NO ALREADY-COLORED neighbor of v has
#      color c (uncolored neighbors are checked later, when their turn
#      comes — do not look ahead).
#   5. For the first allowed color, record it and recurse into dfs(v + 1).
#      If that recursion eventually succeeds, propagate the success
#      immediately — the FIRST complete coloring found is the answer.
#      If it fails, UNDO the assignment (restore the "uncolored" sentinel)
#      before trying the next color. Forgetting to undo is the classic
#      backtracking bug: stale colors pollute every later branch.
#   6. If all three colors fail for v, return failure so the caller
#      backtracks (i.e. undoes its own assignment and tries its next color).
#
# Example: the triangle (3 vertices, edges 0-1, 1-2, 2-0) must produce
# coloring [0, 1, 2] with nodes == 4 (dfs(0), dfs(1), dfs(2), dfs(3)).
#
# Hints:
#   - Keep a list `coloring` of length n filled with UNCOLORED, shared by
#     all recursive calls (no copying needed at this scale).
#   - Keep the node counter somewhere the recursive calls can reach — e.g. a
#     one-element list, a dict, or a nonlocal variable.
#   - Worst case the search explores up to 3^n assignments; that exponential
#     blow-up is the point of this exercise (it is an NP-complete problem).
#     The test data stays small enough for the naive method.
#   - Recursion depth can reach n (up to 500); main() has already raised
#     Python's recursion limit for you.
#
# Reference reading:
#   - https://en.wikipedia.org/wiki/Graph_coloring
#   - Russell & Norvig, "Artificial Intelligence: A Modern Approach",
#     Chapter 6 (Constraint Satisfaction Problems) — the backtracking
#     search you are writing is its Figure 6.5 in disguise.
def three_color(n: int, adj: List[Set[int]]) -> Tuple[Optional[List[int]], int]:
    coloring = [UNCOLORED] * n
    nodes = 0

    def dfs(v: int) -> bool:
        nonlocal nodes
        nodes += 1
        if v == n:
            return True
        for c in range(NUM_COLORS):
            if any(coloring[u] == c for u in adj[v]):
                continue
            coloring[v] = c
            if dfs(v + 1):
                return True
            coloring[v] = UNCOLORED
        return False

    if dfs(0):
        return coloring, nodes
    return None, nodes


# ---------------- provided I/O below — do not modify ----------------

def main() -> None:
    # The recursion depth of a correct solution is at most n + 1, but
    # Python's default limit (1000) is close to the maximum n = 500 once
    # the interpreter stack is counted, so raise it to be safe.
    sys.setrecursionlimit(1_000_000)

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, m = int(data[0]), int(data[1])

    adj: List[Set[int]] = [set() for _ in range(n)]
    pos = 2
    for _ in range(m):
        u, v = int(data[pos]), int(data[pos + 1])
        pos += 2
        if u == v:
            # A self-loop can never be colored: the vertex would have to
            # differ from itself. This check is done HERE, in main(), so
            # three_color() never sees a self-loop. No recursive call is
            # made, hence NODES 0.
            sys.stdout.write("UNSAT\nNODES 0\n")
            return
        adj[u].add(v)
        adj[v].add(u)

    coloring, nodes = three_color(n, adj)
    if coloring is None:
        sys.stdout.write(f"UNSAT\nNODES {nodes}\n")
    else:
        sys.stdout.write("SAT\n")
        sys.stdout.write(" ".join(map(str, coloring)) + "\n")
        sys.stdout.write(f"NODES {nodes}\n")


if __name__ == "__main__":
    main()
