# LeetCode Speedrun & Offline Answer Verification (Python Edition)

A concise, **pattern‑first** study plan to cover the majority of interview‑style coding questions—plus optional **offline** checks to critique or validate your solutions locally. All code examples are in **Python 3.10+** using only the standard library.

---

## 0) Mindset & Method

- **Solve by invariant**: state the loop invariant / feasible predicate *before* coding.
- **Prove complexity**: time & space in Big‑O; note monotonicity or amortized behavior.
- **Edge cases first**: empty, single element, all equal, negatives/zeros, duplicates, extremes.
- **When stuck**: shrink (2–3 elements), write inequalities/recurrences, look for monotonicity or substructure.

---

## 1) The 12 Patterns You Must Master

1. **Two pointers** — same/opposite direction; dedup, partition, palindromes.
2. **Sliding window** — fixed size; variable; **at‑most K** → **exactly K** via difference.
3. **Prefix sums / hashing** — subarray sums, counts, modulo classes; 2‑sum/3‑sum.
4. **Monotonic stack/deque** — next greater/smaller; histogram; sliding max (deque).
5. **Binary search** — index; **on answer** using a monotone predicate.
6. **Graph BFS/DFS** — grids; shortest path (unweighted via BFS); components.
7. **Topological sort** — Kahn’s algo; DFS order; cycle detection.
8. **Union–Find (DSU)** — connectivity, cycle detection, Kruskal‑style merges.
9. **Heaps & greedy** — k‑best, scheduling, interval rooms; min/max‑heap.
10. **Intervals** — merge/insert; sweep line for overlaps.
11. **Dynamic programming** — 1D/2D, knapsack family, LIS (n log n), edit distance.
12. **Bit tricks** — parity, masks, subset iteration.

> If you can map a problem to one pattern in ~30 seconds, you’re 80% done.

---

## 2) Minimal Python Templates (Paste & Adapt)

### Two Pointers (in‑place partition)
```python
def partition(a, pred):
    i = 0
    for j, v in enumerate(a):
        if pred(v):
            a[i], a[j] = a[j], a[i]
            i += 1
    return i  # first index of the 'false' region
```

### Sliding Window (at‑most K distinct; exactly K via difference)
```python
from collections import Counter

def at_most_k(s: str, k: int) -> int:
    cnt, i, ans = Counter(), 0, 0
    for j, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:
            cnt[s[i]] -= 1
            if cnt[s[i]] == 0: cnt.pop(s[i])
            i += 1
        ans += j - i + 1
    return ans

def exactly_k(s: str, k: int) -> int:
    return at_most_k(s, k) - at_most_k(s, k - 1)
```

### Prefix Sum / Hashing (subarray sum == k)
```python
def subarray_sum(nums: list[int], k: int) -> int:
    seen = {0: 1}
    s = ans = 0
    for v in nums:
        s += v
        ans += seen.get(s - k, 0)
        seen[s] = seen.get(s, 0) + 1
    return ans
```

### Monotonic Stack (next smaller to left)
```python
def next_smaller_left(a: list[int]) -> list[int]:
    st, res = [], [-1] * len(a)
    for i, v in enumerate(a):
        while st and a[st[-1]] >= v:
            st.pop()
        res[i] = st[-1] if st else -1
        st.append(i)
    return res
```

### Largest Rectangle in Histogram (full pattern)
```python
def largest_rectangle_area(heights: list[int]) -> int:
    st = []  # stack of indices with increasing heights
    ans = 0
    for i, h in enumerate(heights + [0]):
        while st and heights[st[-1]] > h:
            H = heights[st.pop()]
            L = st[-1] if st else -1
            ans = max(ans, H * (i - L - 1))
        st.append(i)
    return ans
```

### Binary Search on Answer (helper + Koko example)
```python
def binary_search_answer(lo: int, hi: int, feasible) -> int:
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def min_eating_speed(piles: list[int], h: int) -> int:
    def ok(k: int) -> bool:
        return sum((p + k - 1) // k for p in piles) <= h
    return binary_search_answer(1, max(piles), ok)
```

### Graph BFS (grid shortest path)
```python
from collections import deque

def bfs_grid(grid: list[list[int]], sr: int, sc: int) -> list[list[int]]:
    R, C = len(grid), len(grid[0]) if grid else 0
    dist = [[-1] * C for _ in range(R)]
    q = deque()
    if 0 <= sr < R and 0 <= sc < C and grid[sr][sc]:
        dist[sr][sc] = 0
        q.append((sr, sc))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist
```

### Topological Sort (Kahn’s)
```python
from collections import defaultdict, deque

def topo(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    g, indeg = defaultdict(list), [0] * n
    for u, v in edges:
        g[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for w in g[u]:
            indeg[w] -= 1
            if indeg[w] == 0: q.append(w)
    return order if len(order) == n else None
```

### Union–Find (DSU)
```python
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n)); self.r = [0] * n
    def find(self, x: int) -> int:
        while x != self.p[x]:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1
        return True
```

### Intervals (merge)
```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort()
    out = []
    for s, e in intervals:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out
```

### Heap (k‑way merge)
```python
import heapq

def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    h, out = [], []
    for i, L in enumerate(lists):
        if L: heapq.heappush(h, (L[0], i, 0))
    while h:
        v, i, j = heapq.heappop(h)
        out.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(h, (lists[i][j + 1], i, j + 1))
    return out
```

### DP: LIS (n log n)
```python
import bisect

def lis_length(a: list[int]) -> int:
    tails = []
    for x in a:
        i = bisect.bisect_left(tails, x)
        if i == len(tails): tails.append(x)
        else: tails[i] = x
    return len(tails)
```

---

## 3) A 10‑Day Speedrun (90–120 minutes/day)

- **Day 1 – Arrays & Two Pointers:** sorted two‑sum, Dutch flag, palindrome, dedup.
- **Day 2 – Sliding Windows:** fixed, at‑most, exactly‑K; anagrams; longest substring.
- **Day 3 – Prefix Sums/Hashing:** subarray sum==k; equal 0/1; modulo classes.
- **Day 4 – Monotonic Stack/Deque:** histogram area; daily temperatures; sliding max.
- **Day 5 – Binary Search:** index search; **on answer** (Koko; ship capacity).
- **Day 6 – Graph BFS/DFS:** islands; shortest path in binary matrix; word ladder.
- **Day 7 – Topo + DSU:** course schedule; redundant connection; Kruskal sketch.
- **Day 8 – Heaps & Greedy:** meeting rooms II; task scheduling; k‑lists merge.
- **Day 9 – Intervals:** merge/insert; min arrows; sweep line basics.
- **Day 10 – DP Core:** LIS (n log n); edit distance; 0/1 knapsack; house robber.

**Daily loop:** pick 3–5 problems → state invariant & edges → code in 15–20 min → add 2–3 tests → write 1–2 line complexity/correctness note.

---

## 4) Offline Answer Checking (Two Routes)

### A) Property‑based tests (no LLM, fully offline)
Install Hypothesis and generate fuzz tests that cross‑check against a slow oracle.

```bash
pip install hypothesis pytest
```

**Example (Kadane vs. brute force)**
```python
# tools/property_checks.py
from hypothesis import given, strategies as st
import python.python_practice as pp

@given(st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=30))
def test_max_subarray(lst):
    def brute(a):
        m = -10**9
        for i in range(len(a)):
            s = 0
            for j in range(i, len(a)):
                s += a[j]
                m = max(m, s)
        return m
    assert pp.problem_max_subarray(lst) == brute(lst)
```

Run:
```bash
pytest -q
```

### B) Local LLM “critique” (after a one‑time model pull)
Run a small model with **Ollama** (or `llama.cpp`) and request a **JSON‑only** verdict (no prose) with complexity & missing tests.

**Install & pull once (then offline):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b   # or mistral:7b
```

**Minimal checker:**
```python
# tools/llm_checker.py
import json, os, requests, inspect
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
SYSTEM = "You are a strict code reviewer. Output ONLY compact JSON. No prose."
TEMPLATE = """Task: check a candidate solution.\nReturn JSON with fields: verdict, time_complexity, space_complexity, issues, missing_tests.\nProblem: {prompt}\n\n```python\n{source}\n```\n\nTest log:\n{test_log}\n"""

def check(func, prompt: str, test_log: str = "") -> dict:
    try:
        src = inspect.getsource(func)
    except OSError:
        src = f"# source unavailable: {func.__name__}"
    payload = {
        "model": MODEL,
        "prompt": SYSTEM + "\n\n" + TEMPLATE.format(
            prompt=prompt.strip(),
            source=src.strip(),
            test_log=(test_log or "N/A")[:4000],
        ),
        "options": {"temperature": 0.2},
        "stream": False,
    }
    r = requests.post(LLM_URL, json=payload, timeout=60)
    r.raise_for_status()
    txt = r.json().get("response", "").strip()
    start, end = txt.find("{"), txt.rfind("}")
    return json.loads(txt[start:end+1]) if start != -1 else {"verdict": "parse_error", "raw": txt}
```

**Hook into your tests (opt‑in):**
```python
# practice_tests.py (snippet)
import os
USE_LLM = os.getenv("LLM_CHECK", "0") == "1"
if USE_LLM:
    from tools.llm_checker import check
    res = check(
        pp.problem23_largest_rectangle,
        prompt="Largest rectangle in histogram; return max area.",
        test_log="basic case [2,1,5,6,2,3] -> 10",
    )
    print("LLM verdict:", res.get("verdict"),
          "| issues:", res.get("issues"),
          "| add tests:", res.get("missing_tests"))
```

**Run:**
```bash
# terminal 1
ollama serve
# terminal 2
LLM_CHECK=1 python practice_tests.py
```

**Caveats:** Treat the LLM verdict as advisory; your unit/property tests remain the source of truth.

---

## 5) Final Checklist (Before an Interview)

- Map a new problem to **one** pattern in 30–60 seconds.
- State the **invariant/feasible predicate** clearly.
- Prove **time/space** complexity in two sentences.
- Recall 3–5 **edge cases** per pattern.
- Recently practiced: **binary search on answer**, **monotonic stack**, **topo/DSU**.
- Can implement **LIS (n log n)**, **edit distance**, **merge intervals** from memory.

---

## 6) References (authoritative)

- Cormen et al., *Introduction to Algorithms (CLRS)* — invariants & proofs.
- Kleinberg & Tardos, *Algorithm Design* — greedy proofs & exchange arguments.
- Dasgupta, Papadimitriou, Vazirani, *Algorithms* — clean graph DP & reductions.
- CP‑Algorithms (e‑maxx), VisuAlgo — succinct templates & visuals.

---

**Tip:** In each solution file, add this 3‑line header:
1) **Pattern:** (e.g., sliding window, topo)
2) **Invariant/Feasible predicate:** (one sentence)
3) **Complexity:** time/space

This mirrors how interviewers evaluate reasoning and keeps you honest about correctness.

