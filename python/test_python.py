import numpy as np
import time
from . import python_practice as pp
# Use unified helpers
from common.test_utils import _pass, _fail


def test_python():
    print("\nTesting Python Problems:")
    counter = 1
    # Problem 1
    arr = [1, 2, 3, 4, 5]
    target = 3
    try:
        result = pp.problem1(arr, target)
        assert result == 2
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 2
    arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    try:
        result = pp.problem2(arr)
        assert result == 6
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 3
    graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
    try:
        result = pp.problem3(graph, 'A')
        assert result == ['A', 'B', 'D', 'C']
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 4
    head = [1, 2, 3, 4, 5]
    try:
        result = pp.problem4(head)
        assert result == [5, 4, 3, 2, 1]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 5
    try:
        result = pp.problem5(5)
        assert result == 120
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 7
    arr = [1, 2, 3, 4, 5]
    target = 5
    try:
        indices = pp.problem7(arr, target)
        assert indices == [1, 3]  # 2 + 3 = 5
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 8
    arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
    k = 4
    try:
        result = pp.problem8(arr, k)
        assert result == 39  # 4 + 2 + 10 + 23
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 9
    try:
        result = pp.problem9(10)
        assert result == 55  # Fibonacci of 10
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 10
    try:
        result = pp.problem10("racecar")
        assert result == True
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 11
    arr = [3, 41, 52, 26, 38, 57, 9, 49]
    try:
        result = pp.problem11(arr.copy())
        assert result == sorted(arr)
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 12
    arr = [3, 41, 52, 26, 38, 57, 9, 49]
    try:
        result = pp.problem12(arr.copy())
        assert result == sorted(arr)
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 13
    # Create a simple binary tree for testing
    root = pp.TreeNode(1)
    root.left = pp.TreeNode(2)
    root.right = pp.TreeNode(3)
    root.left.left = pp.TreeNode(4)
    root.left.right = pp.TreeNode(5)
    try:
        inorder = pp.problem13_inorder(root)
        assert inorder == [4, 2, 5, 1, 3]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 14
    s1 = "abc"
    s2 = "def"
    try:
        length = pp.problem14(s1, s2)
        assert length == 0  # No common subsequence
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 15
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    try:
        result = pp.problem15(weights, values, capacity)
        assert result == 9  # 4 + 5
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 16
    try:
        t0 = time.perf_counter()
        out = pp.problem16_topk_frequent([1, 1, 1, 2, 2, 3], 2)
        assert set(out) == {1, 2}
        dt = time.perf_counter() - t0
        print(f"⏱ {dt:.3f}s")
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 17
    try:
        out = pp.problem17_kth_largest([3, 2, 1, 5, 6, 4], 2)
        assert out == 5
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 18
    try:
        graph = {'A': [('B', 1), ('C', 4)], 'B': [
            ('C', 2), ('D', 5)], 'C': [('D', 1)], 'D': []}
        out = pp.problem18_dijkstra(graph, 'A')
        assert out == {'A': 0, 'B': 1, 'C': 3, 'D': 4}
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 19
    try:
        assert pp.problem19_course_schedule(2, [(1, 0)]) is True
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 20
    try:
        grid = [['1', '1', '0', '0'], [
            '1', '0', '0', '1'], ['0', '0', '1', '1']]
        assert pp.problem20_num_islands(grid) == 3
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 21
    try:
        edges = [[1, 2], [1, 3], [2, 3]]
        assert pp.problem21_redundant_connection(edges) == [2, 3]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 22
    try:
        temps = [73, 74, 75, 71, 69, 72, 76, 73]
        assert pp.problem22_daily_temperatures(
            temps) == [1, 1, 4, 2, 1, 1, 0, 0]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 23
    try:
        assert pp.problem23_largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 24
    try:
        out = pp.problem24_merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
        assert out == [[1, 6], [8, 10], [15, 18]]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 25
    try:
        out = pp.problem25_insert_interval([[1, 3], [6, 9]], [2, 5])
        assert out == [[1, 5], [6, 9]]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 26
    try:
        out = pp.problem26_meeting_rooms_ii([[0, 30], [5, 10], [15, 20]])
        assert out == 2
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 27
    try:
        assert pp.problem27_length_longest_substring("abcabcbb") == 3
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 28
    try:
        assert pp.problem28_char_replacement("AABABBA", 1) == 4
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 29
    try:
        assert pp.problem29_search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 30
    try:
        out = pp.problem30_first_last_pos([5, 7, 7, 8, 8, 10], 8)
        assert out == [3, 4]
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 31
    try:
        assert pp.problem31_koko([3, 6, 7, 11], 8) == 4
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 32
    try:
        assert pp.problem32_subarray_sum([1, 1, 1], 2) == 2
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 33
    try:
        mat = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5],
        ]
        out = pp.problem33_matrix_sum_region(mat, 2, 1, 4, 3)  # expect 8
        assert out == 8
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 34
    try:
        subs = pp.problem34_subsets([1, 2, 3])
        assert len(subs) == 8 and [1, 2] in subs
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 35
    try:
        perms = pp.problem35_permutations([1, 2, 3])
        assert len(perms) == 6 and [1, 2, 3] in perms
        _pass(counter, "Python")
        counter += 1
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(counter, "Python", e)
        counter += 1

    # Problem 36 (LRUCache)
    try:
        n = 36
        c = pp.LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        assert c.get(1) == 1
        c.put(3, 3)   # evicts 2
        assert c.get(2) == -1
        c.put(4, 4)   # evicts 1
        assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(36, "Python", e)

    # Problem 37 (LFUCache)
    try:
        n = 37
        c = pp.LFUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        assert c.get(1) == 1         # freq(1)=2
        c.put(3, 3)                  # evicts key 2 (freq=1)
        assert c.get(2) == -1 and c.get(3) == 3
        c.put(4, 4)                  # evict LFU among {1,3}
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(37, "Python", e)

    # Problem 38 (KMP)
    try:
        n = 38
        assert pp.problem38_kmp_search("abxabcabcaby", "abcaby") == 6
        assert pp.problem38_kmp_search("aaaaa", "bba") == -1
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(38, "Python", e)

    # Problem 39 (Rabin–Karp)
    try:
        n = 39
        assert pp.problem39_rabin_karp("the quick brown fox", "quick") == 4
        assert pp.problem39_rabin_karp("aaaab", "aaab") == 1
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(39, "Python", e)

    # Problem 40 (MST Kruskal)
    try:
        n = 40
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 4)]
        assert pp.problem40_mst_kruskal(3, edges) == 3
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(40, "Python", e)

    # Problem 41 (Floyd–Warshall)
    try:
        n = 41
        INF = 10**9
        dist = [[0, 3, INF], [INF, 0, 2], [1, INF, 0]]
        out = pp.problem41_floyd_warshall(dist)
        assert out[0][2] == 5 and out[2][1] == 4
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(41, "Python", e)

    # Problem 42 (0–1 BFS)
    try:
        n = 42
        grid = [[0, 0, 1], [1, 0, 0], [1, 1, 0]]
        assert pp.problem42_zero_one_bfs(grid) == 1
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(42, "Python", e)

    # Problem 43 (Segment Tree)
    try:
        n = 43
        st = pp.SegmentTree([1, 3, 5])
        assert st.query(0, 2) in (9,)
        st.update(1, +2)  # [1,5,5]
        assert st.query(0, 2) in (11,)
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(43, "Python", e)

    # Problem 44 (Fenwick)
    try:
        n = 44
        ft = pp.Fenwick(5)
        ft.add(1, 3)
        ft.add(3, 2)
        assert ft.prefix(3) in (5,)
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(44, "Python", e)

    # Problem 45 (Max overlap)
    try:
        n = 45
        intervals = [(1, 5), (2, 6), (4, 7), (8, 10)]
        assert pp.problem45_max_overlap(intervals) in (3,)
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(45, "Python", e)

    # Problem 46 (Min window)
    try:
        n = 46
        assert pp.problem46_min_window("ADOBECODEBANC", "ABC") == "BANC"
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(46, "Python", e)

    # Problem 47 (LIS length)
    try:
        n = 47
        assert pp.problem47_lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(47, "Python", e)

    # Problem 48 (Edit distance)
    try:
        n = 48
        assert pp.problem48_edit_distance("horse", "ros") == 3
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(48, "Python", e)

    # Problem 49 (Count paths in DAG)
    try:
        n = 49
        n_nodes = 5
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
        assert pp.problem49_count_paths_dag(n_nodes, edges, 0, 4) == 2
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(49, "Python", e)

    # Problem 50 (Topo order)
    try:
        n = 50
        n_nodes = 4
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        order = pp.problem50_topo_order(n_nodes, edges)
        pos = {v: i for i, v in enumerate(order)}
        assert pos[0] < pos[1] and pos[0] < pos[2] and pos[1] < pos[3] and pos[2] < pos[3]
        _pass(n, "Python")
    except NotImplementedError:
        pass
    except Exception as e:
        _fail(50, "Python", e)
