# Python Algorithms Study Guide

This guide covers fundamental Python algorithms and data structures commonly tested in technical interviews, with focus on robotics and ML applications. Robotics interviews often test problem-solving skills, algorithmic thinking, and Python proficiency.

## Key Concepts

### Binary Search
- Efficient search in sorted arrays
- Time: O(log n), Space: O(1)
- Returns index or -1 if not found

### Kadane's Algorithm
- Maximum subarray sum
- Handles negative numbers
- Time: O(n), Space: O(1)

### DFS Traversal
- Depth-first search on graphs
- Uses stack (recursive or iterative)
- Returns visited nodes in order

### Linked List Reversal
- Reverse singly linked list
- Iterative: 3 pointers, Recursive: stack-based

### Factorial
- Recursive function
- Base case: n <= 1
- Stack overflow risk for large n

### Tower of Hanoi
- Recursive puzzle solution
- 2^n - 1 moves
- Demonstrates recursion

## Interview-Ready Concepts

### Time/Space Complexity Analysis
- Big O notation
- Best/Average/Worst case scenarios
- Trade-offs between different approaches

### Common Patterns
- Two pointers (for arrays, strings)
- Sliding window
- Divide and conquer
- Dynamic programming
- Greedy algorithms

### Data Structures
- Lists, tuples, sets, dictionaries
- Heaps, stacks, queues
- Trees, graphs
- Hash tables

## Worked Examples

### Problem 1: Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Test
arr = [1, 3, 5, 7, 9]
print(binary_search(arr, 5))  # 2
```

### Problem 2: Kadane's Algorithm
```python
def max_subarray_sum(arr):
    if not arr:
        return 0
    
    max_current = max_global = arr[0]
    for num in arr[1:]:
        max_current = max(num, max_current + num)
        max_global = max(max_global, max_current)
    return max_global

# Test
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray_sum(arr))  # 6 (subarray [4, -1, 2, 1])
```

### Problem 3: DFS Traversal
```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# Test
graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
print(dfs(graph, 'A'))  # ['A', 'B', 'D', 'C']
```

### Problem 4: Reverse Linked List
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

# Test
# Create list: 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
reversed_head = reverse_list(head)
# Print: 5 -> 4 -> 3 -> 2 -> 1
```

### Problem 5: Recursive Factorial
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test
print(factorial(5))  # 120
```

### Problem 6: Tower of Hanoi
```python
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n-1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n-1, auxiliary, target, source)

# Test
tower_of_hanoi(3, 'A', 'C', 'B')
```

## Advanced Interview Topics

### Two Pointers Technique
```python
# Find pair that sums to target
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Test
arr = [1, 3, 4, 5, 7, 10, 11]
print(two_sum_sorted(arr, 9))  # [1, 4] (3 + 7)
```

### Sliding Window
```python
# Maximum sum of subarray of size k
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return 0
    
    max_sum = sum(arr[:k])
    current_sum = max_sum
    
    for i in range(k, len(arr)):
        current_sum = current_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Test
arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
print(max_sum_subarray(arr, 4))  # 24
```

### Dynamic Programming
```python
# Fibonacci with memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

# Test
print(fib_memo(10))  # 55
```

### String Manipulation
```python
# Check if string is palindrome
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

# Test
print(is_palindrome("racecar"))  # True
```

## Practice Tips
- Understand time/space complexity
- Consider edge cases: empty arrays, single elements
- Practice both recursive and iterative solutions
- Use Python's built-in data structures effectively
- Think about follow-up questions (optimize space, handle duplicates, etc.)
- For robotics: Consider real-time constraints and memory limitations
