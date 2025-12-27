# Алгоритми пошуку
# main.py
# ============================================================
# HW:
# 1) HashTable з delete
# 2) Binary search (upper bound)
# 3) Порівняння пошуку підрядка (BM / KMP / RK)
#    Тексти статей ВБУДОВАНІ в програму
# ============================================================

import timeit
from typing import Any, List, Optional, Tuple, Dict, Callable


# ============================================================
# Завдання 1
# ============================================================

class HashTable:
    def __init__(self, size: int):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key: Any) -> int:
        return hash(key) % self.size

    def insert(self, key: Any, value: Any) -> None:
        idx = self.hash_function(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                pair[1] = value
                return
        self.table[idx].append([key, value])

    def get(self, key: Any) -> Any:
        idx = self.hash_function(key)
        for pair in self.table[idx]:
            if pair[0] == key:
                return pair[1]
        return None

    def delete(self, key: Any) -> bool:
        idx = self.hash_function(key)
        for i, pair in enumerate(self.table[idx]):
            if pair[0] == key:
                self.table[idx].pop(i)
                return True
        return False


# ============================================================
# Завдання 2
# ============================================================

def binary_search_upper_bound(arr: List[float], target: float) -> Tuple[int, Optional[float]]:
    left, right = 0, len(arr) - 1
    iterations = 0
    upper = None

    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        if arr[mid] >= target:
            upper = arr[mid]
            right = mid - 1
        else:
            left = mid + 1

    return iterations, upper


# ============================================================
# Завдання 3 — ТЕКСТИ СТАТЕЙ (ВБУДОВАНІ)
# ============================================================

ARTICLE_1 = """
Методи та структури даних для реалізації бази даних рекомендаційної системи соціальної мережі.
Рекомендаційні системи є важливою складовою соціальних мереж та значним чином впливають
на те, яким користувачі сприймають інформаційний простір.
У статті проведено дослідження різних структур даних, які можна використати
для створення бази даних рекомендаційної системи, зокрема зв’язний список,
розгорнутий зв’язний список, хеш-таблиця, B-дерево та B+-дерево.
Відповідно до результатів експериментів розгорнутий список показав
найкращі показники швидкодії та використання пам’яті.
"""

ARTICLE_2 = """
Використання алгоритмів у бібліотеках мов програмування.
Алгоритми та структури даних є фундаментальною частиною комп’ютерних наук.
У роботі розглянуто лінійний, двійковий, інтерполяційний та експоненціальний пошук.
Правильно підібраний алгоритм пошуку відіграє визначальну роль
у продуктивності програмних систем.
"""


# ============================================================
# Алгоритми пошуку підрядка
# ============================================================

def boyer_moore(text: str, pattern: str) -> int:
    if not pattern:
        return 0
    last = {c: i for i, c in enumerate(pattern)}
    m, n = len(pattern), len(text)
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            return i
        i += max(1, j - last.get(text[i + j], -1))
    return -1


def kmp(text: str, pattern: str) -> int:
    if not pattern:
        return 0
    lps = [0] * len(pattern)
    j = 0

    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j

    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                return i - j
        else:
            j = lps[j - 1] if j > 0 else 0
            if j == 0:
                i += 1
    return -1


def rabin_karp(text: str, pattern: str) -> int:
    if not pattern:
        return 0
    base, mod = 256, 10**9 + 7
    m, n = len(pattern), len(text)
    h = pow(base, m - 1, mod)

    p_hash = t_hash = 0
    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        t_hash = (t_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash and text[i:i+m] == pattern:
            return i
        if i < n - m:
            t_hash = (t_hash - ord(text[i]) * h) % mod
            t_hash = (t_hash * base + ord(text[i + m])) % mod
    return -1


# ============================================================
# Benchmark
# ============================================================

def benchmark(text: str, present: str, absent: str):
    algos = {
        "Boyer-Moore": boyer_moore,
        "KMP": kmp,
        "Rabin-Karp": rabin_karp
    }

    for name, fn in algos.items():
        t1 = min(timeit.repeat(lambda: fn(text, present), repeat=5, number=1))
        t2 = min(timeit.repeat(lambda: fn(text, absent), repeat=5, number=1))
        print(f"{name:<12} | існує: {t1:.6f}s | вигаданий: {t2:.6f}s")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Завдання 1
    H = HashTable(5)
    H.insert("apple", 10)
    H.insert("orange", 20)
    H.delete("orange")

    # Завдання 2
    arr = [0.5, 1.2, 2.7, 3.14, 5.0]
    print(binary_search_upper_bound(arr, 2.5))

    # Завдання 3
    print("\nСтаття 1")
    benchmark(ARTICLE_1, "структур", "not_existing")

    print("\nСтаття 2")
    benchmark(ARTICLE_2, "алгоритм", "not_existing")
