from __future__ import annotations

from autosklearn.util.functional import roundrobin


def test_roundrobin_empty_iterables() -> None:
    """
    Expects
    -------
    * Should iterate through the non-empty list if one is empty
    * Should return empty iterator if both iterators are empty
    """
    l1: list[str] = []
    l2: list[str] = ["a", "b", "c"]
    l3: list[str] = []

    assert list(roundrobin(l1, l2)) == l2
    assert list(roundrobin(l1, l3)) == []


def test_roundrobin_duplicates() -> None:
    """
    Expects
    -------
    * Expects the two lists to be interleaved with repetition allowed
    """
    # Orange is placed in here, once as a color and once as a fruit
    colours = ["orange", "red", "green"]
    fruits = ["apple", "banana", "orange"]

    # Interleaved order with repetition
    expected = ["orange", "apple", "red", "banana", "green", "orange"]

    answer = list(roundrobin(colours, fruits))

    assert answer == expected


def test_roundrobin_no_duplicates() -> None:
    """
    Expects
    -------
    * Expects the two lists to be interleaved with the return iterator containing no
      duplicates.
    """
    # Orange is placed in here, once as a color and once as a fruit
    colours = ["orange", "red", "green"]
    fruits = ["apple", "banana", "orange"]

    answer = list(roundrobin(colours, fruits, duplicates=False))

    # Interleaved order with the last "orange" removed
    expected = ["orange", "apple", "red", "banana", "green"]
    assert answer == expected


def test_roundrobin_no_duplicates_with_key() -> None:
    """
    Expects
    -------
    * Should use the `key` to determine duplicates when removing them
    """
    colours = ["orange", "red", "green"]
    fruits = ["apple", "banana", "orange"]

    last_char = lambda x: x[-1]

    answer = list(roundrobin(colours, fruits, duplicates=False, key=last_char))

    # Interleaved order with duplicates determined by `last_char` removed
    expected = ["orange", "red", "banana", "green"]
    assert answer == expected
