"""Unit tests for hello_world.py."""

from hello_world.hello_world import hello

def test_hellow_world():
    assert hello("Frank") == "Hello Frank"
    assert hello("Sam") == "Hello Sam"
