import itertools
import pytest

_port_counter = itertools.count(50099)


@pytest.fixture
def port() -> int:
    return next(_port_counter)
