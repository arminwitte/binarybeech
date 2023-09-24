from binarybeech.reporter import Reporter, reporter


def test_reporter_class():
    reporter = Reporter()
    reporter.reset(["col1", "col2"])
    for i in range(10):
        reporter["col1"] = i
        reporter["col2"] = i**2
        reporter.print()
    assert 0 == 0


def test_reporter():
    reporter.reset(["col1", "col2"])
    for i in range(10):
        reporter["col1"] = i
        reporter["col2"] = i**2
        reporter.print()
    assert 0 == 0
