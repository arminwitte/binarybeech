from binarybeech.reporter import Reporter


def test_reporter():
    reporter = Reporter(["col1", "col2"])
    for i in range(10):
        reporter["col1"] = i
        reporter["col2"] = i**2
        reporter.print()
    assert 0 == 0
