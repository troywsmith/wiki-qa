from evals import scorers


def test_keyword_recall_fraction():
    assert scorers.keyword_recall("Grace Hopper built it", ["Grace Hopper"]) == 1.0
    assert scorers.keyword_recall("nope", ["Grace Hopper"]) == 0.0
    assert scorers.keyword_recall("Hopper and A-0", ["Hopper", "A-0", "UNIVAC"]) == 2 / 3


def test_keyword_recall_not_applicable_when_no_facts():
    assert scorers.keyword_recall("anything", []) is None


def test_citation_match_is_bidirectional():
    assert scorers.citation_match(["Canberra"], ["Canberra"]) == 1.0
    assert scorers.citation_match(["A-0 System"], ["A-0"]) == 1.0  # expected is substring of cited
    assert scorers.citation_match(["Compiler"], ["A-0 System"]) == 0.0
    assert scorers.citation_match([], ["Canberra"]) == 0.0


def test_refusal_correct():
    assert scorers.refusal_correct("The sources do not contain the answer.", True) == 1.0
    assert scorers.refusal_correct("The capital is Canberra.", False) == 1.0
    assert scorers.refusal_correct("The capital is Canberra.", True) == 0.0
