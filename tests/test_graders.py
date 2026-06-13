from evals import graders


def test_keyword_recall_fraction():
    assert graders.keyword_recall("Grace Hopper built it", ["Grace Hopper"]) == 1.0
    assert graders.keyword_recall("nope", ["Grace Hopper"]) == 0.0
    assert graders.keyword_recall("Hopper and A-0", ["Hopper", "A-0", "UNIVAC"]) == 2 / 3


def test_keyword_recall_not_applicable_when_no_facts():
    assert graders.keyword_recall("anything", []) is None


def test_citation_match_is_bidirectional():
    assert graders.citation_match(["Canberra"], ["Canberra"]) == 1.0
    assert graders.citation_match(["A-0 System"], ["A-0"]) == 1.0  # expected is substring of cited
    assert graders.citation_match(["Compiler"], ["A-0 System"]) == 0.0
    assert graders.citation_match([], ["Canberra"]) == 0.0


def test_refusal_correct():
    assert graders.refusal_correct("The sources do not contain the answer.", True) == 1.0
    assert graders.refusal_correct("The capital is Canberra.", False) == 1.0
    assert graders.refusal_correct("The capital is Canberra.", True) == 0.0
