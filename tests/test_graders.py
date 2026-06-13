from evals import graders


def test_citation_match_is_bidirectional():
    assert graders.citation_match(["Canberra"], ["Canberra"]) == 1.0
    assert graders.citation_match(["A-0 System"], ["A-0"]) == 1.0  # expected is substring of cited
    assert graders.citation_match(["Compiler"], ["A-0 System"]) == 0.0
    assert graders.citation_match([], ["Canberra"]) == 0.0


def test_refusal_correct():
    assert graders.refusal_correct("The sources do not contain the answer.", True) == 1.0
    assert graders.refusal_correct("The capital is Canberra.", False) == 1.0
    assert graders.refusal_correct("The capital is Canberra.", True) == 0.0
