from src.evaluation import metrics as m


def test_first_hit_rank_and_rr():
    results = [{"t": "x"}, {"t": "gold"}, {"t": "gold"}]
    rank = m.first_hit_rank(results, lambda r: r["t"] == "gold")
    assert rank == 2 and m.reciprocal_rank(rank) == 0.5
    assert m.reciprocal_rank(m.first_hit_rank(results, lambda r: False)) == 0.0


def test_precision_at_k():
    assert m.precision_at_k([2, 0, 2, 1], k=4) == 0.5
    assert m.precision_at_k([2, 0, 2, 1], k=4, min_grade=1) == 0.75


def test_ndcg_perfect_and_reversed():
    assert m.ndcg_at_k([2, 1, 0], k=3) == 1.0
    assert m.ndcg_at_k([0, 1, 2], k=3) < 0.6
    assert m.ndcg_at_k([0, 0], k=2) == 0.0


def test_keyword_match_ignores_case_and_commas():
    assert m.keyword_match("It used 5,000 questions", ["5000"], 1)
    assert m.keyword_match("arabic and English", ["Arabic", "English"], 2)
    assert not m.keyword_match("only English", ["Arabic", "English"], 2)


def test_sentence_support():
    sources = [{"text": "we evaluate on 85 question answer triples from a fintech knowledge base"}]
    rows = m.sentence_support(
        "The system was evaluated on 85 question answer triples [1]. It also won an award for design [1]. "
        "Results were strong overall without any citation.", sources)
    assert [r["supported"] for r in rows] == [True, False, False]
    assert rows[2]["cited"] == []
