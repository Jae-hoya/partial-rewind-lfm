"""`lfm/estimator.py` 와 그 배선 (SPEC_M1prime §1).

가장 중요한 것은 **기본 경로가 안 바뀌었다**는 증명이다. Test A 부터 D14 까지
전부 `nli_max` 로 측정됐으므로, 리팩터가 수치를 한 자리라도 움직이면 논문의 표가
전부 어긋난다. 그래서 옛 계산을 이 파일 안에 그대로 다시 적어 두고 대조한다
(구현을 호출해서 비교하면 둘 다 틀려도 통과한다).
"""

from __future__ import annotations

import pytest

from lfm.estimator import (MiniCheckDocEstimator, NLIMaxEstimator, load_estimator)
from lfm.evidence import grounding, sentences_with_premises
from lfm.types import BranchOutput, Chunk, Perspective, SubAnswer


class FakeNLI:
    """(전제, 문장) -> 전제 안에 문장의 첫 낱말이 있으면 0.9, 없으면 0.1."""

    def __init__(self):
        self.calls = 0

    def entailment(self, pairs):
        self.calls += 1
        out = []
        for premise, sent in pairs:
            head = sent.split()[0] if sent.split() else ""
            out.append(0.9 if head and head in premise else 0.1)
        return out

    def max_entailment(self, premises, claim):
        return max(self.entailment([(p, claim) for p in premises]), default=0.0)


class FakeScorer:
    """단일 문서 전제를 받는 scorer. 문서 길이로 내부 청크 수를 흉내낸다."""

    def __init__(self, table=None, chunk_limit=10_000):
        self.table = table or {}
        self.chunk_limit = chunk_limit
        self.last_n_chunks = 1
        self.seen: list[tuple[str, str]] = []

    def score(self, doc, claim):
        self.seen.append((doc, claim))
        self.last_n_chunks = max(1, -(-len(doc) // self.chunk_limit))
        return self.table.get(claim, 0.0)


def mk(sentences_and_citations, chunks):
    """BranchOutput 을 최소로 만든다."""
    subs = [SubAnswer(index=i, sub_question="q%d" % i, text=text, citations=list(cits))
            for i, (text, cits) in enumerate(sentences_and_citations)]
    persp = Perspective(role="r", question="q",
                        sub_questions=tuple("q%d" % i for i in range(len(subs))))
    return BranchOutput(perspective=persp, sub_answers=subs, claims=[],
                        chunks=[Chunk(chunk_id=c["chunk_id"], doc_id=int(c["doc_id"][1:]),
                                      title=c["title"], section_path="", text=c["text"])
                                for c in chunks])


# ---------------------------------------------------------------- 기본 경로 불변

def test_nli_max_는_옛_계산과_한_자리도_다르지_않다():
    """리팩터 전 코드를 여기 그대로 다시 적어 대조한다."""
    nli = FakeNLI()
    out = mk([("alpha one. beta two.", ["c1"]), ("gamma three.", ["c1", "c2"])],
             [{"chunk_id": "c1", "doc_id": "d1", "title": "t", "text": "alpha gamma"},
              {"chunk_id": "c2", "doc_id": "d2", "title": "t", "text": "beta"}])

    # --- 옛 구현을 그대로 옮긴 것 ---
    cmap = out.chunk_map()
    sents = []
    for sa in out.sub_answers:
        from lfm.types import split_sentences
        for s in split_sentences(sa.text):
            sents.append((s, [c for c in sa.citations if c in cmap]))
    pairs, spans = [], []
    for s, cited in sents:
        start = len(pairs)
        for cid in cited:
            pairs.append((cmap[cid].text, s))
        spans.append((start, len(pairs)))
    scores = FakeNLI().entailment(pairs)
    old = sum(1 for a, b in spans if b > a and max(scores[a:b]) >= 0.5) / len(sents)

    assert grounding(out, nli, 0.5) == old


def test_인용_없는_문장은_grounded_가_아니다():
    nli = FakeNLI()
    out = mk([("alpha one.", [])],
             [{"chunk_id": "c1", "doc_id": "d1", "title": "t", "text": "alpha"}])
    assert grounding(out, nli, 0.5) == 0.0


def test_문장이_없으면_0():
    out = mk([], [])
    assert grounding(out, FakeNLI(), 0.5) == 0.0


def test_grounded_many_는_grounded_와_같은_답을_준다():
    """배치 경로와 단건 경로가 갈리면 표가 조용히 어긋난다."""
    est = NLIMaxEstimator(FakeNLI(), 0.5)
    items = [("alpha x.", ["alpha here"]), ("zeta y.", ["alpha here"]), ("q.", [])]
    assert est.grounded_many(items) == [est.grounded(s, p) for s, p in items]


# ------------------------------------------------------------ 확정 연산자 규칙

def test_minicheck_는_인용_청크를_이어붙인_단일_문서를_전제로_쓴다():
    sc = FakeScorer({"alpha one.": 0.9})
    est = MiniCheckDocEstimator(sc)
    est.grounded("alpha one.", ["첫째 청크", "둘째 청크"])
    doc, claim = sc.seen[-1]
    assert doc == "첫째 청크\n\n둘째 청크"       # 최댓값이 아니라 이어붙이기
    assert claim == "alpha one."
    assert len(sc.seen) == 1                     # 청크마다 부르지 않는다


@pytest.mark.parametrize("p, want", [(0.51, True), (0.5, False), (0.49, False)])
def test_minicheck_는_엄격_부등호다(p, want):
    """스펙은 `> 0.5` 다. 0.5 는 통과가 아니다."""
    est = MiniCheckDocEstimator(FakeScorer({"c": p}))
    assert est.grounded("c", ["prem"]) is want


def test_nli_max_는_비엄격이라_경계에서_둘이_다르다():
    """두 연산자의 경계 규칙이 실제로 다름을 못 박아 둔다."""
    class Half:
        def entailment(self, pairs):
            return [0.5] * len(pairs)
    assert NLIMaxEstimator(Half(), 0.5).grounded("c", ["p"]) is True
    assert MiniCheckDocEstimator(FakeScorer({"c": 0.5})).grounded("c", ["p"]) is False


def test_전제가_내부에서_쪼개지면_죽는다():
    """max 가 뒷문으로 돌아오는 경로다. 조용히 틀리면 안 된다."""
    sc = FakeScorer({"c": 0.9}, chunk_limit=5)
    est = MiniCheckDocEstimator(sc)
    with pytest.raises(AssertionError, match="쪼개졌다"):
        est.grounded("c", ["아주 긴 전제 문서" * 5])


def test_전제가_없으면_지지가_아니다():
    assert MiniCheckDocEstimator(FakeScorer()).grounded("c", []) is False


# ---------------------------------------------------------------------- 설정

class Cfg:
    def __init__(self, d):
        self.d = d

    def get(self, k, default=None):
        return self.d.get(k, default)


def test_설정에_없으면_기본은_nli_max_다():
    """기본값이 바뀌면 과거 결과의 재현이 조용히 깨진다."""
    est = load_estimator(Cfg({}), nli=FakeNLI())
    assert est.name == "nli_max"


def test_minicheck_는_명시적으로_켜야_한다():
    est = load_estimator(Cfg({"estimator.kind": "minicheck_doc"}), scorer=FakeScorer())
    assert est.name == "minicheck_doc"


def test_scorer_없이_켜면_MiniCheck_를_스스로_올린다(monkeypatch):
    """실행 스크립트가 scorer 를 안 넘겨도 확정 연산자가 서야 한다.

    실제 모델(2GB)을 올리면 테스트가 못 돈다. 그래서 **불리는지만** 확인한다.
    """
    import lfm.estimator as E
    made = []

    class Stub:
        def __init__(self):
            made.append(1)

        def score(self, doc, claim):
            return 0.0

    monkeypatch.setattr(E, "MiniCheckScorer", Stub)
    est = E.load_estimator(Cfg({"estimator.kind": "minicheck_doc"}), nli=FakeNLI())
    assert est.name == "minicheck_doc"
    assert made == [1]


def test_nli_없이_nli_max_를_켜면_거부한다():
    with pytest.raises(ValueError, match="NLI"):
        load_estimator(Cfg({"estimator.kind": "nli_max"}))


def test_모르는_종류는_거부한다():
    with pytest.raises(ValueError, match="모르는"):
        load_estimator(Cfg({"estimator.kind": "무언가"}), nli=FakeNLI())


# ------------------------------------------------------------------ 배선 확인

def test_evidence_가_주입된_연산자를_실제로_쓴다():
    out = mk([("alpha one.", ["c1"])],
             [{"chunk_id": "c1", "doc_id": "d1", "title": "t", "text": "alpha"}])
    assert grounding(out, NLIMaxEstimator(FakeNLI(), 0.5)) == 1.0
    assert grounding(out, MiniCheckDocEstimator(FakeScorer({"alpha one.": 0.9}))) == 1.0
    assert grounding(out, MiniCheckDocEstimator(FakeScorer({"alpha one.": 0.4}))) == 0.0


def test_분모는_지지되지_않은_문장까지_포함한다():
    """지지된 문장 수가 아니라 **전체 문장 수**가 분모다.

    (`split_sentences` 는 마침표 뒤 대문자에서만 나눈다 — 그래서 대문자로 쓴다.)
    """
    out = mk([("Alpha one. Zeta two.", ["c1"])],
             [{"chunk_id": "c1", "doc_id": "d1", "title": "t", "text": "Alpha"}])
    assert len(sentences_with_premises(out)) == 2
    assert grounding(out, NLIMaxEstimator(FakeNLI(), 0.5)) == 0.5


class FakeTok:
    """글자 하나 = 토큰 하나."""
    eos_token = "</s>"

    def __call__(self, text, truncation=False):
        return {"input_ids": list(range(len(text)))}


def _bare_scorer(m, max_model_len=7500):
    from lfm.estimator import MiniCheckScorer
    sc = MiniCheckScorer.__new__(MiniCheckScorer)
    sc.chunk_size, sc.max_model_len = 7200, max_model_len
    sc.m, sc.calls, sc.violations, sc.last_n_chunks = m, 0, 0, None
    sc.cpu_fallbacks, sc.cpu_fallback_doc_chars = 0, []
    return sc


def test_입력이_max_model_len_을_넘으면_잘려서_채점되기_전에_죽는다():
    """이탈 3 (2026-11-05): 쪼개기는 낱말 수로, 자르기는 토큰 수로 해서 '1조각·위반 0' 인 채 claim 이 잘릴 수 있다."""
    called = []
    m = type("M", (), {})()
    m.model = type("Inf", (), {})()
    m.model.tokenizer = FakeTok()
    m.score = lambda docs, claims, chunk_size: called.append(1) or ([1], [0.9], [[docs[0]]], [[0.9]])
    sc = _bare_scorer(m, max_model_len=50)
    assert sc.score("d" * 20, "c") == pytest.approx(0.9)   # 'predict: '(9) + 20 + '</s>'(4) + 1 = 34 ≤ 50
    assert sc.max_input_tokens == 34
    with pytest.raises(AssertionError):
        sc.score("d" * 40, "c")                            # 54 > 50
    assert len(called) == 1                                # 넘친 입력은 모델에 들어가지 않는다


def test_GPU_메모리가_모자라면_같은_가중치로_CPU에서_한_조각_채점한다():
    """R2 파일럿 2026-11-05 OOM. 대체 경로는 같은 score 를 CPU 에서 다시 부르고, 장치를 되돌리고, 센다."""
    torch = pytest.importorskip("torch")
    from lfm.estimator import MiniCheckScorer

    class HF:
        def __init__(self):
            self.device = "cuda:0"
            self.moves = []

        def to(self, d):
            self.moves.append(str(d))
            self.device = str(d)
            return self

    class M:
        def __init__(self):
            self.model = type("Inf", (), {})()
            self.model.model = HF()
            self.model.tokenizer = FakeTok()
            self.n = 0

        def score(self, docs, claims, chunk_size):
            self.n += 1
            if self.model.model.device != "cpu":
                raise torch.cuda.OutOfMemoryError("fake oom")
            return [1], [0.73], [[docs[0]]], [[0.73]]

    sc = MiniCheckScorer.__new__(MiniCheckScorer)
    sc.chunk_size, sc.max_model_len = 7200, 7500
    sc.m, sc.calls, sc.violations, sc.last_n_chunks = M(), 0, 0, None
    sc.cpu_fallbacks, sc.cpu_fallback_doc_chars = 0, []
    assert sc.score("x" * 40, "c") == pytest.approx(0.73)
    assert sc.cpu_fallbacks == 1 and sc.cpu_fallback_doc_chars == [40]
    assert sc.last_n_chunks == 1 and sc.violations == 0 and sc.calls == 1
    assert sc.m.model.model.moves == ["cpu", "cuda:0"]      # 채점 뒤 원래 장치로 돌아간다


def test_같은_청크를_여러_번_인용해도_전제에는_한_번만_첫_인용_순서로_넣는다():
    """이탈 2 (2026-11-05): 인용 20, 고유 4 → 81,175자 전제. 고유 청크만, 첫 인용 순서."""
    from lfm.evidence import subq_targets
    chunks = [{"chunk_id": "b", "doc_id": "d2", "title": "t", "text": "beta 본문"},
              {"chunk_id": "a", "doc_id": "d1", "title": "t", "text": "alpha 본문"}]
    out = mk([("alpha one [a][a]. beta two [b][a][b].", ["a", "a", "b", "a", "b"])], chunks)
    items = sentences_with_premises(out)
    assert items and all(p == ["alpha 본문", "beta 본문"] for _, p in items)
    sc = FakeScorer({})
    subq_targets(out, MiniCheckDocEstimator(sc))
    assert sc.seen and all(doc == "alpha 본문\n\nbeta 본문" for doc, _ in sc.seen)
    # nli_max 는 최댓값이라 중복과 무관하다
    nli = FakeNLI()
    assert grounding(out, nli) == grounding(mk([("alpha one [a]. beta two [b].", ["a", "b"])], chunks), nli)


def test_GPU_동치_확인_길이를_넘는_입력은_처음부터_CPU에서_채점한다():
    """이탈 4 (2026-11-06): WDDM 은 OOM 대신 공유 메모리로 넘겨 멈춘 듯 느려진다. 길이로 미리 보낸다."""
    torch = pytest.importorskip("torch")
    seen = []

    class HF:
        device = "cuda:0"

        def to(self, d):
            HF.device = str(d)
            return self

    m = type("M", (), {})()
    m.model = type("Inf", (), {})()
    m.model.model, m.model.tokenizer = HF(), FakeTok()
    m.score = lambda docs, claims, chunk_size: seen.append(HF.device) or ([1], [0.6], [[docs[0]]], [[0.6]])
    sc = _bare_scorer(m)
    sc.GPU_MAX_TOKENS = 100
    sc.score("d" * 50, "c")                  # 64 토큰 → GPU
    sc.score("d" * 200, "c")                 # 214 토큰 → CPU, 뒤에 GPU 로 돌아온다
    assert seen == ["cuda:0", "cpu"] and HF.device == "cuda:0"
    assert sc.cpu_fallbacks == 1 and sc.cpu_fallback_doc_chars == [200]
