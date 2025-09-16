
from router.route import heuristic_route

def test_structured():
    r, conf, feats = heuristic_route("Which Nolan movie has the highest IMDb rating?")
    assert r in ("structured","both")  # allow both due to comparison cue
    assert conf > 0.5

def test_unstructured():
    r, conf, feats = heuristic_route("What themes do critics mention about Interstellar?")
    assert r in ("unstructured","both")
    assert conf > 0.5

def test_both():
    r, conf, feats = heuristic_route("Compare Inception and Interstellar box office and themes")
    assert r == "both"
    assert feats["comparative"] is True
