"""G7 validation: DSML tool-call parameter regexes (deepseek_v4 parser).

Drives the PRODUCTION module's _PARAM_RE / _PARTIAL_PARAM_RE (no re-derivation,
so a stale builder cannot fool the test).

The REBASE-3 merge united two bugfixes:
- upstream #56271: optional `string=` attribute (model omits it); missing ->
  false-like (JSON try, literal fallback)
- G7: value capture tempered at DSML tag boundaries — `(.*?)` latches onto a
  LATER parameter when the closer is mis-spelled (absorbs subsequent
  parameters); `(.*)$` leaks the closer into streamed partial arguments
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _HERE)

import vllm.parser.deepseek_v4 as pv  # noqa: E402

PS = pv.DSML_PARAM_START          # '<｜DSML｜parameter'
PC = pv.DSML_PARAM_CLOSE          # '</｜DSML｜parameter>'  (includes '</' )


def test_complete_params_both_string_modes():
    m = pv._PARAM_RE.search(f'{PS} name="path" string="true">/tmp/a{PC}')
    assert m and m.group(1) == "path" and m.group(2) == "true"
    assert m.group(3) == "/tmp/a"
    m = pv._PARAM_RE.search(f'{PS} name="n" string="false">42{PC}')
    assert m and m.group(2) == "false" and m.group(3) == "42"
    # MISSING string attribute (upstream #56271): group(2) is None,
    # value intact
    m = pv._PARAM_RE.search(f'{PS} name="n">left{{x: 1}}{PC}')
    assert m and m.group(2) is None, "optional string= not accepted!"
    assert m.group(3) == "left{x: 1}", m.group(3)
    print("complete params (string true/false/missing): OK")


def test_multi_param_adjacent_no_absorb():
    p1 = f'{PS} name="a" string="true">one{PC}'
    p2 = f'{PS} name="b" string="true">two{PC}'
    ms = list(pv._PARAM_RE.finditer(p1 + p2))
    assert len(ms) == 2, f"expected 2 params, got {len(ms)}"
    assert ms[0].group(3) == "one" and ms[1].group(3) == "two"
    print("adjacent params not absorbed: OK")


def test_malformed_closer_does_not_latch():
    """G7 bug: with `(.*?)`, a mis-spelled closer makes the value run on until
    it latches onto the NEXT real parameter opener and absorbs it."""
    good = f'{PS} name="b" string="true">two{PC}'
    bad = f'{PS} name="a" string="true">one</DSMLparameter>{good}'
    ms = list(pv._PARAM_RE.finditer(bad))
    assert len(ms) == 2, f"latch not prevented: {len(ms)} matches"
    assert ms[0].group(1) == "a" and ms[0].group(3) == "one</DSMLparameter>"
    assert ms[1].group(1) == "b" and ms[1].group(3) == "two"
    print("malformed closer no-latch: OK")


def test_streamed_value_does_not_leak_closer():
    """`(.*)$` would put closer text into the streamed partial value."""
    m = pv._PARTIAL_PARAM_RE.search(f'{PS} name="s" string="true">hello {PC}tail')
    assert m is not None
    assert m.group(3) == "hello ", f"leak: {m.group(3)!r}"
    print("streamed value no closer-leak: OK")


def test_value_with_embedded_angle_not_dsml_tag():
    """A value containing '<' that is NOT a DSML tag must survive the temper."""
    val = "compare a<b and i<j ok"
    m = pv._PARAM_RE.search(f'{PS} name="x" string="true">{val}{PC}')
    assert m.group(3) == val, f"over-tempered: {m.group(3)!r}"
    print("non-DSML '<' inside value survives: OK")


def test_converter_group_indexing():
    """The converter consumes groups (1=name, 2=string, 3=value). With the
    optional-string merge, absent string= must be false-like (JSON-try then
    literal): exercise the production converter directly."""
    conv = pv._dsml_arg_converter
    # literal-string param
    out = conv(f'{PS} name="path" string="true">/tmp/x{PC}', partial=False)
    assert out == '{"path": "/tmp/x"}', out
    # JSON param, explicit false
    out = conv(f'{PS} name="n" string="false">42{PC}', partial=False)
    assert out == '{"n": 42}', out
    # JSON param, attribute MISSING -> treated as false-like
    json_arg = '{ "a": 1 }'
    out = conv(f'{PS} name="n">{json_arg}{PC}', partial=False)
    # JSON decode failure with missing attr -> literal fallback
    out = conv(f'{PS} name="n">not-json{PC}', partial=False)
    assert out == '{"n": "not-json"}', out
    print("converter group indexing + missing-string semantics: OK")


if __name__ == "__main__":
    test_complete_params_both_string_modes()
    test_multi_param_adjacent_no_absorb()
    test_malformed_closer_does_not_latch()
    test_streamed_value_does_not_leak_closer()
    test_value_with_embedded_angle_not_dsml_tag()
    test_converter_group_indexing()
    print("G7 DSML parser validation: ALL OK")
