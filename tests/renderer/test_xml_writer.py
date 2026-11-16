from unflatten.renderer.xml_writer import NSMAP, make_root, qn, serialize


def test_qn_builds_clark_notation():
    assert qn("w", "document") == f"{{{NSMAP['w']}}}document"


def test_make_root_declares_only_used_prefixes():
    root = make_root("w", "document", used_prefixes=["w"])
    assert root.nsmap == {"w": NSMAP["w"]}


def test_serialize_declaration_is_double_quoted_and_no_pretty_print():
    root = make_root("w", "document", used_prefixes=["w"])
    body = root.makeelement(qn("w", "body"), {})
    root.append(body)
    data = serialize(root)
    assert data.startswith(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n')
    # no pretty-print whitespace between elements
    assert b"\n  <" not in data


def test_serialize_preserves_attribute_insertion_order():
    root = make_root("w", "p", used_prefixes=["w"])
    root.set(qn("w", "b"), "2")
    root.set(qn("w", "a"), "1")
    data = serialize(root)
    assert data.index(b'w:b="2"') < data.index(b'w:a="1"')


def test_serialize_is_deterministic_across_calls():
    root1 = make_root("w", "document", used_prefixes=["w"])
    root2 = make_root("w", "document", used_prefixes=["w"])
    assert serialize(root1) == serialize(root2)
