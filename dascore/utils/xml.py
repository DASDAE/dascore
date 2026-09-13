"""Utilities for XML files."""

from xml.etree import ElementTree


def xml_to_dict(xml_string):
    """Convert a simple xml string to a dict."""
    root = ElementTree.fromstring(xml_string)
    return _element_to_dict(root)


def _element_to_dict(element):
    """
    Recursively convert a simple element tree to a dictionary.

    Complex XML structures are unsupported.
    """
    if len(element) == 0:
        return element.text

    result = {}
    for child in element:
        child_value = _element_to_dict(child)
        if child.tag in result:
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_value)
            else:
                result[child.tag] = [result[child.tag], child_value]
        else:
            result[child.tag] = child_value
    return result
