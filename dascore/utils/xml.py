"""
Utilities for working with xml files.
"""

from xml.etree import ElementTree


def xml_to_dict(xml_string):
    """Convert a simple xml string to a dict."""
    root = ElementTree.fromstring(xml_string)
    return _element_to_dict(root)


def _element_to_dict(element):
    """
    Recursively convert an element tree into a dict.

    Note: This function is probably not general enough to handle complicated
    xml, use with caution.
    """
    # Base case: If the element has no children, return its text content
    if len(element) == 0:
        return element.text

    # Recursive case: Convert children to dictionary
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
