"""
Tests for xml utilities.
"""

from dascore.utils.xml import xml_to_dict


class TestXMLtoDict:
    """Tests for converting XML to dictionaries."""

    def test_single_element(self):
        """Test conversion of XML with a single element."""
        xml_string = "<root>Hello</root>"
        assert xml_to_dict(xml_string) == "Hello"

    def test_multiple_elements(self):
        """Test conversion of XML with multiple elements."""
        xml_string = "<root><a>1</a><b>2</b></root>"
        expected_dict = {"a": "1", "b": "2"}
        assert xml_to_dict(xml_string) == expected_dict

    def test_nested_elements(self):
        """Test conversion of XML with nested elements."""
        xml_string = "<root><a><b>1</b></a></root>"
        expected_dict = {"a": {"b": "1"}}
        assert xml_to_dict(xml_string) == expected_dict

    def test_multiple_nested_elements(self):
        """Test conversion of XML with multiple nested elements."""
        xml_string = "<root><a><b>1</b></a><c><d>2</d></c></root>"
        expected_dict = {"a": {"b": "1"}, "c": {"d": "2"}}
        assert xml_to_dict(xml_string) == expected_dict

    def test_elements_repeated_twice(self):
        """Test conversion of XML with repeated elements."""
        xml_string = "<root><a>1</a><a>2</a></root>"
        expected_dict = {"a": ["1", "2"]}
        assert xml_to_dict(xml_string) == expected_dict

    def test_repeated_elements(self):
        """Test conversion of XML with repeated elements."""
        xml_string = "<root><a>1</a><a>2</a><a>3</a></root>"
        expected_dict = {"a": ["1", "2", "3"]}
        assert xml_to_dict(xml_string) == expected_dict
