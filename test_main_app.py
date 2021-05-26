# Unit test file
# Test all major functions in main_app.py to ensure that no errors occur

import pytest
from main_app import lowercase_text


def test_lowercase_text():
	assert lowercase_text("This is a test") == "this is a test"