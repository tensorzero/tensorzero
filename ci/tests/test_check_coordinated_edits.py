#!/usr/bin/env python3
"""
Tests for check_coordinated_edits.py

These tests create temporary git repositories to test various scenarios
of the coordinated edits check.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from check_coordinated_edits import check_coordinated_edits


class TestCheckCoordinatedEdits(unittest.TestCase):
    """Test suite for coordinated edits checking."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True, capture_output=True)

    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)

    def _create_file(self, path: str, content: str):
        """Create a file with the given content."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def _commit(self, message: str):
        """Add all files and commit."""
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], check=True, capture_output=True)

    def test_adding_lines_within_block_violations(self):
        """Test that adding lines within a Lint.IfEdited block triggers violations."""
        # Create initial file with Lint.IfEdited block
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target1.py, target2.py)

def bar():
    return 100
""",
        )
        self._commit("Initial commit")

        # Modify line within the block but don't edit required files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 43  # Changed value
# Lint.ThenEdit(target1.py, target2.py)

def bar():
    return 100
""",
        )
        self._commit("Modified line in block")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have violations
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["file"], "source.py")
        self.assertEqual(set(violations[0]["required"]), {"target1.py", "target2.py"})
        self.assertEqual(set(violations[0]["missing"]), {"target1.py", "target2.py"})

    def test_adding_lines_within_block_no_violations(self):
        """Test that adding lines within a Lint.IfEdited block with required edits passes."""
        # Create initial files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target1.py, target2.py)

def bar():
    return 100
""",
        )
        self._create_file("target1.py", "# Target 1\n")
        self._create_file("target2.py", "# Target 2\n")
        self._commit("Initial commit")

        # Modify line within the block and edit required files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 43  # Changed value
# Lint.ThenEdit(target1.py, target2.py)

def bar():
    return 100
""",
        )
        self._create_file("target1.py", "# Target 1 updated\n")
        self._create_file("target2.py", "# Target 2 updated\n")
        self._commit("Modified line in block with required edits")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have no violations
        self.assertEqual(len(violations), 0)

    def test_deleting_lines_within_block(self):
        """Test that deleting lines within a Lint.IfEdited block triggers violations."""
        # Create initial file
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    x = 1
    y = 2
    return x + y
# Lint.ThenEdit(target.py)
""",
        )
        self._commit("Initial commit")

        # Delete a line within the block
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    y = 2
    return y
# Lint.ThenEdit(target.py)
""",
        )
        self._commit("Deleted line in block")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have violations
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["file"], "source.py")
        self.assertEqual(violations[0]["missing"], ["target.py"])

    def test_deleting_entire_block_with_comments(self):
        """Test that deleting an entire Lint.IfEdited block doesn't trigger violations."""
        # Create initial file
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target.py)

def bar():
    return 100
""",
        )
        self._commit("Initial commit")

        # Delete the entire block including markers
        self._create_file(
            "source.py",
            """# This is a source file

def bar():
    return 100
""",
        )
        self._commit("Deleted entire block")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have no violations because the entire block (including markers) was deleted
        # The parse_lint_blocks function won't find the block in the new version
        self.assertEqual(len(violations), 0)

    def test_multiple_blocks_only_some_changed(self):
        """Test file with multiple blocks where only some are changed."""
        # Create initial file with multiple blocks
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target1.py)

def middle():
    return 50

# Lint.IfEdited()
def bar():
    return 100
# Lint.ThenEdit(target2.py)
""",
        )
        self._create_file("target1.py", "# Target 1\n")
        self._commit("Initial commit")

        # Modify only the first block and edit its required file
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 43  # Changed
# Lint.ThenEdit(target1.py)

def middle():
    return 50

# Lint.IfEdited()
def bar():
    return 100
# Lint.ThenEdit(target2.py)
""",
        )
        self._create_file("target1.py", "# Target 1 updated\n")
        self._commit("Modified first block with required edit")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have no violations
        self.assertEqual(len(violations), 0)

    def test_multiple_blocks_missing_some_edits(self):
        """Test file with multiple blocks where required edits are missing."""
        # Create initial file with multiple blocks
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target1.py)

# Lint.IfEdited()
def bar():
    return 100
# Lint.ThenEdit(target2.py)
""",
        )
        self._commit("Initial commit")

        # Modify both blocks but only edit one required file
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 43  # Changed
# Lint.ThenEdit(target1.py)

# Lint.IfEdited()
def bar():
    return 101  # Changed
# Lint.ThenEdit(target2.py)
""",
        )
        self._create_file("target1.py", "# Target 1 updated\n")
        self._commit("Modified both blocks but only edited target1")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have one violation for the second block
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["file"], "source.py")
        self.assertEqual(violations[0]["missing"], ["target2.py"])

    def test_multiline_then_edit_format(self):
        """Test multiline Lint.ThenEdit format."""
        # Create initial file with multiline ThenEdit
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(
#     target1.py,
#     target2.py,
#     target3.py
# )
""",
        )
        self._commit("Initial commit")

        # Modify line within the block without editing required files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 43  # Changed
# Lint.ThenEdit(
#     target1.py,
#     target2.py,
#     target3.py
# )
""",
        )
        self._commit("Modified line in block")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have violations
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["file"], "source.py")
        self.assertEqual(set(violations[0]["required"]), {"target1.py", "target2.py", "target3.py"})
        self.assertEqual(set(violations[0]["missing"]), {"target1.py", "target2.py", "target3.py"})

    def test_multiline_then_edit_with_partial_edits(self):
        """Test multiline Lint.ThenEdit format with some required edits present."""
        # Create initial files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(
#     target1.py,
#     target2.py,
#     target3.py
# )
""",
        )
        self._create_file("target1.py", "# Target 1\n")
        self._create_file("target2.py", "# Target 2\n")
        self._commit("Initial commit")

        # Modify line within the block and only edit some required files
        self._create_file(
            "source.py",
            """# This is a source file
# Lint.IfEdited()
def foo():
    return 43  # Changed
# Lint.ThenEdit(
#     target1.py,
#     target2.py,
#     target3.py
# )
""",
        )
        self._create_file("target1.py", "# Target 1 updated\n")
        self._create_file("target2.py", "# Target 2 updated\n")
        # Note: target3.py is NOT edited
        self._commit("Modified line in block with partial edits")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have violations for missing target3.py
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["file"], "source.py")
        self.assertEqual(set(violations[0]["required"]), {"target1.py", "target2.py", "target3.py"})
        self.assertEqual(violations[0]["missing"], ["target3.py"])

    def test_modifying_outside_block_no_violation(self):
        """Test that modifying lines outside the block doesn't trigger violations."""
        # Create initial file
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target.py)

def bar():
    return 100
""",
        )
        self._commit("Initial commit")

        # Modify line outside the block
        self._create_file(
            "source.py",
            """# This is a source file

# Lint.IfEdited()
def foo():
    return 42
# Lint.ThenEdit(target.py)

def bar():
    return 101  # Changed, but outside block
""",
        )
        self._commit("Modified line outside block")

        # Check for violations
        violations = check_coordinated_edits("HEAD~1", "HEAD")

        # Should have no violations
        self.assertEqual(len(violations), 0)

    def test_different_comment_styles(self):
        """Test that Lint markers work with different comment styles."""
        # Test with different comment styles
        test_cases = [
            ("source.js", "// Lint.IfEdited()\nconst x = 1;\n// Lint.ThenEdit(target.js)"),
            ("source.rs", "// Lint.IfEdited()\nlet x = 1;\n// Lint.ThenEdit(target.rs)"),
            ("source.sh", "# Lint.IfEdited()\nx=1\n# Lint.ThenEdit(target.sh)"),
        ]

        for i, (filename, initial_content) in enumerate(test_cases):
            with self.subTest(filename=filename):
                # Create initial file
                self._create_file(filename, initial_content)
                self._commit(f"Initial commit for {filename}")

                # Modify within block
                modified_content = initial_content.replace("= 1;", "= 2;").replace("=1", "=2")
                self._create_file(filename, modified_content)
                self._commit(f"Modified {filename}")

                # Check for violations
                violations = check_coordinated_edits("HEAD~1", "HEAD")

                # Should have violations
                self.assertGreater(len(violations), 0, f"Expected violations for {filename}")

                # Clean up files for next test (but don't reset git history)
                if os.path.exists(filename):
                    os.remove(filename)


if __name__ == "__main__":
    unittest.main()
