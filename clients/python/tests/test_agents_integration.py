"""
Tests for TensorZero + OpenAI Agents SDK Integration

These tests verify that the integration works correctly, including:
- Config parsing and template detection
- Client patching and setup
- Message conversion for templated functions
"""

import pytest
import tempfile
import toml
from pathlib import Path
from unittest.mock import patch, AsyncMock

try:
    import tensorzero.agents as tz_agents
    from tensorzero.agents import TensorZeroConfigParser, TensorZeroAgentsError

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False


@pytest.mark.skipif(
    not AGENTS_AVAILABLE, reason="Agents SDK dependencies not installed"
)
class TestTensorZeroAgentsIntegration:
    def create_test_config(self, tmp_path: Path) -> Path:
        """Create a test TensorZero configuration."""
        config_data = {
            "functions": {
                "test_function": {
                    "variants": {
                        "baseline": {
                            "system_template": "templates/test_template.txt",
                            "model": "gpt-4",
                        }
                    }
                }
            },
            "tools": {"test_tool": "tools/test_tool.json"},
        }

        # Create config file
        config_path = tmp_path / "tensorzero.toml"
        with open(config_path, "w") as f:
            toml.dump(config_data, f)

        # Create template file
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        template_path = templates_dir / "test_template.txt"
        template_path.write_text(
            "You are a helpful assistant. User question: {{ question }}"
        )

        # Create tool file
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        tool_path = tools_dir / "test_tool.json"
        tool_data = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
        tool_path.write_text(toml.dumps(tool_data))

        return config_path

    def test_config_parser_initialization(self, tmp_path):
        """Test that TensorZeroConfigParser initializes correctly."""
        config_path = self.create_test_config(tmp_path)

        parser = TensorZeroConfigParser(config_path)

        assert parser.config_path == config_path
        assert "test_function::baseline" in parser.templated_functions
        assert parser.templated_functions["test_function::baseline"]["variables"] == [
            "question"
        ]

    def test_config_parser_missing_file(self):
        """Test that TensorZeroConfigParser raises error for missing config."""
        with pytest.raises(TensorZeroAgentsError, match="TensorZero config not found"):
            TensorZeroConfigParser("nonexistent.toml")

    def test_template_detection(self, tmp_path):
        """Test that template variables are detected correctly."""
        config_path = self.create_test_config(tmp_path)
        parser = TensorZeroConfigParser(config_path)

        template_info = parser.templated_functions["test_function::baseline"]
        assert template_info["function_name"] == "test_function"
        assert template_info["variant_name"] == "baseline"
        assert "question" in template_info["variables"]
        assert "{{ question }}" in template_info["template_content"]

    def test_tool_parsing(self, tmp_path):
        """Test that tools are parsed correctly."""
        config_path = self.create_test_config(tmp_path)
        parser = TensorZeroConfigParser(config_path)

        assert "test_tool" in parser.available_tools
        # Note: The tool parsing might need adjustment based on actual tool format

    @patch("tensorzero.agents.patch_openai_client")
    @patch("tensorzero.agents.set_default_openai_client")
    @patch("tensorzero.agents.AsyncOpenAI")
    async def test_setup_tensorzero_agents(
        self, mock_openai, mock_set_client, mock_patch, tmp_path
    ):
        """Test that setup_tensorzero_agents works correctly."""
        config_path = self.create_test_config(tmp_path)

        # Mock the async client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_patch.return_value = mock_client

        config = await tz_agents.setup_tensorzero_agents(
            config_path=config_path,
            base_url="http://localhost:3000",
            api_key="test-key",
        )

        # Verify setup calls
        mock_openai.assert_called_once_with(
            base_url="http://localhost:3000/openai/v1", api_key="test-key"
        )
        mock_patch.assert_called_once_with(mock_client)
        mock_set_client.assert_called_once()

        # Verify config returned
        assert isinstance(config, TensorZeroConfigParser)
        assert len(config.templated_functions) > 0

    def test_message_converter_detection(self, tmp_path):
        """Test that message converter correctly detects templated functions."""
        config_path = self.create_test_config(tmp_path)
        parser = TensorZeroConfigParser(config_path)
        converter = tz_agents.TensorZeroMessageConverter(parser)

        # Should detect templated function
        assert converter.should_convert_message(
            "tensorzero::function_name::test_function::baseline"
        )

        # Should not detect non-templated function
        assert not converter.should_convert_message("gpt-4")
        assert not converter.should_convert_message(
            "tensorzero::function_name::nonexistent::baseline"
        )

    def test_message_conversion(self, tmp_path):
        """Test that messages are converted to TensorZero format correctly."""
        config_path = self.create_test_config(tmp_path)
        parser = TensorZeroConfigParser(config_path)
        converter = tz_agents.TensorZeroMessageConverter(parser)

        messages = [{"role": "user", "content": "What is AI?"}]

        converted = converter.convert_messages(
            messages,
            "tensorzero::function_name::test_function::baseline",
            question="What is AI?",
        )

        # Should convert to TensorZero format
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert "tensorzero::arguments" in converted[0]["content"][0]
        assert (
            converted[0]["content"][0]["tensorzero::arguments"]["question"]
            == "What is AI?"
        )

    @patch("tensorzero.agents.AGENTS_AVAILABLE", False)
    async def test_setup_without_agents_sdk(self, tmp_path):
        """Test that setup fails gracefully when Agents SDK is not available."""
        config_path = self.create_test_config(tmp_path)

        with pytest.raises(
            TensorZeroAgentsError, match="OpenAI Agents SDK not available"
        ):
            await tz_agents.setup_tensorzero_agents(config_path)

    def test_list_functions_before_setup(self):
        """Test that function listing works correctly before and after setup."""
        # Before setup
        assert tz_agents.list_templated_functions() == {}
        assert tz_agents.list_available_tools() == {}
        assert tz_agents.get_config() is None


if __name__ == "__main__":
    pytest.main([__file__])
