/**
Tests for tensorzero::include_raw_usage parameter using the OpenAI Go SDK.

These tests verify that raw provider-specific usage data is correctly returned
when tensorzero::include_raw_usage is set to true via the OpenAI-compatible API.
*/

package tests

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRawUsage(t *testing.T) {
	assertOpenAIChatRawUsageFields := func(t *testing.T, entry map[string]interface{}) {
		t.Helper()

		usageValue, ok := entry["usage"]
		require.True(t, ok, "raw_usage entry should include usage")

		usage, ok := usageValue.(map[string]interface{})
		require.True(t, ok, "raw_usage entry usage should be an object")

		_, ok = usage["total_tokens"]
		assert.True(t, ok, "raw_usage usage should include total_tokens")

		promptDetails, ok := usage["prompt_tokens_details"].(map[string]interface{})
		require.True(t, ok, "raw_usage usage should include prompt_tokens_details")
		_, ok = promptDetails["cached_tokens"]
		assert.True(t, ok, "raw_usage usage should include prompt_tokens_details.cached_tokens")

		completionDetails, ok := usage["completion_tokens_details"].(map[string]interface{})
		require.True(t, ok, "raw_usage usage should include completion_tokens_details")
		_, ok = completionDetails["reasoning_tokens"]
		assert.True(t, ok, "raw_usage usage should include completion_tokens_details.reasoning_tokens")
	}

	t.Run("should return tensorzero_raw_usage in non-streaming response when requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":        episodeID.String(),
			"tensorzero::include_raw_usage": true,
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Verify usage exists
		require.NotNil(t, resp.Usage, "Response should have usage")

		// Check for tensorzero_raw_usage inside usage extra fields
		rawUsageField, ok := resp.Usage.JSON.ExtraFields["tensorzero_raw_usage"]
		require.True(t, ok, "Usage should have tensorzero_raw_usage field when requested")

		var rawUsage []map[string]interface{}
		err = json.Unmarshal([]byte(rawUsageField.Raw()), &rawUsage)
		require.NoError(t, err, "Failed to parse tensorzero_raw_usage")
		require.Greater(t, len(rawUsage), 0, "tensorzero_raw_usage should have at least one entry")

		// Verify structure of first entry
		entry := rawUsage[0]
		assert.NotNil(t, entry["model_inference_id"], "Entry should have model_inference_id")
		assert.NotNil(t, entry["provider_type"], "Entry should have provider_type")
		assert.NotNil(t, entry["api_type"], "Entry should have api_type")
		assertOpenAIChatRawUsageFields(t, entry)
	})

	t.Run("should not return tensorzero_raw_usage when not requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":        episodeID.String(),
			"tensorzero::include_raw_usage": false,
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Verify usage exists
		require.NotNil(t, resp.Usage, "Response should have usage")

		// tensorzero_raw_usage should not be present
		_, ok := resp.Usage.JSON.ExtraFields["tensorzero_raw_usage"]
		assert.False(t, ok, "tensorzero_raw_usage should not be present when not requested")
	})

	t.Run("should return tensorzero_raw_usage in streaming response when requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		// Note: tensorzero::include_raw_usage automatically enables include_usage for streaming
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":        episodeID.String(),
			"tensorzero::include_raw_usage": true,
		})

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		var allChunks []openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			allChunks = append(allChunks, chunk)
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")
		require.NotEmpty(t, allChunks, "No chunks were received")

		// The final chunk should have usage with tensorzero_raw_usage
		foundRawUsage := false
		for _, chunk := range allChunks {
			if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
				rawUsageField, ok := chunk.Usage.JSON.ExtraFields["tensorzero_raw_usage"]
				if ok {
					foundRawUsage = true

					var rawUsage []map[string]interface{}
					err := json.Unmarshal([]byte(rawUsageField.Raw()), &rawUsage)
					require.NoError(t, err, "Failed to parse tensorzero_raw_usage")
					require.Greater(t, len(rawUsage), 0, "tensorzero_raw_usage should have at least one entry")

					// Verify structure of first entry
					entry := rawUsage[0]
					assert.NotNil(t, entry["model_inference_id"], "Entry should have model_inference_id")
					assert.NotNil(t, entry["provider_type"], "Entry should have provider_type")
					assert.NotNil(t, entry["api_type"], "Entry should have api_type")
					assertOpenAIChatRawUsageFields(t, entry)
				}
			}
		}

		assert.True(t, foundRawUsage, "Streaming response should include tensorzero_raw_usage in final chunk")
	})
}
