/**
Tests for tensorzero::include_raw_response parameter using the OpenAI Go SDK.

These tests verify that raw provider-specific response data is correctly returned
when tensorzero::include_raw_response is set to true via the OpenAI-compatible API.
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

func TestRawResponse(t *testing.T) {
	assertRawResponseEntryStructure := func(t *testing.T, entry map[string]interface{}) {
		t.Helper()

		// Verify model_inference_id exists
		assert.NotNil(t, entry["model_inference_id"], "Entry should have model_inference_id")

		// Verify provider_type exists and is a string
		providerType, ok := entry["provider_type"]
		assert.True(t, ok, "Entry should have provider_type")
		_, ok = providerType.(string)
		assert.True(t, ok, "provider_type should be a string")

		// Verify data exists and is a string (raw response from provider)
		data, ok := entry["data"]
		assert.True(t, ok, "Entry should have data")
		_, ok = data.(string)
		assert.True(t, ok, "data should be a string (raw response from provider)")
	}

	t.Run("should return tensorzero_raw_response in non-streaming response when requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":           episodeID.String(),
			"tensorzero::include_raw_response": true,
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Check for tensorzero_raw_response at the response level
		rawResponseField, ok := resp.JSON.ExtraFields["tensorzero_raw_response"]
		require.True(t, ok, "Response should have tensorzero_raw_response field when requested")

		var rawResponse []map[string]interface{}
		err = json.Unmarshal([]byte(rawResponseField.Raw()), &rawResponse)
		require.NoError(t, err, "Failed to parse tensorzero_raw_response")
		require.Greater(t, len(rawResponse), 0, "tensorzero_raw_response should have at least one entry")

		// Verify structure of first entry
		entry := rawResponse[0]
		assertRawResponseEntryStructure(t, entry)
	})

	t.Run("should not return tensorzero_raw_response when not requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":           episodeID.String(),
			"tensorzero::include_raw_response": false,
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// tensorzero_raw_response should not be present at the response level
		_, ok := resp.JSON.ExtraFields["tensorzero_raw_response"]
		assert.False(t, ok, "tensorzero_raw_response should not be present when not requested")
	})

	t.Run("should return tensorzero_raw_chunk in streaming response when requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":           episodeID.String(),
			"tensorzero::include_raw_response": true,
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

		// Chunks should have tensorzero_raw_chunk (the raw chunk data as a string)
		foundRawChunk := false
		for _, chunk := range allChunks {
			rawChunkField, ok := chunk.JSON.ExtraFields["tensorzero_raw_chunk"]
			if ok {
				foundRawChunk = true

				// raw_chunk should be a string (can be empty for some chunks)
				var rawChunk string
				err := json.Unmarshal([]byte(rawChunkField.Raw()), &rawChunk)
				require.NoError(t, err, "Failed to parse tensorzero_raw_chunk as string")
			}
		}

		assert.True(t, foundRawChunk, "Streaming response should include tensorzero_raw_chunk in at least one chunk")
	})

	t.Run("should not return tensorzero_raw_chunk in streaming response when not requested", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::gpt-4o-mini-2024-07-18",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id":           episodeID.String(),
			"tensorzero::include_raw_response": false,
		})

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		for stream.Next() {
			chunk := stream.Current()
			// tensorzero_raw_chunk should not be present when not requested
			_, ok := chunk.JSON.ExtraFields["tensorzero_raw_chunk"]
			assert.False(t, ok, "tensorzero_raw_chunk should not be present when not requested")
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")
	})
}
