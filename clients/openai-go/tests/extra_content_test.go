/**
Tests for `tensorzero_extra_content` round-trip support.

These tests verify that extra content blocks (Thought, Unknown) can be:
1. Received from the API in responses
2. Sent back to the API in follow-up requests (round-trip)
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

func TestExtraContent(t *testing.T) {
	t.Run("should round-trip extra content non-streaming", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		// Step 1: Make inference request with a model that returns Thought content
		// The dummy::reasoner model returns [Thought, Text] content
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::dummy::reasoner",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "Initial API request failed")

		// Step 2: Verify response structure
		require.NotEmpty(t, resp.Choices, "Response should have choices")
		content := resp.Choices[0].Message.Content
		require.NotEmpty(t, content, "Response should have content")

		// Check for tensorzero_extra_content at the message level
		extraContentField, ok := resp.Choices[0].Message.JSON.ExtraFields["tensorzero_extra_content"]
		require.True(t, ok, "Response message should have tensorzero_extra_content field")

		var extraContent []map[string]interface{}
		err = json.Unmarshal([]byte(extraContentField.Raw()), &extraContent)
		require.NoError(t, err, "Failed to parse tensorzero_extra_content")
		require.NotEmpty(t, extraContent, "Extra content should have at least one block")

		// Verify the structure of the thought block
		thoughtBlock := extraContent[0]
		assert.Equal(t, "thought", thoughtBlock["type"], "First block should be a thought")
		assert.NotNil(t, thoughtBlock["insert_index"], "Thought block should have insert_index")
		assert.NotNil(t, thoughtBlock["text"], "Thought block should have text field")

		// Step 3: Round-trip - send the extra content back as an assistant message
		roundtripReq := &openai.ChatCompletionNewParams{
			Model: "tensorzero::model_name::dummy::echo",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
				openai.AssistantMessage(content),
				openai.UserMessage("Continue"),
			},
		}

		// Set extra content on the assistant message via extra fields
		assistantMsg := roundtripReq.Messages[1].OfAssistant()
		if assistantMsg != nil {
			assistantMsg.SetExtraFields(map[string]any{
				"tensorzero_extra_content": extraContent,
			})
		}

		roundtripReq.SetExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		roundtripResp, err := client.Chat.Completions.New(ctx, *roundtripReq)
		require.NoError(t, err, "Round-trip API request failed")

		// Verify round-trip succeeded
		require.NotEmpty(t, roundtripResp.Choices, "Round-trip response should have choices")
	})

	t.Run("should round-trip extra content streaming", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		// Step 1: Make streaming inference request
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::dummy::reasoner",
			Messages: messages,
		}
		req.SetExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		// Step 2: Collect chunks and extract extra content
		var extraContentChunks []map[string]interface{}
		var contentText string

		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta

				// Collect text content
				if delta.Content != "" {
					contentText += delta.Content
				}

				// Collect extra content chunks from delta
				extraContentField, ok := delta.JSON.ExtraFields["tensorzero_extra_content"]
				if ok {
					var chunkExtraContent []map[string]interface{}
					err := json.Unmarshal([]byte(extraContentField.Raw()), &chunkExtraContent)
					if err == nil {
						extraContentChunks = append(extraContentChunks, chunkExtraContent...)
					}
				}
			}
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")

		// Step 3: Verify we received extra content in streaming
		assert.NotEmpty(t, extraContentChunks, "Streaming should include extra content chunks")

		// Reconstruct extra content for round-trip (filter for chunks with insert_index)
		var reconstructedExtraContent []map[string]interface{}
		for _, chunk := range extraContentChunks {
			if chunk["insert_index"] != nil {
				reconstructedExtraContent = append(reconstructedExtraContent, chunk)
			}
		}

		// Step 4: Round-trip if we have valid content
		if len(reconstructedExtraContent) > 0 && len(contentText) > 0 {
			roundtripReq := &openai.ChatCompletionNewParams{
				Model: "tensorzero::model_name::dummy::echo",
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Hello"),
					openai.AssistantMessage(contentText),
					openai.UserMessage("Continue"),
				},
			}

			// Set extra content on the assistant message via extra fields
			assistantMsg := roundtripReq.Messages[1].OfAssistant()
			if assistantMsg != nil {
				assistantMsg.SetExtraFields(map[string]any{
					"tensorzero_extra_content": reconstructedExtraContent,
				})
			}

			roundtripReq.SetExtraFields(map[string]any{
				"tensorzero::episode_id": episodeID.String(),
			})

			roundtripResp, err := client.Chat.Completions.New(ctx, *roundtripReq)
			require.NoError(t, err, "Streaming round-trip API request failed")

			// Verify round-trip succeeded
			assert.NotEmpty(t, roundtripResp.Choices, "Streaming round-trip response should have choices")
		}
	})
}
