/**
Tests for the TensorZero OpenAI-compatible endpoint using the OpenAI GO client

We use the Go Testing framework to run the tests.

These tests cover the major functionality of the translation
layer between the OpenAI interface and TensorZero. They do not
attempt to comprehensively cover all of TensorZero's functionality.
See the tests across the Rust codebase for more comprehensive tests.

To run:
	go test
or with verbose output:
	go test -v
*/

package tests

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared/constant"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	Options []option.RequestOption
	client  openai.Client
	ctx     context.Context
)

func TestMain(m *testing.M) {
	ctx = context.Background()
	client = openai.NewClient(
		option.WithBaseURL("http://127.0.0.1:3000/openai/v1"),
		option.WithAPIKey("donotuse"),
	)

	// Run the tests and exit with the result code
	os.Exit(m.Run())
}

func OldFormatSystemMessageWithAssistant(t *testing.T, assistant_name string) *openai.ChatCompletionSystemMessageParam {
	t.Helper()

	sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
		"content": []map[string]interface{}{
			{"assistant_name": assistant_name},
		},
		"role": "system",
	})
	return &sysMsg
}

func addEpisodeIDToRequest(t *testing.T, req *openai.ChatCompletionNewParams, episodeID uuid.UUID) {
	t.Helper()
	// Add the episode ID to the request as an extra field
	req.WithExtraFields(map[string]any{
		"tensorzero::episode_id": episodeID.String(),
	})
}

// Test basic inference with old model format
func TestBasicInference(t *testing.T) {
	t.Run("Basic Inference using Old Model Format and Header", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages,
			Temperature: openai.Float(0.4),
		}
		req.WithExtraFields(map[string]any{
			"episode_id": episodeID.String(), //old format
		})

		// Send API request
		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// If episode_id is passed in the old format,
		// verify its presence in the response extras and ensure it's a valid UUID,
		// without checking the exact value.
		rawEpisodeID, ok := resp.JSON.ExtraFields["episode_id"]
		require.True(t, ok, "Response does not contain an episode_id")
		var responseEpisodeID string
		err = json.Unmarshal([]byte(rawEpisodeID.Raw()), &responseEpisodeID)
		require.NoError(t, err, "Failed to parse episode_id from response extras")
		_, err = uuid.Parse(responseEpisodeID)
		require.NoError(t, err, "Response episode_id is not a valid UUID")

		// Validate response fields
		assert.Equal(t, `Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.`,
			resp.Choices[0].Message.Content)

		// Validate Usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})
	// TODO: [test_async_basic_inference]
	t.Run("Basic Inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages,
			Temperature: openai.Float(0.4),
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send API request
		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate episode id
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}

		// Validate response fields
		assert.Equal(t, `Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.`,
			resp.Choices[0].Message.Content)

		// Validate Usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)

	})
	// TODO: [test_async_basic_inference_json_schema]
	// TODO: [test_async_inference_cache]
}

func TestStreamingInference(t *testing.T) {
	t.Run("it should handle streaming inference", func(t *testing.T) {
		startTime := time.Now()
		episodeID, _ := uuid.NewV7()
		expectedText := []string{
			"Wally,",
			" the",
			" golden",
			" retriever,",
			" wagged",
			" his",
			" tail",
			" excitedly",
			" as",
			" he",
			" devoured",
			" a",
			" slice",
			" of",
			" cheese",
			" pizza.",
		}
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::basic_test",
			Messages: messages,
			Seed:     openai.Int(69),
			StreamOptions: openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(true),
			},
			MaxTokens: openai.Int(300),
		}
		addEpisodeIDToRequest(t, req, episodeID)

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		var firstChunkDuration time.Duration // Variable to store the duration of the first chunk

		// Collecting all chunks
		var allChunks []openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			allChunks = append(allChunks, chunk)

			if firstChunkDuration == 0 {
				firstChunkDuration = time.Since(startTime)
			}
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")
		require.NotEmpty(t, allChunks, "No chunks were received")

		// Validate chunk duration
		lastChunkDuration := time.Since(startTime) - firstChunkDuration
		assert.Greater(t, lastChunkDuration.Seconds(), firstChunkDuration.Seconds()+0.1,
			"Last chunk duration should be greater than first chunk duration")

		// Validate the stop chunk
		require.GreaterOrEqual(t, len(allChunks), 2, "Expected at least two chunks, but got fewer")
		stopChunk := allChunks[len(allChunks)-2]
		assert.Empty(t, stopChunk.Choices[0].Delta.Content)
		assert.Equal(t, stopChunk.Choices[0].FinishReason, "stop")

		// Validate the Completion chunk
		completionChunk := allChunks[len(allChunks)-1]
		assert.Equal(t, int64(10), completionChunk.Usage.PromptTokens)
		assert.Equal(t, int64(16), completionChunk.Usage.CompletionTokens)
		assert.Equal(t, int64(26), completionChunk.Usage.TotalTokens)

		var previousInferenceID, previousEpisodeID string
		textIndex := 0
		// Validate the chunk Content
		for i := range len(allChunks) - 2 {
			chunk := allChunks[i]
			if len(chunk.Choices) == 0 {
				continue
			}
			// Validate the model
			assert.Equal(t, "tensorzero::function_name::basic_test::variant_name::test", chunk.Model, "Model mismatch")
			// Validate inference ID consistency
			if previousInferenceID != "" {
				assert.Equal(t, previousInferenceID, chunk.ID, "Inference ID should remain consistent across chunks")
			}
			var chunkResponseEpisodeID string
			if extra, ok := chunk.JSON.ExtraFields["episode_id"]; ok {
				err := json.Unmarshal([]byte(extra.Raw()), &chunkResponseEpisodeID)
				require.NoError(t, err, "Failed to parse episode_id from chunk extras")
				assert.Equal(t, episodeID.String(), chunkResponseEpisodeID, "Episode ID mismatch")
			} else {
				t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
			}
			// Validate episode ID consistency
			if previousEpisodeID != "" {
				assert.Equal(t, previousEpisodeID, chunkResponseEpisodeID, "Episode ID should remain consistent across chunks")
			}
			previousInferenceID = chunk.ID
			previousEpisodeID = chunkResponseEpisodeID

			// Validating the content
			content := chunk.Choices[0].Delta.Content
			if content != "" {
				assert.Equal(t, expectedText[i], content,
					"Content mismatch at index %d: expected '%s', got '%s'", i, expectedText[i], content)
				textIndex++ // inside the scope, not to increment for empty content
			}
		}
		assert.Equal(t, len(expectedText), textIndex, "Not all expected texts were matched")
	})
	// TODO: [test_async_inference_streaming_with_cache]
	// TODO: [test_async_inference_streaming_nonexistent_function]
	// TODO: [test_async_inference_streaming_missing_function]
	// TODO: [test_async_inference_streaming_missing_model]
	// TODO: [test_async_inference_streaming_malformed_function]
	// TODO: [test_async_inference_streaming_malformed_input]
}

func TestToolCallingInference(t *testing.T) {
	t.Run("it should handle tool calling inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hi I'm visiting Brooklyn from Brazil. What's the weather?"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::weather_helper",
			Messages: messages,
			TopP:     openai.Float(0.5),
		}
		addEpisodeIDToRequest(t, req, episodeID)

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate episode id
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id from response extras")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}
		//Validate the model
		assert.Equal(t, "tensorzero::function_name::weather_helper::variant_name::variant", resp.Model)
		// Validate the response
		assert.Empty(t, resp.Choices[0].Message.Content, "Message content should be empty")
		require.NotNil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should not be nil")
		//Validate the tool call details
		toolCalls := resp.Choices[0].Message.ToolCalls
		require.Len(t, toolCalls, 1, "There should be exactly one tool call")
		toolCall := toolCalls[0]
		assert.Equal(t, constant.Function("function"), toolCall.Type, "Tool call type should be 'function'")
		assert.Equal(t, "get_temperature", toolCall.Function.Name, "Function name should be 'get_temperature'")
		assert.Equal(t, `{"location":"Brooklyn","units":"celsius"}`, toolCall.Function.Arguments, "Function arguments do not match")
		// Validate the Usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	})

	t.Run("it should handle malformed tool call inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hi I'm visiting Brooklyn from Brazil. What's the weather?"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:           "tensorzero::function_name::weather_helper",
			Messages:        messages,
			PresencePenalty: openai.Float(0.5),
		}
		addEpisodeIDToRequest(t, req, episodeID)
		req.WithExtraFields(map[string]any{
			"tensorzero::variant_name": "bad_tool",
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::weather_helper::variant_name::bad_tool", resp.Model)
		// Validate the message content
		assert.Empty(t, resp.Choices[0].Message.Content, "Message content should be empty")
		// Validate tool calls
		require.NotNil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should not be nil")
		toolCalls := resp.Choices[0].Message.ToolCalls
		require.Equal(t, 1, len(toolCalls), "There should be exactly one tool call")
		toolCall := toolCalls[0]
		assert.Equal(t, constant.Function("function"), toolCall.Type, "Tool call type should be 'function'")
		assert.Equal(t, "get_temperature", toolCall.Function.Name, "Function name should be 'get_temperature'")
		assert.Equal(t, `{"location":"Brooklyn","units":"Celsius"}`, toolCall.Function.Arguments, "Function arguments do not match")
		// Validate usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	})

	t.Run("it should handle tool call streaming", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hi I'm visiting Brooklyn from Brazil. What's the weather?"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::weather_helper",
			Messages: messages,
			StreamOptions: openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(true),
			},
		}
		addEpisodeIDToRequest(t, req, episodeID)

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		var allChunks []openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			allChunks = append(allChunks, chunk)
		}

		expectedText := []string{
			`{"location"`,
			`:"Brooklyn"`,
			`,"units"`,
			`:"celsius`,
			`"}`,
		}

		// Validate the stop chunk
		require.GreaterOrEqual(t, len(allChunks), 2, "Expected at least two chunks, but got fewer")
		stopChunk := allChunks[len(allChunks)-2]
		assert.Empty(t, stopChunk.Choices[0].Delta.Content)
		assert.Empty(t, stopChunk.Choices[0].Delta.ToolCalls)
		assert.Equal(t, stopChunk.Choices[0].FinishReason, "tool_calls")

		// Validate the Completion chunk
		completionChunk := allChunks[len(allChunks)-1]
		assert.Equal(t, int64(10), completionChunk.Usage.PromptTokens)
		assert.Equal(t, int64(5), completionChunk.Usage.CompletionTokens)
		assert.Equal(t, int64(15), completionChunk.Usage.TotalTokens)

		var previousInferenceID, previousEpisodeID string
		//Test for intermediate chunks
		for i := range len(allChunks) - 2 {
			chunk := allChunks[i]
			if len(chunk.Choices) == 0 {
				continue
			}

			assert.Equal(t, "tensorzero::function_name::weather_helper::variant_name::variant", chunk.Model)
			// Validate inference ID consistency
			if previousInferenceID != "" {
				assert.Equal(t, previousInferenceID, chunk.ID, "Inference ID should remain consistent across chunks")
			}
			var chunkResponseEpisodeID string
			if extra, ok := chunk.JSON.ExtraFields["episode_id"]; ok {
				err := json.Unmarshal([]byte(extra.Raw()), &chunkResponseEpisodeID)
				require.NoError(t, err, "Failed to parse episode_id from chunk extras")
				assert.Equal(t, episodeID.String(), chunkResponseEpisodeID, "Episode ID mismatch")
			} else {
				t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
			}
			// Validate episode ID consistency
			if previousEpisodeID != "" {
				assert.Equal(t, previousEpisodeID, chunkResponseEpisodeID, "Episode ID should remain consistent across chunks")
			}
			previousInferenceID = chunk.ID
			previousEpisodeID = chunkResponseEpisodeID

			assert.Len(t, chunk.Choices, 1, "Each chunk should have exactly one choice")
			assert.Empty(t, chunk.Choices[0].Delta.Content, "Content should be empty for intermediate chunks")
			assert.Len(t, chunk.Choices[0].Delta.ToolCalls, 1, "Each intermediate chunk should have one tool call")

			toolCall := chunk.Choices[0].Delta.ToolCalls[0]
			//BUG : other toolcall.type arr of `constant.Function("function")`
			assert.Equal(t, "function", toolCall.Type, "Tool call type should be 'function'")
			assert.Equal(t, "get_temperature", toolCall.Function.Name, "Function name should be 'get_temperature'")
			assert.Equal(t, expectedText[i], toolCall.Function.Arguments, "Function arguments do not match expected text")
		}
	})

	// t.Run("it should handle streaming inference with malformed input", func(t *testing.T) {
	//     episodeID, _ := uuid.NewV7()

	//     messages := []openai.ChatCompletionMessageParamUnion{
	//         {OfSystem: openai.ChatCompletionSystemMessageParam{
	//             Content: []map[string]any{
	//                 {"name_of_assistant": "Alfred Pennyworth"},
	//             },
	//         }},
	//         openai.UserMessage("Hello"),
	//     }

	//     req := &openai.ChatCompletionNewParams{
	//         Model:    "tensorzero::function_name::basic_test",
	//         Messages: messages,
	//         Stream:   true,
	//     }
	//     req.WithExtraFields(map[string]any{
	//         "tensorzero::episode_id": episodeID.String(),
	//     })

	//     // Expect an error due to malformed input
	//     _, err := client.Chat.Completions.NewStreaming(ctx, *req)
	//     require.Error(t, err, "Expected an error due to malformed input")

	//     // Validate the error
	//     var apiErr *openai.APIError
	//     require.ErrorAs(t, err, &apiErr)
	//     assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 400")
	//     assert.Contains(t, apiErr.Message, "JSON Schema validation failed", "Expected JSON Schema validation error")
	// })
}
