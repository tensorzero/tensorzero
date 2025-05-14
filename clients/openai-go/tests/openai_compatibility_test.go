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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
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

// func GenerateSchema[T any]() interface{} {
// 	// t.Helper()
// 	// Structured Outputs uses a subset of JSON schema.
// 	// These flags are necessary to comply with that subset.
// 	reflector := jsonschema.Reflector{
// 		AllowAdditionalProperties: false,
// 		DoNotReference:            true,
// 	}
// 	var v T
// 	schema := reflector.Reflect(v)
// 	// fmt.Println("Generated Schema :", schema)
// 	return schema
// }

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
	// t.Run("it should handle basic inference and validate response schema manually", func(t *testing.T) {
	// 	episodeID, _ := uuid.NewV7()

	// 	// Define the messages
	// 	messages := []openai.ChatCompletionMessageParamUnion{
	// 		{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
	// 		openai.UserMessage("Hello"),
	// 	}

	// 	// Define a dummy model with an intentionally incorrect schema
	// 	type DummyModel struct {
	// 		Name int `json:"name"` // Intentionally incorrect type for testing
	// 	}

	// 	// Define the JSON schema response format
	// 	schema := map[string]interface{}{
	// 		"type": "object",
	// 		"properties": map[string]interface{}{
	// 			"name": map[string]interface{}{
	// 				"type": "string", // Expecting a string, but the model might return something else
	// 			},
	// 		},
	// 		"required": []string{"name"},
	// 	}

	// 	responseFormat := openai.ResponseFormatJSONSchemaJSONSchemaParam{
	// 		Name:        "dummy_model",
	// 		Strict:      openai.Bool(true),
	// 		Description: openai.String("A dummy model for testing schema validation"),
	// 		Schema:      schema,
	// 	}

	// 	// Create the request
	// 	req := &openai.ChatCompletionNewParams{
	// 		Model:    "tensorzero::function_name::basic_test",
	// 		Messages: messages,
	// 		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
	// 			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
	// 				// Type:       "json_schema",
	// 				JSONSchema: responseFormat,
	// 			},
	// 		},
	// 		Temperature: openai.Float(0.4),
	// 	}
	// 	req.WithExtraFields(map[string]any{
	// 		"tensorzero::episode_id": episodeID.String(),
	// 	})

	// 	// Send the request
	// 	resp, err := client.Chat.Completions.New(ctx, *req)
	// 	require.NoError(t, err, "Unexpected error while getting completion")

	// 	// Debugging output
	// 	fmt.Printf("Response: %+v\n", resp.RawJSON())

	// 	// Unmarshal the response into the DummyModel struct
	// 	var dummyModel DummyModel
	// 	err = json.Unmarshal([]byte(resp.Choices[0].Message.Content), &dummyModel)
	// 	fmt.Println("########Error####", err)
	// 	require.Error(t, err, "Expected a validation error due to type mismatch")

	// 	// Validate the error message
	// 	expectedSubstring := "cannot unmarshal"
	// 	require.Contains(t, err.Error(), expectedSubstring, "Error message should mention unmarshal failure")

	// 	fmt.Println("Validation error simulated:", err.Error())
	// })

	// TODO: [test_async_inference_cache]
	t.Run("it should handle inference with cache", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		// First request (non-cached)
		req := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages,
			Temperature: openai.Float(0.4),
		}

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "Unexpected error while getting completion")

		// Validate the response
		require.NotNil(t, resp.Choices)
		require.Equal(t, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.", resp.Choices[0].Message.Content)

		// Validate usage
		require.NotNil(t, resp.Usage)
		require.Equal(t, int64(10), resp.Usage.PromptTokens)
		require.Equal(t, int64(10), resp.Usage.CompletionTokens)
		require.Equal(t, int64(20), resp.Usage.TotalTokens)

		// Second request (cached)
		req.WithExtraFields(map[string]any{
			"tensorzero::cache_options": map[string]any{
				"max_age_s": 10,
				"enabled":   "on",
			},
		})

		cachedResp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "Unexpected error while getting cached completion")

		// Validate the cached response
		require.NotNil(t, cachedResp.Choices)
		require.Equal(t, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.", cachedResp.Choices[0].Message.Content)

		// Validate cached usage
		require.NotNil(t, cachedResp.Usage)
		require.Equal(t, int64(0), cachedResp.Usage.PromptTokens)
		require.Equal(t, int64(0), cachedResp.Usage.CompletionTokens)
		require.Equal(t, int64(0), cachedResp.Usage.TotalTokens)
	})
	// TODO [should handle json success with non-deprecated format]
	t.Run("it should handle JSON success with non-deprecated format", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
			"content": []map[string]interface{}{
				{
					"type": "text",
					"tensorzero::arguments": map[string]interface{}{
						"assistant_name": "Alfred Pennyworth",
					},
				},
			},
			"role": "system",
		})

		userMsg := param.OverrideObj[openai.ChatCompletionUserMessageParam](map[string]interface{}{
			"content": []map[string]interface{}{
				{
					"type": "text",
					"tensorzero::arguments": map[string]interface{}{
						"country": "Japan",
					},
				},
			},
			"role": "user",
		})

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &sysMsg},
			{OfUser: &userMsg},
		}

		// Create the request
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_success",
			Messages: messages,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::json_success::variant_name::test", resp.Model)

		// Validate the episode ID
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id from response extras")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}

		// Validate the response content
		assert.Equal(t, `{"answer":"Hello"}`, resp.Choices[0].Message.Content)
		// Validate tool calls
		assert.Nil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should be nil")
		// Validate usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
	})
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
	t.Run("it should handle streaming inference with cache", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

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

		// First request without cache to populate the cache
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::basic_test",
			Messages: messages,
			Seed:     openai.Int(69),
			StreamOptions: openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(true),
			},
		}
		addEpisodeIDToRequest(t, req, episodeID)

		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

		var chunks []openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			chunks = append(chunks, chunk)
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")
		require.NotEmpty(t, chunks, "No chunks were received")

		// Verify the response
		content := ""
		for i, chunk := range chunks[:len(chunks)-1] { // All but the last chunk
			if i < len(expectedText) {
				require.Equal(t, expectedText[i], chunk.Choices[0].Delta.Content)
				content += chunk.Choices[0].Delta.Content
			}
		}

		// Check second-to-last chunk has correct finish reason
		stopChunk := chunks[len(chunks)-2]
		require.Equal(t, "stop", stopChunk.Choices[0].FinishReason)

		finalChunk := chunks[len(chunks)-1]
		require.Equal(t, int64(10), finalChunk.Usage.PromptTokens)
		require.Equal(t, int64(16), finalChunk.Usage.CompletionTokens)

		// Simulate waiting for trailing cache write
		time.Sleep(1 * time.Second)

		// Second request with cache
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
			"tensorzero::cache_options": map[string]any{
				"max_age_s": nil,
				"enabled":   "on",
			},
		})

		cachedStream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, cachedStream, "Cached streaming response should not be nil")

		var cachedChunks []openai.ChatCompletionChunk
		for cachedStream.Next() {
			chunk := cachedStream.Current()
			cachedChunks = append(cachedChunks, chunk)
		}
		require.NoError(t, cachedStream.Err(), "Cached stream encountered an error")
		require.NotEmpty(t, cachedChunks, "No cached chunks were received")

		// Verify we get the same content
		cachedContent := ""
		for i, chunk := range cachedChunks[:len(cachedChunks)-1] { // All but the last chunk
			if i < len(expectedText) {
				require.Equal(t, expectedText[i], chunk.Choices[0].Delta.Content)
				cachedContent += chunk.Choices[0].Delta.Content
			}
		}
		require.Equal(t, content, cachedContent)

		// Check second-to-last chunk has the correct finish reason
		finishChunk := cachedChunks[len(cachedChunks)-2]
		require.Equal(t, "stop", finishChunk.Choices[0].FinishReason)
		// Verify zero usage
		finalCachedChunk := cachedChunks[len(cachedChunks)-1]
		require.Equal(t, int64(0), finalCachedChunk.Usage.PromptTokens)
		require.Equal(t, int64(0), finalCachedChunk.Usage.CompletionTokens)
		require.Equal(t, int64(0), finalCachedChunk.Usage.TotalTokens)
	})
	// TODO: [test_async_inference_streaming_nonexistent_function]
	t.Run("it should handle streaming inference with a nonexistent function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::does_not_exist", // Nonexistent function
			Messages: messages,
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request and expect an error
		_, err := client.Chat.Completions.New(ctx, *req)
		// fmt.Println("########Error####", err)
		require.Error(t, err, "Expected an error for nonexistent function")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError") // ErrorAs assign err to apiErr
		assert.Equal(t, 404, apiErr.StatusCode, "Expected status code 404")
		assert.Contains(t, err.Error(), "404 Not Found \"Unknown function: does_not_exist\"", "Error should indicate 404 Not Found")
	})
	// TODO: [test_async_inference_streaming_missing_function]
	t.Run("it should handle streaming inference with a missing function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::", // missing function
			Messages: messages,
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request and expect an error
		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for nonexistent function")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 404")
		assert.Contains(t, apiErr.Error(), "400 Bad Request", "Error should indicate 400 Bad Request")
	})
	// TODO: [test_async_inference_streaming_malformed_function]
	t.Run("it should handle streaming inference with a malformed function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "chatgpt", // malformed function
			Messages: messages,
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request and expect an error
		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for nonexistent function")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 404")
		assert.Contains(t, apiErr.Error(), "400 Bad Request \"Invalid request to OpenAI-compatible endpoint", "Error should indicate invalid request to OpenAI compartible endpoint")
	})
	// TODO: [test_async_inference_streaming_missing_model]
	t.Run("it should handle streaming inference with a missing model", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Messages: messages,
			// Missing model
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request and expect an error
		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for nonexistent function")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 404")
		assert.Contains(t, apiErr.Error(), "missing field `model`", "Error should indicate model field is missing")
	})
	// TODO: [test_async_inference_streaming_malformed_input]
	t.Run("it should handle streaming inference with a missing model", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
			"content": []map[string]interface{}{
				{"name_of_assistant": "Alfred Pennyworth"},
			},
			"role": "system",
		})

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: &sysMsg}, //malformed sys message
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Messages: messages,
			Model:    "tensorzero::function_name::basic_test",
			StreamOptions: openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(true),
			},
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request and expect an error
		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for nonexistent function")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 404")
		assert.Contains(t, apiErr.Error(), "JSON Schema validation failed", "Error should indicate JSON schema validation failed")
	})
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

	t.Run("it should handle dynamic tool use inference with OpenAI", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		// Define the messages
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: OldFormatSystemMessageWithAssistant(t, "Dr. Mehta")},
			openai.UserMessage("What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."),
		}

		tools := []openai.ChatCompletionToolParam{
			{
				Function: openai.FunctionDefinitionParam{
					Name:        "get_temperature",
					Description: openai.String("Get the current temperature in a given location"),
					Parameters: openai.FunctionParameters{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]string{
								"type":        "string",
								"description": "The location to get the temperature for (e.g. 'New York')",
							},
							"units": map[string]interface{}{
								"type":        "string",
								"description": "The units to get the temperature in (must be 'fahrenheit' or 'celsius')",
								"enum":        []string{"fahrenheit", "celsius"},
							},
						},
						"required":             []string{"location"},
						"additionalProperties": false,
					},
				},
			},
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::basic_test",
			Messages: messages,
			Tools:    tools,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "openai",
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::basic_test::variant_name::openai", resp.Model)

		// Validate the episode ID
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id from response extras")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}

		// Validate the response content
		assert.Empty(t, resp.Choices[0].Message.Content, "Message content should be nil")

		// // Validate tool calls
		require.NotNil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should not be nil")
		require.Len(t, resp.Choices[0].Message.ToolCalls, 1, "There should be exactly one tool call")

		toolCall := resp.Choices[0].Message.ToolCalls[0]
		assert.Equal(t, "function", string(toolCall.Type), "Tool call type should be 'function'")
		assert.Equal(t, "get_temperature", toolCall.Function.Name, "Function name should be 'get_temperature'")

		var arguments map[string]interface{}
		err = json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments)
		require.NoError(t, err, "Failed to parse tool call arguments")
		assert.Equal(t, map[string]interface{}{
			"location": "Tokyo",
			"units":    "celsius",
		}, arguments, "Tool call arguments do not match")

		// Validate usage
		require.NotNil(t, resp.Usage, "Usage should not be nil")
		assert.Greater(t, resp.Usage.PromptTokens, int64(100), "Prompt tokens should be greater than 100")
		assert.Greater(t, resp.Usage.CompletionTokens, int64(10), "Completion tokens should be greater than 10")
	})

}

func TestImageInference(t *testing.T) {
	t.Run("it should handle image inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		usrMsg := openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png",
			}),
			openai.TextContentPart("Output exactly two words describing the image"),
		})
		messages := []openai.ChatCompletionMessageParamUnion{
			usrMsg,
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::openai::gpt-4o-mini",
			Messages: messages,
		}
		addEpisodeIDToRequest(t, req, episodeID)

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::model_name::openai::gpt-4o-mini", resp.Model)

		// Validate the episode ID
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id from response extras")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}

		// Validate the response content
		assert.Contains(t, resp.Choices[0].Message.Content, "crab")
	})

	t.Run("it should handle multi-block image_base64", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		// Read image and convert to base64
		imagePath := "g:/tensorzero/tensorzero-internal/tests/e2e/providers/ferris.png"
		imageData, err := os.ReadFile(imagePath)
		require.NoError(t, err, "Failed to read image file")
		imageBase64 := base64.StdEncoding.EncodeToString(imageData)

		// Define the messages
		usrMsg := openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Output exactly two words describing the image"),
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: fmt.Sprintf("data:image/png;base64,%s", imageBase64),
			}),
		})
		messages := []openai.ChatCompletionMessageParamUnion{
			usrMsg,
		}

		// Create the request
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::openai::gpt-4o-mini",
			Messages: messages,
		}
		addEpisodeIDToRequest(t, req, episodeID)

		// Send the request
		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::model_name::openai::gpt-4o-mini", resp.Model)

		// Validate the episode ID
		if extra, ok := resp.JSON.ExtraFields["episode_id"]; ok {
			var responseEpisodeID string
			err := json.Unmarshal([]byte(extra.Raw()), &responseEpisodeID)
			require.NoError(t, err, "Failed to parse episode_id from response extras")
			assert.Equal(t, episodeID.String(), responseEpisodeID)
		} else {
			t.Errorf("Key 'tensorzero::episode_id' not found in response extras")
		}

		// Validate the response content
		require.NotNil(t, resp.Choices[0].Message.Content, "Message content should not be nil")
		assert.Contains(t, strings.ToLower(resp.Choices[0].Message.Content), "crab", "Response should contain 'crab'")
	})
}
