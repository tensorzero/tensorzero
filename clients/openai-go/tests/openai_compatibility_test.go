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
	"net/http"
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

func systemMessageWithAssistant(t *testing.T, assistant_name string) *openai.ChatCompletionSystemMessageParam {
	t.Helper()

	sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
		"content": []map[string]interface{}{
			{
				"type": "text",
				"tensorzero::arguments": map[string]interface{}{
					"assistant_name": assistant_name,
				},
			},
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

func sendRequestTzGateway(t *testing.T, body map[string]interface{}) (map[string]interface{}, error) {
	// Send a request to the TensorZero gateway
	t.Helper()
	url := "http://127.0.0.1:3000/openai/v1/chat/completions"
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequest("POST", url, strings.NewReader(string(jsonBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer donotuse")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP error! status: %d", resp.StatusCode)
	}

	var responseBody map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&responseBody)
	if err != nil {
		return nil, fmt.Errorf("failed to decode response body: %w", err)
	}

	return responseBody, nil
}

func TestTags(t *testing.T) {
	t.Run("Test tensorzero tags", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages,
			Temperature: openai.Float(0.4),
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
			"tensorzero::tags":       map[string]any{"foo": "bar"},
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})
}

func TestMultiStep(t *testing.T) {
	t.Run("Test multi-step inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages,
			Temperature: openai.Float(0.4),
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "test",
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)

		messages2 := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
			openai.AssistantMessage(resp.Choices[0].Message.Content),
			openai.UserMessage("Greetings"),
		}

		req2 := &openai.ChatCompletionNewParams{
			Model:       "tensorzero::function_name::basic_test",
			Messages:    messages2,
			Temperature: openai.Float(0.4),
		}
		req2.WithExtraFields(map[string]any{
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "test2",
		})

		// Send API request
		resp2, err := client.Chat.Completions.New(ctx, *req2)
		require.NoError(t, err, "API request failed")

		// If episode_id is passed in the old format,
		// verify its presence in the response extras and ensure it's a valid UUID,
		// without checking the exact value.
		rawEpisodeID, ok = resp2.JSON.ExtraFields["episode_id"]
		require.True(t, ok, "Response does not contain an episode_id")
		err = json.Unmarshal([]byte(rawEpisodeID.Raw()), &responseEpisodeID)
		require.NoError(t, err, "Failed to parse episode_id from response extras")
		_, err = uuid.Parse(responseEpisodeID)
		require.NoError(t, err, "Response episode_id is not a valid UUID")

		// Validate response fields
		assert.Equal(t, `Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.`,
			resp2.Choices[0].Message.Content)

		// Validate Usage
		assert.Equal(t, int64(10), resp2.Usage.PromptTokens)
		assert.Equal(t, int64(1), resp2.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp2.Usage.TotalTokens)
		assert.Equal(t, "stop", resp2.Choices[0].FinishReason)

		// Ensure there are two unique responses on one episode ID.
		require.True(t, resp2.JSON.ExtraFields["episode_id"].Raw() == resp.JSON.ExtraFields["episode_id"].Raw(),
			"Second response episode ID must be the same as the first")
		require.False(t, resp2.ID == resp.ID, "Response IDs must be unique")
	})
}

// Test basic inference with old model format
func TestBasicInference(t *testing.T) {
	t.Run("Basic Inference using Old Model Format and Header", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})
	// TODO: [test_async_basic_inference]
	t.Run("Basic Inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)

	})

	t.Run("it should handle basic json schema parsing and throw proper validation error", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello"),
		}

		responseSchema := map[string]interface{}{
			"name": "string",
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_success",
			Messages: messages,
			ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        "response_schema",
						Strict:      openai.Bool(true),
						Description: openai.String("Schema for response validation"),
						Schema:      responseSchema,
					},
				},
			},
		}

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected to raise Error")

		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 400 Bad Request")
		assert.Contains(t, err.Error(), "JSON Schema validation failed", "Error should indicate JSON Schema validation failure")
	})

	t.Run("it should handle inference with cache", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		require.Equal(t, int64(1), resp.Usage.CompletionTokens)
		require.Equal(t, int64(11), resp.Usage.TotalTokens)

		// Sleep for 1s
		time.Sleep(time.Second)

		// Second request (cached)
		req.WithExtraFields(map[string]any{
			"tensorzero::*.tomltions": map[string]any{
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
	})

	t.Run("it should handle chat function null response", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("No yapping!"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::null_chat",
			Messages: messages,
		}

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::null_chat::variant_name::variant", resp.Model)

		// Validate the response content
		assert.Empty(t, resp.Choices[0].Message.Content, "Message content should be nil")
	})

	t.Run("it should handle json function null response", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Extract no data!"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::null_json",
			Messages: messages,
		}

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::null_json::variant_name::variant", resp.Model)

		// Validate the response content
		assert.Empty(t, resp.Choices[0].Message.Content, "Message content should be empty")
	})

	t.Run("it should handle extra headers parameter", func(t *testing.T) {
		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello, world!"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::dummy::echo_extra_info",
			Messages: messages,
		}

		req.WithExtraFields(map[string]any{
			"tensorzero::extra_headers": []map[string]any{
				{
					"model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
					"name":                "x-my-extra-header",
					"value":               "my-extra-header-value",
				},
			},
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::model_name::dummy::echo_extra_info", resp.Model)

		// Validate the response content
		var content map[string]interface{}
		err = json.Unmarshal([]byte(resp.Choices[0].Message.Content), &content)
		require.NoError(t, err, "Failed to parse response content")

		expectedContent := map[string]interface{}{
			"extra_body": map[string]interface{}{
				"inference_extra_body": []interface{}{},
			},
			"extra_headers": map[string]interface{}{
				"inference_extra_headers": []interface{}{
					map[string]interface{}{
						"model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
						"name":                "x-my-extra-header",
						"value":               "my-extra-header-value",
					},
				},
				"variant_extra_headers": nil,
			},
		}
		assert.Equal(t, expectedContent, content, "Response content does not match expected content")
	})

	t.Run("it should handle extra body parameter", func(t *testing.T) {

		messages := []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Hello, world!"),
		}

		// request with extra body
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::model_name::dummy::echo_extra_info",
			Messages: messages,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::extra_body": []map[string]any{
				{
					"model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
					"pointer":             "/thinking",
					"value": map[string]any{
						"type":          "enabled",
						"budget_tokens": 1024,
					},
				},
			},
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::model_name::dummy::echo_extra_info", resp.Model)

		// Validate the response content
		var content map[string]interface{}
		err = json.Unmarshal([]byte(resp.Choices[0].Message.Content), &content)
		require.NoError(t, err, "Failed to parse response content")

		expectedContent := map[string]interface{}{
			"extra_body": map[string]interface{}{
				"inference_extra_body": []interface{}{
					map[string]interface{}{
						"model_provider_name": "tensorzero::model_name::dummy::echo_extra_info::provider_name::dummy",
						"pointer":             "/thinking",
						"value": map[string]interface{}{
							"type":          "enabled",
							"budget_tokens": float64(1024),
						},
					},
				},
			},
			"extra_headers": map[string]interface{}{
				"variant_extra_headers":   nil,
				"inference_extra_headers": []interface{}{},
			},
		}
		assert.Equal(t, expectedContent, content, "Response content does not match expected content")
	})

	t.Run("it should handle json success", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

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
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			{OfUser: &userMsg},
		}

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
		assert.Nil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should be nil")

		// Validate usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
	})

	t.Run("it should handle json invalid system", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
			"content": []map[string]interface{}{
				{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "https://example.com/image.jpg",
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

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_success",
			Messages: messages,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for invalid system message")

		// Validate the error
		assert.Contains(t, err.Error(), "System message must contain only text or template content blocks", "Error should indicate invalid system message")
	})

	t.Run("it should handle json failure", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hello, world!"),
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_fail",
			Messages: messages,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the model
		assert.Equal(t, "tensorzero::function_name::json_fail::variant_name::test", resp.Model)

		// Validate the response content
		assert.Equal(t, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.", resp.Choices[0].Message.Content)
		assert.Nil(t, resp.Choices[0].Message.ToolCalls, "Tool calls should be nil")

		// Validate usage
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
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
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with cache", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()
		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with a nonexistent function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with a missing function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with a malformed function", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with a missing model", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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

	t.Run("it should handle streaming inference with a missing model", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		sysMsg := param.OverrideObj[openai.ChatCompletionSystemMessageParam](map[string]interface{}{
			"content": []map[string]interface{}{
				{
					"type": "text",
					"tensorzero::arguments": map[string]interface{}{
						"name_of_assistant": "Alfred Pennyworth",
					},
				},
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

	t.Run("it should handle JSON streaming", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

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
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			{OfUser: &userMsg},
		}

		// Create the request
		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_success",
			Messages: messages,
			StreamOptions: openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.Bool(false), // No usage information
			},
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "test-diff-schema",
		})

		// Start streaming
		stream := client.Chat.Completions.NewStreaming(ctx, *req)
		require.NotNil(t, stream, "Streaming response should not be nil")

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

		var allChunks []openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			allChunks = append(allChunks, chunk)
		}
		require.NoError(t, stream.Err(), "Stream encountered an error")
		require.NotEmpty(t, allChunks, "No chunks were received")

		// Validate the stop chunk
		stopChunk := allChunks[len(allChunks)-1]
		assert.Empty(t, stopChunk.Choices[0].Delta.Content)
		assert.Equal(t, stopChunk.Choices[0].FinishReason, "stop")
		assert.Empty(t, stopChunk.Usage, "Usage should be empty for streaming with no usage information")

		var previousInferenceID, previousEpisodeID string
		textIndex := 0
		// Validate the chunk Content
		for i := range len(allChunks) - 1 {
			chunk := allChunks[i]
			if len(chunk.Choices) == 0 {
				continue
			}
			// Validate the model
			assert.Equal(t, "tensorzero::function_name::json_success::variant_name::test-diff-schema", chunk.Model, "Model mismatch")
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

}

func TestToolCallingInference(t *testing.T) {
	t.Run("it should handle tool calling inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	})

	t.Run("it should handle malformed tool call inference", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		assert.Equal(t, int64(1), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(11), resp.Usage.TotalTokens)
		assert.Equal(t, "tool_calls", resp.Choices[0].FinishReason)
	})

	t.Run("it should handle tool call streaming", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
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
		nameSeen := false
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
			if toolCall.Function.Name != "" {
				assert.False(t, nameSeen, "Function name should only appear once")
				assert.Equal(t, "get_temperature", toolCall.Function.Name, "Function name should be 'get_temperature'")
				nameSeen = true
			}
			assert.Equal(t, expectedText[i], toolCall.Function.Arguments, "Function arguments do not match expected text")
		}
		assert.True(t, nameSeen, "Function name should have been seen in at least one chunk")
	})

	t.Run("it should handle dynamic tool use inference with OpenAI", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Dr. Mehta")},
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

		// Validate tool calls
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

	t.Run("it should reject string input for function with input schema", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		usrMsg := param.OverrideObj[openai.ChatCompletionUserMessageParam](map[string]interface{}{
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
			{OfSystem: systemMessageWithAssistant(t, "Alfred Pennyworth")},
			openai.UserMessage("Hi how are you?"),
			{OfUser: &usrMsg},
		}

		req := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::json_success",
			Messages: messages,
		}
		req.WithExtraFields(map[string]any{
			"tensorzero::episode_id": episodeID.String(),
		})

		_, err := client.Chat.Completions.New(ctx, *req)
		require.Error(t, err, "Expected an error for invalid input schema")

		// Validate the error
		var apiErr *openai.Error
		assert.ErrorAs(t, err, &apiErr, "Expected error to be of type APIError")
		assert.Equal(t, 400, apiErr.StatusCode, "Expected status code 400")
		assert.Contains(t, apiErr.Error(), "JSON Schema validation failed", "Error should indicate JSON Schema validation failure")
	})

	t.Run("it should handle multi-turn parallel tool calls", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []openai.ChatCompletionMessageParamUnion{
			{OfSystem: systemMessageWithAssistant(t, "Dr. Mehta")},
			openai.UserMessage("What is the weather like in Tokyo? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."),
		}

		req := &openai.ChatCompletionNewParams{
			Model:             "tensorzero::function_name::weather_helper_parallel",
			Messages:          messages,
			ParallelToolCalls: openai.Bool(true),
		}
		addEpisodeIDToRequest(t, req, episodeID)
		req.WithExtraFields(map[string]any{
			"tensorzero::variant_name": "openai",
		})

		// Initial request
		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate the assistant's response
		assistantMessage := resp.Choices[0].Message
		messages = append(messages, assistantMessage.ToParam())
		require.NotNil(t, assistantMessage.ToolCalls, "Tool calls should not be nil")
		require.Len(t, assistantMessage.ToolCalls, 2, "There should be exactly two tool calls")

		// Handle tool calls
		for _, toolCall := range assistantMessage.ToolCalls {
			if toolCall.Function.Name == "get_temperature" {
				messages = append(messages, openai.ToolMessage("70", toolCall.ID))
			} else if toolCall.Function.Name == "get_humidity" {
				messages = append(messages, openai.ToolMessage("30", toolCall.ID))
			} else {
				t.Fatalf("Unknown tool call: %s", toolCall.Function.Name)
			}
		}

		finalReq := &openai.ChatCompletionNewParams{
			Model:    "tensorzero::function_name::weather_helper_parallel",
			Messages: messages,
		}
		addEpisodeIDToRequest(t, finalReq, episodeID)
		finalReq.WithExtraFields(map[string]any{
			"tensorzero::variant_name": "openai",
		})

		// mullti-turn/final request
		finalResp, err := client.Chat.Completions.New(ctx, *finalReq)
		require.NoError(t, err, "API request failed")

		// Validate the final assistant's response
		finalAssistantMessage := finalResp.Choices[0].Message
		require.NotNil(t, finalAssistantMessage.Content, "Final assistant message content should not be nil")
		assert.Contains(t, finalAssistantMessage.Content, "70", "Final response should contain '70'")
		assert.Contains(t, finalAssistantMessage.Content, "30", "Final response should contain '30'")
	})

	t.Run("it should handle multi-turn parallel tool calls using TensorZero gateway directly", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		messages := []map[string]interface{}{
			{
				"role": "system",
				"content": []map[string]interface{}{
					{
						"type": "text",
						"tensorzero::arguments": map[string]string{
							"assistant_name": "Dr. Mehta",
						},
					},
				},
			},
			{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type": "text",
						"text": "What is the weather like in Tokyo (in Fahrenheit)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions.",
					},
				},
			},
		}

		// First request to get tool calls
		firstRequestBody := map[string]interface{}{
			"messages":                 messages,
			"model":                    "tensorzero::function_name::weather_helper_parallel",
			"parallel_tool_calls":      true,
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "openai",
		}

		// Initial request
		firstResponse, err := sendRequestTzGateway(t, firstRequestBody)
		require.NoError(t, err, "First API request failed")

		// Validate the assistant's response
		assistantMessage := firstResponse["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
		messages = append(messages, assistantMessage)

		toolCalls := assistantMessage["tool_calls"].([]interface{})
		require.Len(t, toolCalls, 2, "There should be exactly two tool calls")

		// Handle tool calls
		for _, toolCall := range toolCalls {
			toolCallMap := toolCall.(map[string]interface{})
			toolName := toolCallMap["function"].(map[string]interface{})["name"].(string)
			toolCallID := toolCallMap["id"].(string)

			if toolName == "get_temperature" {
				messages = append(messages, map[string]interface{}{
					"role": "tool",
					"content": []map[string]interface{}{
						{"type": "text", "text": "70"},
					},
					"tool_call_id": toolCallID,
				})
			} else if toolName == "get_humidity" {
				messages = append(messages, map[string]interface{}{
					"role": "tool",
					"content": []map[string]interface{}{
						{"type": "text", "text": "30"},
					},
					"tool_call_id": toolCallID,
				})
			} else {
				t.Fatalf("Unknown tool call: %s", toolName)
			}
		}

		secondRequestBody := map[string]interface{}{
			"messages":                 messages,
			"model":                    "tensorzero::function_name::weather_helper_parallel",
			"tensorzero::episode_id":   episodeID.String(),
			"tensorzero::variant_name": "openai",
		}

		secondResponse, err := sendRequestTzGateway(t, secondRequestBody)
		require.NoError(t, err, "Second request failed")

		finalAssistantMessage := secondResponse["choices"].([]interface{})[0].(map[string]interface{})["message"].(map[string]interface{})
		finalContent := finalAssistantMessage["content"].(string)

		// Validate the final assistant's response
		assert.Contains(t, finalContent, "70", "Final response should contain '70'")
		assert.Contains(t, finalContent, "30", "Final response should contain '30'")
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
		assert.Contains(t, strings.ToLower(resp.Choices[0].Message.Content), "crab", "Response should contain 'crab'")
	})

	t.Run("it should handle multi-block image_base64", func(t *testing.T) {
		episodeID, _ := uuid.NewV7()

		// Read image and convert to base64
		imagePath := "../../../tensorzero-core/tests/e2e/providers/ferris.png"
		imageData, err := os.ReadFile(imagePath)
		require.NoError(t, err, "Failed to read image file")
		imageBase64 := base64.StdEncoding.EncodeToString(imageData)

		usrMsg := openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
			openai.TextContentPart("Output exactly two words describing the image"),
			openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
				URL: fmt.Sprintf("data:image/png;base64,%s", imageBase64),
			}),
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
		require.NotNil(t, resp.Choices[0].Message.Content, "Message content should not be nil")
		assert.Contains(t, strings.ToLower(resp.Choices[0].Message.Content), "crab", "Response should contain 'crab'")
	})
}
