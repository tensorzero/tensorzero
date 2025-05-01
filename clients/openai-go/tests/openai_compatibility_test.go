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

	// "time"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
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
		addEpisodeIDToRequest(t, req, episodeID)

		// Send API request
		resp, err := client.Chat.Completions.New(ctx, *req)
		require.NoError(t, err, "API request failed")

		// Validate response fields
		var responseEpisodeID string
		json.Unmarshal([]byte(resp.JSON.ExtraFields["episode_id"].Raw()), &responseEpisodeID)
		assert.Equal(t, episodeID.String(), responseEpisodeID)
		assert.Equal(t, `Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.`,
			resp.Choices[0].Message.Content)
		assert.Equal(t, int64(10), resp.Usage.PromptTokens)
		assert.Equal(t, int64(10), resp.Usage.CompletionTokens)
		assert.Equal(t, int64(20), resp.Usage.TotalTokens)
		assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	})
	// TODO: [test_async_basic_inference]
	// TODO: [test_async_basic_inference_json_schema]
	// TODO: [test_async_inference_cache]
}
