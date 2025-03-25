import { TensorZeroGateway } from '../src';

// Simple example of using the TensorZero JavaScript client to create a chat application
async function main() {
  // Create a TensorZero client instance
  const client = new TensorZeroGateway({
    gatewayUrl: 'http://localhost:3000', // Replace with your gateway URL
    apiKey: process.env.TENSORZERO_API_KEY, // Optional API key
  });

  try {
    // Non-streaming inference example
    console.log('Making a non-streaming inference request...');
    const nonStreamingResponse = await client.inference({
      modelName: 'openai::gpt-4o-mini', // Replace with your model
      input: {
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'What is the capital of France?' },
        ],
      },
      params: {
        temperature: 0.7,
        max_tokens: 100,
      },
    });

    console.log('\nNon-streaming response:');
    console.log(`ID: ${nonStreamingResponse.inference_id}`);
    console.log(`Model: ${nonStreamingResponse.model}`);
    console.log(`Content: ${nonStreamingResponse.output.content}`);
    console.log(`Tokens: ${nonStreamingResponse.usage.total_tokens}`);

    // Send feedback for the inference
    const feedbackResponse = await client.feedback({
      metricName: 'helpful',
      inferenceId: nonStreamingResponse.inference_id,
      value: true,
      tags: {
        category: 'geography',
      },
    });

    console.log('\nFeedback sent:');
    console.log(`Feedback ID: ${feedbackResponse.feedback_id}`);
    console.log(`Metric: ${feedbackResponse.metric_name}`);
    console.log(`Value: ${feedbackResponse.value}`);

    // Streaming inference example
    console.log('\nMaking a streaming inference request...');
    const stream = await client.inferenceStream({
      modelName: 'openai::gpt-4o-mini', // Replace with your model
      input: {
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Tell me about the solar system.' },
        ],
      },
      params: {
        temperature: 0.7,
        max_tokens: 200,
      },
    });

    console.log('\nStreaming response:');
    let inferenceId = '';
    let fullContent = '';

    // Process the stream chunks
    for await (const chunk of stream) {
      inferenceId = chunk.inference_id;
      
      if (chunk.chunk.content) {
        process.stdout.write(chunk.chunk.content);
        fullContent += chunk.chunk.content;
      }
      
      if (chunk.is_final && chunk.usage) {
        console.log(`\n\nTotal tokens: ${chunk.usage.total_tokens}`);
      }
    }

    // Send feedback for the streaming inference
    if (inferenceId) {
      const streamFeedbackResponse = await client.feedback({
        metricName: 'informative',
        inferenceId: inferenceId,
        value: true,
        tags: {
          category: 'science',
        },
      });

      console.log('\nFeedback sent for streaming inference:');
      console.log(`Feedback ID: ${streamFeedbackResponse.feedback_id}`);
    }

  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
main().catch(console.error); 