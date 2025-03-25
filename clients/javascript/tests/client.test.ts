// Import necessary libraries for testing
import axios from 'axios';
import { Readable } from 'stream';
import { TensorZeroGateway } from '../src/client';
import {
  InferenceRequest,
  InferenceResponse,
  FeedbackRequest,
  FeedbackResponse,
} from '../src/types';

// Mock axios for testing
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('TensorZeroGateway', () => {
  let client: TensorZeroGateway;

  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks();
    
    // Create a new client instance for each test
    client = new TensorZeroGateway({
      gatewayUrl: 'http://localhost:3000',
      apiKey: 'test-api-key',
    });
    
    // Setup the axios create mock
    mockedAxios.create.mockReturnValue(mockedAxios as any);
  });

  describe('inference', () => {
    it('should make a post request to the inference endpoint', async () => {
      // Setup mock response
      const mockResponse: InferenceResponse = {
        inference_id: 'test-id',
        model: 'openai::gpt-4o-mini',
        output: {
          content: 'Hello, world!',
        },
        input: {
          messages: [
            { role: 'user', content: 'Hi there' },
          ],
        },
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
        },
        created_at: new Date().toISOString(),
      };
      
      // Setup the mock implementation
      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });
      
      // Create request parameters
      const request: InferenceRequest = {
        modelName: 'openai::gpt-4o-mini',
        input: {
          messages: [
            { role: 'user', content: 'Hi there' },
          ],
        },
      };
      
      // Execute the method
      const response = await client.inference(request);
      
      // Verify the response
      expect(response).toEqual(mockResponse);
      
      // Verify the request
      expect(mockedAxios.post).toHaveBeenCalledWith('/inference', {
        model_name: 'openai::gpt-4o-mini',
        input: {
          messages: [
            { role: 'user', content: 'Hi there' },
          ],
        },
        params: undefined,
        tags: undefined,
        functions: undefined,
        tools: undefined,
      });
    });
    
    it('should properly handle errors', async () => {
      // Setup the mock implementation to throw an error
      const errorMessage = 'API Error';
      mockedAxios.isAxiosError.mockReturnValueOnce(true);
      mockedAxios.post.mockRejectedValueOnce({
        isAxiosError: true,
        message: errorMessage,
        response: {
          data: {
            error: 'Invalid input',
          },
        },
      });
      
      // Create request parameters
      const request: InferenceRequest = {
        modelName: 'openai::gpt-4o-mini',
        input: {
          messages: [
            { role: 'user', content: 'Hi there' },
          ],
        },
      };
      
      // Execute the method and expect it to throw
      await expect(client.inference(request)).rejects.toThrow('TensorZero Gateway error: Invalid input');
    });
  });
  
  describe('feedback', () => {
    it('should make a post request to the feedback endpoint', async () => {
      // Setup mock response
      const mockResponse: FeedbackResponse = {
        feedback_id: 'feedback-test-id',
        inference_id: 'inference-test-id',
        metric_name: 'thumbs_up',
        value: true,
        created_at: new Date().toISOString(),
      };
      
      // Setup the mock implementation
      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });
      
      // Create request parameters
      const request: FeedbackRequest = {
        metricName: 'thumbs_up',
        inferenceId: 'inference-test-id',
        value: true,
      };
      
      // Execute the method
      const response = await client.feedback(request);
      
      // Verify the response
      expect(response).toEqual(mockResponse);
      
      // Verify the request
      expect(mockedAxios.post).toHaveBeenCalledWith('/feedback', {
        metric_name: 'thumbs_up',
        inference_id: 'inference-test-id',
        value: true,
        tags: undefined,
      });
    });
  });
}); 