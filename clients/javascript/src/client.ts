import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import { Readable } from 'stream';
import {
  TensorZeroGatewayOptions,
  InferenceRequest,
  InferenceResponse,
  InferenceStreamChunk,
  FeedbackRequest,
  FeedbackResponse,
} from './types';

// Define an interface for the error response
interface TensorZeroErrorResponse {
  error?: string;
}

/**
 * TensorZero JavaScript client implementation
 */
export class TensorZeroGateway {
  private client: AxiosInstance;
  private gatewayUrl: string;

  /**
   * Create a new TensorZero client
   * @param options Client configuration options
   */
  constructor(options: TensorZeroGatewayOptions) {
    this.gatewayUrl = options.gatewayUrl.endsWith('/')
      ? options.gatewayUrl.slice(0, -1)
      : options.gatewayUrl;

    // Create Axios client with default configuration
    this.client = axios.create({
      baseURL: this.gatewayUrl,
      timeout: options.timeout || 60000,
      headers: {
        'Content-Type': 'application/json',
        ...(options.apiKey && { 'Authorization': `Bearer ${options.apiKey}` }),
      },
    });
  }

  /**
   * Make a non-streaming inference request
   * @param request Inference request parameters
   * @returns Promise resolving to inference response
   */
  public async inference(request: InferenceRequest): Promise<InferenceResponse> {
    try {
      const response = await this.client.post<InferenceResponse>('/inference', {
        model_name: request.modelName,
        input: request.input,
        params: request.params,
        tags: request.tags,
        functions: request.functions,
        tools: request.tools,
      });

      return response.data;
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<TensorZeroErrorResponse>;
        throw new Error(`TensorZero Gateway error: ${axiosError.response?.data?.error || axiosError.message}`);
      }
      throw error;
    }
  }

  /**
   * Make a streaming inference request
   * @param request Inference request parameters
   * @returns Promise resolving to a readable stream of inference chunks
   */
  public async inferenceStream(request: InferenceRequest): Promise<AsyncIterable<InferenceStreamChunk>> {
    try {
      const response = await this.client.post('/inference', {
        model_name: request.modelName,
        input: request.input,
        params: request.params,
        tags: request.tags,
        functions: request.functions,
        tools: request.tools,
        stream: true,
      }, {
        responseType: 'stream',
      });

      const stream = response.data as Readable;
      
      // Create an async generator that yields chunks of SSE data
      return this.parseSSEStream(stream);
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<TensorZeroErrorResponse>;
        throw new Error(`TensorZero Gateway error: ${axiosError.response?.data?.error || axiosError.message}`);
      }
      throw error;
    }
  }

  /**
   * Send feedback for a previous inference
   * @param request Feedback request parameters
   * @returns Promise resolving to feedback response
   */
  public async feedback(request: FeedbackRequest): Promise<FeedbackResponse> {
    try {
      const response = await this.client.post<FeedbackResponse>('/feedback', {
        metric_name: request.metricName,
        inference_id: request.inferenceId,
        value: request.value,
        tags: request.tags,
      });

      return response.data;
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<TensorZeroErrorResponse>;
        throw new Error(`TensorZero Gateway error: ${axiosError.response?.data?.error || axiosError.message}`);
      }
      throw error;
    }
  }

  /**
   * Parses a Server-Sent Events (SSE) stream into an async iterable of inference chunks
   * @param stream Readable stream of SSE data
   * @returns Async iterable of parsed inference chunks
   */
  private async *parseSSEStream(stream: Readable): AsyncIterable<InferenceStreamChunk> {
    let buffer = '';
    
    for await (const chunk of stream) {
      buffer += chunk.toString();
      
      // Process all complete SSE events in the buffer
      while (true) {
        const eventEnd = buffer.indexOf('\n\n');
        if (eventEnd === -1) break;
        
        const event = buffer.substring(0, eventEnd);
        buffer = buffer.substring(eventEnd + 2);
        
        // Parse the SSE event
        const lines = event.split('\n');
        const dataLines = lines
          .filter(line => line.startsWith('data: '))
          .map(line => line.substring(6));
        
        if (dataLines.length === 0) continue;
        
        try {
          for (const dataLine of dataLines) {
            // Skip empty or heartbeat messages
            if (!dataLine || dataLine === '[DONE]') continue;
            
            const data = JSON.parse(dataLine) as InferenceStreamChunk;
            yield data;
          }
        } catch (error: unknown) {
          console.error('Error parsing SSE data:', error);
          continue;
        }
      }
    }
  }
} 