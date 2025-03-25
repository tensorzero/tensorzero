/**
 * TensorZero JavaScript Client Types
 */

export interface TensorZeroGatewayOptions {
  /**
   * URL of the TensorZero Gateway API
   */
  gatewayUrl: string;
  
  /**
   * Optional API key for authentication
   */
  apiKey?: string;
  
  /**
   * Request timeout in milliseconds (default: 60000)
   */
  timeout?: number;
}

export interface Message {
  /**
   * Role of the message sender (e.g., "user", "assistant", "system")
   */
  role: string;
  
  /**
   * Content of the message
   */
  content: string | null;
  
  /**
   * Optional name of the message sender
   */
  name?: string;
}

export interface InferenceRequest {
  /**
   * Name of the model to use for inference
   * Format: "provider::model_name" (e.g., "openai::gpt-4o-mini")
   */
  modelName: string;
  
  /**
   * Input data for the model
   */
  input: {
    /**
     * List of messages for chat-based models
     */
    messages?: Message[];
    
    /**
     * Prompt for completion-based models
     */
    prompt?: string;
    
    /**
     * Model-specific parameters
     */
    [key: string]: any;
  };
  
  /**
   * Optional parameters for inference
   */
  params?: {
    /**
     * Temperature for controlling randomness
     */
    temperature?: number;
    
    /**
     * Top-p value for nucleus sampling
     */
    top_p?: number;
    
    /**
     * Maximum number of tokens to generate
     */
    max_tokens?: number;
    
    /**
     * Stop sequences to end generation
     */
    stop?: string | string[];
    
    /**
     * Model-specific parameters
     */
    [key: string]: any;
  };
  
  /**
   * Optional tags for categorizing and filtering inferences
   */
  tags?: Record<string, string>;
  
  /**
   * Optional function calls configuration
   */
  functions?: any[];
  
  /**
   * Optional tool calls configuration
   */
  tools?: any[];
}

export interface InferenceResponse {
  /**
   * Unique ID of the inference
   */
  inference_id: string;
  
  /**
   * Model used for the inference
   */
  model: string;
  
  /**
   * Generated content
   */
  output: {
    /**
     * Text output from the model
     */
    content?: string;
    
    /**
     * Generated message for chat-based models
     */
    message?: Message;
    
    /**
     * Function call response if applicable
     */
    function_call?: any;
    
    /**
     * Tool call response if applicable
     */
    tool_calls?: any[];
  };
  
  /**
   * Input provided to the model
   */
  input: {
    /**
     * List of messages for chat-based models
     */
    messages?: Message[];
    
    /**
     * Prompt for completion-based models
     */
    prompt?: string;
    
    /**
     * Other input parameters
     */
    [key: string]: any;
  };
  
  /**
   * Usage statistics
   */
  usage: {
    /**
     * Number of tokens in the prompt
     */
    prompt_tokens: number;
    
    /**
     * Number of tokens in the completion
     */
    completion_tokens: number;
    
    /**
     * Total number of tokens used
     */
    total_tokens: number;
  };
  
  /**
   * Timestamp of the inference
   */
  created_at: string;
}

export interface InferenceStreamChunk {
  /**
   * Unique ID of the inference
   */
  inference_id: string;
  
  /**
   * Chunk of content from the model
   */
  chunk: {
    /**
     * Text content in this chunk
     */
    content?: string;
    
    /**
     * Message in this chunk for chat-based models
     */
    message?: Partial<Message>;
    
    /**
     * Function call chunk if applicable
     */
    function_call?: any;
    
    /**
     * Tool call chunk if applicable
     */
    tool_calls?: any[];
  };
  
  /**
   * Whether this is the final chunk
   */
  is_final: boolean;
  
  /**
   * Usage statistics (only included in the final chunk)
   */
  usage?: {
    /**
     * Number of tokens in the prompt
     */
    prompt_tokens: number;
    
    /**
     * Number of tokens in the completion
     */
    completion_tokens: number;
    
    /**
     * Total number of tokens used
     */
    total_tokens: number;
  };
}

export interface FeedbackRequest {
  /**
   * Name of the metric to record
   */
  metricName: string;
  
  /**
   * ID of the inference to provide feedback for
   */
  inferenceId: string;
  
  /**
   * Feedback value (can be boolean, number, string, or object)
   */
  value: boolean | number | string | object;
  
  /**
   * Optional tags for categorizing and filtering feedback
   */
  tags?: Record<string, string>;
}

export interface FeedbackResponse {
  /**
   * Unique ID of the feedback
   */
  feedback_id: string;
  
  /**
   * ID of the inference the feedback is for
   */
  inference_id: string;
  
  /**
   * Name of the metric recorded
   */
  metric_name: string;
  
  /**
   * Feedback value
   */
  value: any;
  
  /**
   * Timestamp of the feedback
   */
  created_at: string;
} 