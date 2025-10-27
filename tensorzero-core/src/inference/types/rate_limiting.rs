use crate::error::{Error, ErrorDetails::RateLimitMissingMaxTokens};
use crate::inference::types::{
    ContentBlock, FileWithPath, LazyFile, ModelInferenceRequest, RequestMessage, Text, Thought,
};
use crate::rate_limiting::{
    get_estimated_tokens, EstimatedRateLimitResourceUsage, RateLimitResource,
    RateLimitedInputContent, RateLimitedRequest,
};

impl RateLimitedInputContent for Text {
    fn estimated_input_token_usage(&self) -> u64 {
        let Text { text } = self;
        get_estimated_tokens(text)
    }
}

impl RateLimitedInputContent for Thought {
    fn estimated_input_token_usage(&self) -> u64 {
        let Thought {
            text,
            signature,
            // We intentionally do *not* count the summary towards the token usage
            // Even though OpenAI responses requires passing the summaries back in a multi-turn
            // conversation, we expect that the actual model will ignore them, since they're
            // not the internal model thoughts.
            summary: _,
            provider_type: _,
        } = self;
        text.as_ref().map_or(0, |text| get_estimated_tokens(text))
            + signature
                .as_ref()
                .map_or(0, |signature| get_estimated_tokens(signature))
    }
}

impl RateLimitedInputContent for ContentBlock {
    fn estimated_input_token_usage(&self) -> u64 {
        match self {
            ContentBlock::Text(text) => text.estimated_input_token_usage(),
            ContentBlock::ToolCall(tool_call) => tool_call.estimated_input_token_usage(),
            ContentBlock::ToolResult(tool_result) => tool_result.estimated_input_token_usage(),
            ContentBlock::File(file) => file.estimated_input_token_usage(),
            ContentBlock::Thought(thought) => thought.estimated_input_token_usage(),
            ContentBlock::Unknown { .. } => 0,
        }
    }
}

impl RateLimitedInputContent for RequestMessage {
    fn estimated_input_token_usage(&self) -> u64 {
        let RequestMessage {
            #[expect(unused_variables)]
            role,
            content,
        } = self;
        content
            .iter()
            .map(RateLimitedInputContent::estimated_input_token_usage)
            .sum()
    }
}

impl RateLimitedRequest for ModelInferenceRequest<'_> {
    fn estimated_resource_usage(
        &self,
        resources: &[RateLimitResource],
    ) -> Result<EstimatedRateLimitResourceUsage, Error> {
        let ModelInferenceRequest {
            inference_id: _,
            messages,
            system,
            tool_config: _, // TODO: should we account for this in advance?
            temperature: _,
            top_p: _,
            max_tokens,
            presence_penalty: _,
            frequency_penalty: _,
            seed: _,
            stop_sequences: _,
            stream: _,
            json_mode: _,
            function_type: _,
            output_schema: _,
            extra_body: _,
            fetch_and_encode_input_files_before_inference: _,
            extra_headers: _,
            extra_cache_key: _,
        } = self;

        let tokens = if resources.contains(&RateLimitResource::Token) {
            let system_tokens = system
                .as_ref()
                .map(|s| get_estimated_tokens(s))
                .unwrap_or(0);
            let messages_tokens: u64 = messages
                .iter()
                .map(RateLimitedInputContent::estimated_input_token_usage)
                .sum();
            let output_tokens =
                max_tokens.ok_or_else(|| Error::new(RateLimitMissingMaxTokens))? as u64;
            Some(system_tokens + messages_tokens + output_tokens)
        } else {
            None
        };

        let model_inferences = if resources.contains(&RateLimitResource::ModelInference) {
            Some(1)
        } else {
            None
        };

        Ok(EstimatedRateLimitResourceUsage {
            model_inferences,
            tokens,
        })
    }
}

impl RateLimitedInputContent for LazyFile {
    fn estimated_input_token_usage(&self) -> u64 {
        match self {
            LazyFile::FileWithPath(FileWithPath {
                file: _,
                storage_path: _,
            }) => {}
            LazyFile::ObjectStorage { .. } => {}
            // Forwarding a url is inherently incompatible with input token estimation,
            // so we'll need to continue using a hardcoded value here, even if we start
            // estimating tokens LazyFile::FileWithPath
            LazyFile::Url {
                file_url: _,
                future: _,
            } => {}
        }
        10_000 // Hardcoded value for file size estimation, we will improve later
    }
}
