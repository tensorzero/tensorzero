use crate::common::{mock_inference_provider_base, OptimizationTestCase};
use tensorzero_core::optimization::{
    together_sft::{
        TogetherBatchSize, TogetherLRScheduler, TogetherTrainingMethod, TogetherTrainingType,
        UninitializedTogetherSFTConfig,
    },
    UninitializedOptimizerConfig, UninitializedOptimizerInfo,
};

pub struct TogetherSFTTestCase();

impl OptimizationTestCase for TogetherSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::TogetherSFT(Box::new(
                UninitializedTogetherSFTConfig {
                    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference".to_string(),
                    credentials: None,
                    api_base: if use_mock_inference_provider {
                        Some(mock_inference_provider_base().join("together/").unwrap())
                    } else {
                        None
                    },
                    // Minimal hyperparameters for economical testing
                    n_epochs: 1,
                    n_checkpoints: 1,
                    n_evals: None,
                    batch_size: TogetherBatchSize::default(),
                    learning_rate: 0.00001,
                    warmup_ratio: 0.0,
                    max_grad_norm: 1.0,
                    weight_decay: 0.0,
                    suffix: None,
                    // Learning rate scheduler
                    lr_scheduler: TogetherLRScheduler::default(),
                    // Weights & Biases integration
                    wandb_api_key: None,
                    wandb_base_url: None,
                    wandb_project_name: None,
                    wandb_name: None,
                    // Training method
                    training_method: TogetherTrainingMethod::default(),
                    // Training type - use defaults
                    training_type: TogetherTrainingType::default(),
                    // Advanced options
                    from_checkpoint: None,
                    from_hf_model: None,
                    hf_model_revision: None,
                    hf_api_token: None,
                    hf_output_repo_name: None,
                },
            )),
        }
    }
}
