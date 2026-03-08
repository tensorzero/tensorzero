use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    together_sft::{
        TogetherBatchSize, TogetherLRScheduler, TogetherTrainingMethod, TogetherTrainingType,
        UninitializedTogetherSFTConfig,
    },
};

pub struct TogetherSFTTestCase();

impl OptimizationTestCase for TogetherSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false
    }

    fn get_optimizer_info(&self) -> UninitializedOptimizerInfo {
        // Note: mock mode is configured via provider_types.together.sft in the test config file
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::TogetherSFT(Box::new(
                UninitializedTogetherSFTConfig {
                    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference".to_string(),
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
                    // Per-job wandb name (provider-level wandb settings come from config)
                    wandb_name: None,
                    // Training method
                    training_method: TogetherTrainingMethod::default(),
                    // Training type - use defaults
                    training_type: TogetherTrainingType::default(),
                    // Advanced options
                    from_checkpoint: None,
                    from_hf_model: None,
                    hf_model_revision: None,
                    hf_output_repo_name: None,
                },
            )),
        }
    }
}
