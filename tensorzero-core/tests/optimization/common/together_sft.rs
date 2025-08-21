use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    together_sft::UninitializedTogetherSFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
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
                        Some("http://localhost:3030/together/".parse().unwrap())
                    } else {
                        None
                    },
                    // Minimal hyperparameters for economical testing
                    n_epochs: Some(1),
                    n_checkpoints: None,
                    n_evals: None,
                    batch_size: None,
                    learning_rate: None,
                    warmup_ratio: None,
                    max_grad_norm: None,
                    weight_decay: None,
                    suffix: None,
                    // Learning rate scheduler
                    lr_scheduler_type: None,
                    lr_scheduler_min_lr_ratio: None,
                    // Weights & Biases integration
                    wandb_api_key: None,
                    wandb_base_url: None,
                    wandb_project_name: None,
                    wandb_name: None,
                    // Training method
                    train_on_inputs: None,
                    // Training type - use defaults
                    training_type: None,
                    lora_r: None,
                    lora_alpha: None,
                    lora_dropout: None,
                    lora_trainable_modules: None,
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
