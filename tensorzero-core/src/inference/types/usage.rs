use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct Usage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

impl Usage {
    pub fn total_tokens(&self) -> Option<u32> {
        match (self.input_tokens, self.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        }
    }
}
