use crate::error::Error;

mod static_weights;

pub trait VariantSampler {
    async fn setup(&self) -> Result<(), Error>;
    async fn sample(&self) -> Result<String, Error>;
}
