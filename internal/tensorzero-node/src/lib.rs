#![recursion_limit = "256"]
#![deny(clippy::all)]

mod config_writer;
mod postgres;

#[macro_use]
extern crate napi_derive;
