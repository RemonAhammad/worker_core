//! co_worker_lite — library surface used by both the binary and integration
//! tests. The binary (`main.rs`) just wires these modules together and starts
//! the axum server.

pub mod api;
pub mod config;
pub mod context;
pub mod db;
pub mod error;
pub mod memory;
pub mod model;
pub mod state;
pub mod tools;
pub mod types;
