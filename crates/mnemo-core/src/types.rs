use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DocId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: DocId,
    pub index: usize,
    pub text: String,
    pub path: String,
    pub context: Option<String>,
    pub embedding: Option<Vec<f32>>,
}
