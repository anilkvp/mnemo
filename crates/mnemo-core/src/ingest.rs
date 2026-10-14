//! Ingetion related functions

use crate::{
    embed::Embedder,
    types::{Chunk, DocId},
};
use std::{
    fs,
    hash::{Hash, Hasher},
    path::PathBuf,
};
use text_splitter::{ChunkConfig, TextSplitter};

pub fn load_and_chunk(path: &PathBuf) -> anyhow::Result<Vec<Chunk>> {
    let docid = doc_id_from_path(path);
    // let text_splitter = TextSplitter::new(CHUNK_MAX_CHARS);
    let text_splitter = TextSplitter::new(ChunkConfig::new(1000).with_overlap(50)?);

    let content = fs::read_to_string(path.clone())?;

    let texts: Vec<String> = text_splitter
        .chunks(&content)
        .into_iter()
        .map(|k| k.to_string())
        .collect();

    let embeddings = Embedder::new()?.embed(&texts)?;

    let chunks: Vec<Chunk> = texts
        .iter()
        .zip(embeddings)
        .enumerate()
        .map(|(i, (text, embedding))| Chunk {
            id: docid,
            index: i,
            text: text.to_string(),
            path: path.to_string_lossy().to_string(),
            context: None,
            embedding: Some(embedding),
        })
        .collect();

    Ok(chunks)
}

fn doc_id_from_path(path: &PathBuf) -> DocId {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    DocId(h.finish())
}
