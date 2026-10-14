/// lancedb tale creation, data persisiance and retrival
use crate::types::Chunk;
use futures::TryStreamExt;
use lancedb::arrow::arrow::array::AsArray;
use lancedb::arrow::arrow::datatypes::UInt32Type;
use lancedb::arrow::arrow_array::Array;
use lancedb::arrow::arrow_array::{
    FixedSizeListArray, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    UInt32Array, types::Float32Type,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::query::QueryBase;
use lancedb::query::{ExecutableQuery, Select};
use lancedb::{Connection, Table, connect};
use std::path::Path;
use std::sync::Arc;

/// chunk table name
const CHUNKS_TABLE: &str = "chunks";
const DIM: i32 = 384;
const RESULT_COLS: &[&str] = &["id", "chunk_index", "text", "path", "context"];

pub struct Store {
    db: Connection,
    table_chunk: Table,
}

#[derive(Debug, Clone)]
pub struct Hits {
    pub doc_id: String,
    pub chunk_index: u32,
    pub text: String,
    pub path: String,
    pub context: Option<String>,
    pub score: f32,
}

impl Store {
    /// Open database connection.
    ///
    /// * `path`: database store path
    pub async fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let uri = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("store path is not valid UTF-8 {}", path.display()))?;

        let db = connect(uri).execute().await?;

        let chunks = match db.open_table(CHUNKS_TABLE).execute().await {
            Ok(table) => table,
            Err(lancedb::Error::TableNotFound { .. }) => {
                db.create_empty_table(CHUNKS_TABLE, chunk_schema())
                    .execute()
                    .await?
            }
            Err(e) => {
                return Err(e.into());
            }
        };
        Ok(Self {
            db,
            table_chunk: chunks,
        })
    }

    /// Build FTS (Full Text Search) index
    pub async fn build_fts_index(&self) -> anyhow::Result<()> {
        self.table_chunk
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
        Ok(())
    }

    /// Perfor FTS with give query and top-k size
    ///
    /// * `query`: query string
    /// * `k`: top-k limit
    pub async fn fts_search(&self, query: &str, k: usize) -> anyhow::Result<Vec<Hits>> {
        let result = self
            .table_chunk
            .query()
            .full_text_search(FullTextSearchQuery::new(query.to_owned()))
            .select(Select::columns(RESULT_COLS))
            .limit(k)
            .execute()
            .await?
            .try_collect()
            .await?;
        process_result(result, "_score")
    }

    /// Insert chunks
    ///
    /// * `chunks`: slice of chunks
    pub async fn upsert(&self, chunks: &[Chunk]) -> anyhow::Result<bool> {
        if chunks.is_empty() {
            return Ok(false);
        }
        let schema = chunk_schema();
        let batches = chunks
            .chunks(1_000)
            .map(|part| to_record_batch(part, schema.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            schema,
        ));
        self.table_chunk.add(reader).execute().await?;
        Ok(true)
    }

    /// Perform vector search with in build function available
    ///
    /// * `query_vector`: query string as vector
    /// * `k`: top-k limit
    pub async fn vector_search(&self, query_vector: &[f32], k: usize) -> anyhow::Result<Vec<Hits>> {
        let result: Vec<RecordBatch> = self
            .table_chunk
            .vector_search(query_vector)?
            .distance_type(lancedb::DistanceType::Cosine)
            .select(Select::columns(RESULT_COLS))
            .limit(k)
            .execute()
            .await?
            .try_collect()
            .await?;
        let result_data = process_result(result, "_distance")?;
        Ok(result_data)
    }

    /// Perform vector search.
    ///
    /// * `query_vector`: query string vector
    /// * `k`: top-k limit
    pub async fn vector_search_v1(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> anyhow::Result<Vec<Hits>> {
        let result = self
            .table_chunk
            .query()
            .nearest_to(query_vector)?
            .distance_type(lancedb::DistanceType::Cosine)
            .select(Select::columns(RESULT_COLS))
            .limit(k)
            .execute()
            .await?
            .try_collect()
            .await?;

        let result_data = process_result(result, "_distance")?;
        Ok(result_data)
    }
}

fn process_result(data: Vec<RecordBatch>, score_col: &str) -> anyhow::Result<Vec<Hits>> {
    let mut result: Vec<Hits> = Vec::new();
    for rb in data {
        let col = |name: &str| {
            rb.column_by_name(name)
                .ok_or_else(|| anyhow::anyhow!("missing column name {name}"))
        };
        let ids = col("id")?.as_string::<i32>();
        let indexes = col("chunk_index")?.as_primitive::<UInt32Type>();
        let texts = col("text")?.as_string::<i32>();
        let contexts = col("context")?.as_string::<i32>();
        let paths = col("path")?.as_string::<i32>();
        let scores = col(score_col)?.as_primitive::<Float32Type>();

        for i in 0..rb.num_rows() {
            result.push(Hits {
                doc_id: ids.value(i).to_owned(),
                chunk_index: indexes.value(i),
                text: texts.value(i).to_owned(),
                path: paths.value(i).to_owned(),
                context: contexts.is_valid(i).then(|| contexts.value(i).to_owned()),
                score: scores.value(i),
            });
        }
    }
    Ok(result)
}

fn chunk_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("context", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM),
            false,
        ),
    ]))
}

fn to_record_batch(chunks: &[Chunk], schema: Arc<Schema>) -> anyhow::Result<RecordBatch> {
    let ids = StringArray::from_iter_values(chunks.iter().map(|c| c.id.0.to_string()));
    let chunk_indexs = UInt32Array::from_iter_values(
        chunks
            .iter()
            .map(|c| u32::try_from(c.index))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let texts = StringArray::from_iter_values(chunks.iter().map(|c| c.text.as_str()));
    let contexts: StringArray = chunks.iter().map(|c| c.context.as_deref()).collect();
    let paths: StringArray = StringArray::from_iter_values(chunks.iter().map(|c| c.path.as_str()));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        chunks.iter().map(|c| {
            c.embedding
                .as_ref()
                .map(|e| e.iter().copied().map(Some).collect::<Vec<_>>())
        }),
        DIM,
    );

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(chunk_indexs),
            Arc::new(texts),
            Arc::new(paths),
            Arc::new(contexts),
            Arc::new(vectors),
        ],
    )?;

    Ok(batch)
}
