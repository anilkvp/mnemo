/// lancedb tale creation, data persisiance and retrival
use crate::types::{Chunk, DocId, Document};
use futures::TryStreamExt;
use lancedb::arrow::arrow::array::AsArray;
use lancedb::arrow::arrow::datatypes::{UInt32Type, UInt64Type};
use lancedb::arrow::arrow_array::{Array, Int64Array, UInt64Array};
use lancedb::arrow::arrow_array::{
    FixedSizeListArray, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    UInt32Array, types::Float32Type,
};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::query::QueryBase;
use lancedb::query::{ExecutableQuery, Select};
use lancedb::{Table, connect};
use std::path::Path;
use std::sync::Arc;
use tracing::debug;

/// chunk table name
const CHUNKS_TABLE: &str = "chunks";
const DOCUMENT_TABLE: &str = "documents";

const DIM: i32 = 768;
const RESULT_COLS: &[&str] = &["doc_id", "chunk_index", "title", "text", "path", "context"];

pub struct Store {
    table_chunk: Table,
    table_document: Table,
}

#[derive(Debug, Clone)]
pub enum RetrievalSource {
    Reranked,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub doc_id: DocId,
    pub chunk_index: u32,
    pub title: String,
    pub text: String,
    pub path: String,
    pub context: Option<String>,
    pub score: f32,
    pub source: RetrievalSource,
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

        let documents = match db.open_table(DOCUMENT_TABLE).execute().await {
            Ok(table) => table,
            Err(lancedb::Error::TableNotFound { .. }) => {
                db.create_empty_table(DOCUMENT_TABLE, document_schema())
                    .execute()
                    .await?
            }
            Err(e) => {
                return Err(e.into());
            }
        };
        Ok(Self {
            table_chunk: chunks,
            table_document: documents,
        })
    }

    /// Build FTS (Full Text Search) index for text colomn
    pub async fn build_fts_index(&self) -> anyhow::Result<()> {
        self.table_chunk
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
        Ok(())
    }

    /// Build FTS (Full Text Search) index for "context" column
    pub async fn build_fts_context_index(&self) -> anyhow::Result<()> {
        self.table_chunk
            .create_index(&["context"], Index::FTS(FtsIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
        Ok(())
    }
    /// Perfor FTS with give query and top-k size
    ///
    /// * `query`: query string
    /// * `k`: top-k limit
    pub async fn fts_search(&self, query: &str, k: usize) -> anyhow::Result<Vec<Hit>> {
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
    pub async fn insert_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<bool> {
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

    pub async fn upsert_document(
        &self,
        document: &Document,
        content_hash: String,
        content_version: String,
        chunk_count: usize,
    ) -> anyhow::Result<bool> {
        let schema = document_schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![document.id.0])),
                Arc::new(StringArray::from(vec![document.path.as_str()])),
                Arc::new(StringArray::from(vec![document.title.as_str()])),
                Arc::new(StringArray::from(vec![document.body.as_str()])),
                Arc::new(StringArray::from(vec![content_version.as_str()])),
                Arc::new(UInt32Array::from(vec![u32::try_from(chunk_count)?])),
                Arc::new(Int64Array::from(vec![chrono::Utc::now().timestamp()])),
                Arc::new(StringArray::from(vec![content_hash.as_str()])),
            ],
        )?;
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));

        let mut merge = self.table_document.merge_insert(&["doc_id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(reader).await?;

        Ok(true)
    }

    pub async fn document_state(&self, doc_id: DocId) -> anyhow::Result<Option<(String, String)>> {
        let batches: Vec<RecordBatch> = self
            .table_document
            .query()
            .only_if(format!("doc_id = {}", doc_id.0))
            .select(Select::columns(&["content_hash", "content_version"]))
            .limit(1)
            .execute()
            .await?
            .try_collect()
            .await?;

        Ok(batches.first().filter(|b| b.num_rows() > 0).map(|b| {
            (
                b.column(0).as_string::<i32>().value(0).to_owned(),
                b.column(1).as_string::<i32>().value(0).to_owned(),
            )
        }))
    }

    pub async fn delete_chunks_for(&self, doc_id: &DocId) -> anyhow::Result<bool> {
        debug!("Before deleting chunk_index");
        let filter = format!("doc_id = {}", doc_id.0);
        self.table_chunk.delete(&filter).await?;
        debug!("No issues in deleteing");
        //self.table_document.delete(&filter).await?;
        Ok(true)
    }

    /// Perform vector search with in build function available
    ///
    /// * `query_vector`: query string as vector
    /// * `k`: top-k limit
    pub async fn vector_search(&self, query_vector: &[f32], k: usize) -> anyhow::Result<Vec<Hit>> {
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
    ) -> anyhow::Result<Vec<Hit>> {
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

fn process_result(data: Vec<RecordBatch>, score_col: &str) -> anyhow::Result<Vec<Hit>> {
    let mut result: Vec<Hit> = Vec::new();
    for rb in data {
        let col = |name: &str| {
            rb.column_by_name(name)
                .ok_or_else(|| anyhow::anyhow!("missing column name {name}"))
        };
        let ids = col("doc_id")?.as_primitive::<UInt64Type>();
        let indexes = col("chunk_index")?.as_primitive::<UInt32Type>();
        let titles = col("title")?.as_string::<i32>();
        let texts = col("text")?.as_string::<i32>();
        let contexts = col("context")?.as_string::<i32>();
        let paths = col("path")?.as_string::<i32>();
        let scores = col(score_col)?.as_primitive::<Float32Type>();

        for i in 0..rb.num_rows() {
            result.push(Hit {
                doc_id: DocId(ids.value(i)),
                chunk_index: indexes.value(i),
                title: titles.value(i).to_owned(),
                text: texts.value(i).to_owned(),
                path: paths.value(i).to_owned(),
                context: contexts.is_valid(i).then(|| contexts.value(i).to_owned()),
                score: scores.value(i),
                source: RetrievalSource::Hybrid,
            });
        }
    }
    Ok(result)
}

fn chunk_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new("chunk_index", DataType::UInt32, false),
        Field::new("title", DataType::Utf8, false),
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

fn document_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
        Field::new("content_version", DataType::Utf8, false),
        Field::new("chunk_count", DataType::UInt32, false),
        Field::new("ingested_at", DataType::Int64, false),
        Field::new("content_hash", DataType::Utf8, false),
    ]))
}

fn to_record_batch(chunks: &[Chunk], schema: Arc<Schema>) -> anyhow::Result<RecordBatch> {
    let ids = UInt64Array::from_iter_values(chunks.iter().map(|c| c.id.0));
    let chunk_indexs = UInt32Array::from_iter_values(
        chunks
            .iter()
            .map(|c| u32::try_from(c.index))
            .collect::<Result<Vec<_>, _>>()?,
    );

    let titles = StringArray::from_iter_values(chunks.iter().map(|c| c.title.as_str()));
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
            Arc::new(titles),
            Arc::new(texts),
            Arc::new(paths),
            Arc::new(contexts),
            Arc::new(vectors),
        ],
    )?;

    Ok(batch)
}
