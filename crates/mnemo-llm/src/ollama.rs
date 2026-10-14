use futures::Stream;

use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;
use tokio_stream::wrappers::LinesStream;
use tokio_util::io::StreamReader;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub content: String,
    pub role: String,
}

impl Message {
    pub fn system(s: impl Into<String>) -> Self {
        Self {
            content: s.into(),
            role: "system".into(),
        }
    }

    pub fn user(s: impl Into<String>) -> Self {
        Self {
            content: s.into(),
            role: "user".into(),
        }
    }
}

#[derive(Serialize)]
struct Options {
    temperature: f32,
    num_ctx: u32,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    messages: &'a [Message],
    model: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<&'a str>,
    options: Options,
    think: bool,
}

#[derive(Deserialize, Debug)]
struct ChatChunk {
    message: Option<Message>,
    done: bool,
}

#[derive(Clone)]
pub struct Ollama {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl Ollama {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }

    pub async fn check_ready(&self) -> anyhow::Result<()> {
        #[derive(Deserialize)]
        struct Tags {
            models: Vec<Tag>,
        }
        #[derive(Deserialize)]
        struct Tag {
            name: String,
        }

        let tags: Tags = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Ollama is not reachable at {}; is `ollama serve` is running",
                    self.base_url
                )
            })?
            .json()
            .await?;

        anyhow::ensure!(
            tags.models.iter().any(|m| m.name == self.model),
            "model '{}' not found; run `ollama pull {}",
            self.model,
            self.model
        );

        Ok(())
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        temperature: f32,
        num_ctx: u32,
    ) -> anyhow::Result<String> {
        self.request(messages, false, None, temperature, num_ctx)
            .await
    }

    pub async fn chat_json(
        &self,
        messages: &[Message],
        temperature: f32,
        num_ctx: u32,
    ) -> anyhow::Result<String> {
        self.request(messages, false, Some("json"), temperature, num_ctx)
            .await
    }

    pub async fn chat_stream(
        &self,
        messages: &[Message],
        temperature: f32,
        num_ctx: u32,
    ) -> anyhow::Result<impl Stream<Item = anyhow::Result<String>>> {
        let body = ChatRequest {
            messages,
            model: &self.model,
            stream: true,
            format: None,
            think: false,
            options: Options {
                temperature,
                num_ctx,
            },
        };

        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let body = resp.bytes_stream().map_err(std::io::Error::other);
        let lines = LinesStream::new(StreamReader::new(body).lines());

        Ok(lines
            .map_err(anyhow::Error::from)
            .try_filter_map(|line| async move { parse_line(&line) }))
    }

    async fn request(
        &self,
        messages: &[Message],
        stream: bool,
        format: Option<&str>,
        temperature: f32,
        num_ctx: u32,
    ) -> anyhow::Result<String> {
        let body = ChatRequest {
            messages,
            model: &self.model,
            stream,
            format,
            think: true,
            options: Options {
                temperature,
                num_ctx,
            },
        };

        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let chunk: ChatChunk = resp.json().await?;
        Ok(chunk.message.map(|m| m.content).unwrap_or_default())
    }
}

fn parse_line(line: &str) -> anyhow::Result<Option<String>> {
    if line.trim().is_empty() {
        return Ok(None);
    }
    let chunk: ChatChunk = serde_json::from_str(line)?;
    Ok(chunk.message.map(|m| m.content).filter(|s| !s.is_empty()))
}
