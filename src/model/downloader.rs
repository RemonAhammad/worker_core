//! GGUF model downloader.
//!
//! Resolves a Hugging Face repo + filename, streams the file to local disk
//! with an `indicatif` progress bar, supports resuming interrupted downloads
//! via HTTP `Range` requests, and optionally verifies a SHA256 digest.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{CONTENT_LENGTH, RANGE};
use sha2::{Digest, Sha256};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::error::AppError;

/// A resolved on-disk model file ready for loading.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// Ensure the GGUF file is present locally, downloading and resuming as needed.
///
/// `expected_sha256` may be empty to skip verification. When set, the local
/// file is verified after every download (and on every startup if it already
/// exists locally).
pub async fn ensure_model_present(
    models_dir: &Path,
    repo: &str,
    filename: &str,
    expected_sha256: &str,
) -> Result<ResolvedModel, AppError> {
    tokio::fs::create_dir_all(models_dir).await?;
    let target = models_dir.join(filename);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");

    let client = reqwest::Client::builder()
        .user_agent("co_worker_lite/0.1")
        .build()?;

    let remote_size = head_content_length(&client, &url).await?;
    let local_size = match tokio::fs::metadata(&target).await {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => return Err(e.into()),
    };

    if local_size != remote_size {
        download_with_resume(&client, &url, &target, local_size, remote_size).await?;
    } else {
        tracing::info!(path = %target.display(), bytes = local_size, "model already present");
    }

    if !expected_sha256.is_empty() {
        verify_sha256(&target, expected_sha256).await?;
    }

    let size_bytes = tokio::fs::metadata(&target).await?.len();
    Ok(ResolvedModel {
        path: target,
        size_bytes,
    })
}

async fn head_content_length(client: &reqwest::Client, url: &str) -> Result<u64, AppError> {
    let resp = client.head(url).send().await?.error_for_status()?;
    let len = resp
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| AppError::Internal("missing Content-Length header".into()))?;
    Ok(len)
}

async fn download_with_resume(
    client: &reqwest::Client,
    url: &str,
    target: &Path,
    local_size: u64,
    remote_size: u64,
) -> Result<(), AppError> {
    let mut request = client.get(url);
    if local_size > 0 && local_size < remote_size {
        tracing::info!(
            path = %target.display(),
            already = local_size,
            total = remote_size,
            "resuming download"
        );
        request = request.header(RANGE, format!("bytes={local_size}-"));
    } else if local_size > remote_size {
        tracing::warn!(
            path = %target.display(),
            local = local_size,
            remote = remote_size,
            "local file larger than remote; restarting download"
        );
        tokio::fs::remove_file(target).await?;
    } else {
        tracing::info!(path = %target.display(), bytes = remote_size, "starting download");
    }

    let resp = request.send().await?.error_for_status()?;
    let resuming = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let starting_at = if resuming { local_size } else { 0 };

    let pb = ProgressBar::new(remote_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .map_err(|e| AppError::Internal(e.to_string()))?
        .progress_chars("#>-"),
    );
    pb.set_position(starting_at);

    let mut file = OpenOptions::new()
        .create(true)
        .append(resuming)
        .write(true)
        .truncate(!resuming)
        .open(target)
        .await?;

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }
    file.flush().await?;
    pb.finish_with_message("download complete");
    Ok(())
}

async fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), AppError> {
    tracing::info!(path = %path.display(), "verifying sha256");
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex::encode(hasher.finalize());
    if !actual.eq_ignore_ascii_case(expected_hex) {
        return Err(AppError::ChecksumMismatch {
            expected: expected_hex.to_string(),
            actual,
        });
    }
    Ok(())
}

/// List GGUF files in `models_dir`. Used by the `/v1/models` endpoint.
pub async fn list_local_models(models_dir: &Path) -> Result<Vec<(String, u64)>, AppError> {
    let mut out = Vec::new();
    let mut entries = match tokio::fs::read_dir(models_dir).await {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            let meta = entry.metadata().await?;
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            out.push((name, meta.len()));
        }
    }
    out.sort();
    Ok(out)
}
