use std::{ffi::OsStr, path::PathBuf};

pub fn get_files(path: &PathBuf) -> anyhow::Result<Vec<PathBuf>> {
    let mut result: Vec<PathBuf> = Vec::new();
    for file in std::fs::read_dir(path)? {
        let file = file?;
        let file_path = file.path();
        if file_path.extension() == Some(OsStr::new("md")) {
            result.push(file_path);
        }
    }
    Ok(result)
}
