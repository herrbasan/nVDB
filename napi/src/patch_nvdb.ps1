
$content = Get-Content lib.rs -Raw

$content = $content -replace "pub struct Database \{(\r?\n)\s+inner: Arc<RustDatabase>,(\r?\n)\}", "pub struct Database {`n    inner: std::sync::RwLock<Option<Arc<RustDatabase>>>,`n}`n`nimpl Database {`n    fn inner(&self) -> Result<Arc<RustDatabase>> {`n        self.inner.read().unwrap().clone().ok_or_else(|| Error::from_reason(`"Database closed`"))`n    }`n}"
$content = $content -replace "inner: inner", "inner: std::sync::RwLock::new(Some(inner))"
$content = $content -replace "self\.inner\.", "self.inner()?."

$content = $content -replace "pub struct Collection \{(\r?\n)\s+inner: RustCollection,(\r?\n)\s+// Keep reference to database to prevent premature drop(\r?\n)\s+_db: Arc<RustDatabase>,(\r?\n)\}", "pub struct Collection {`n    inner: std::sync::RwLock<Option<Arc<RustCollection>>>,`n    _db: std::sync::RwLock<Option<Arc<RustDatabase>>>,`n}`n`nimpl Collection {`n    fn inner(&self) -> Result<Arc<RustCollection>> {`n        self.inner.read().unwrap().clone().ok_or_else(|| Error::from_reason(`"Collection closed`"))`n    }`n}"
$content = $content -replace "inner: coll,\r?\n\s+_db: self\.inner\(\)\?\.clone\(\),", "inner: std::sync::RwLock::new(Some(Arc::new(coll))),`n            _db: std::sync::RwLock::new(Some(self.inner()?.clone())),"

Set-Content lib.rs $content

