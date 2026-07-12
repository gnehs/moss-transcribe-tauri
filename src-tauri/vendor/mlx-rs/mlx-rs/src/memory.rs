//! MLX allocator memory management and statistics.

use crate::{error::Result, utils::guard::Guarded};

/// Process-wide MLX allocator memory statistics, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Memory that is still actively referenced by MLX arrays.
    pub active: usize,
    /// Unused memory retained by the allocator for reuse.
    pub cache: usize,
    /// Peak active memory since process start or the last peak reset.
    pub peak: usize,
}

/// Return the current process-wide MLX allocator statistics.
pub fn stats() -> Result<MemoryStats> {
    Ok(MemoryStats {
        active: <usize as Guarded>::try_from_op(|result| unsafe {
            mlx_sys::mlx_get_active_memory(result)
        })?,
        cache: <usize as Guarded>::try_from_op(|result| unsafe {
            mlx_sys::mlx_get_cache_memory(result)
        })?,
        peak: <usize as Guarded>::try_from_op(|result| unsafe {
            mlx_sys::mlx_get_peak_memory(result)
        })?,
    })
}

/// Reset the process-wide peak active-memory counter.
pub fn reset_peak() -> Result<()> {
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_reset_peak_memory() })
}

/// Set the process-wide free allocator cache limit in bytes.
///
/// Returns the previous cache limit.
pub fn set_cache_limit(limit: usize) -> Result<usize> {
    <usize as Guarded>::try_from_op(|result| unsafe {
        mlx_sys::mlx_set_cache_limit(result, limit)
    })
}

/// Return all unused allocator cache memory to the system allocator.
pub fn clear_cache() -> Result<()> {
    <() as Guarded>::try_from_op(|_| unsafe { mlx_sys::mlx_clear_cache() })
}
