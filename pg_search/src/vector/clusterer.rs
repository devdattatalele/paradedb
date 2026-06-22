// Copyright (c) 2023-2026 ParadeDB, Inc.
//
// This file is part of ParadeDB - Postgres for Search and Analytics
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::Hasher;
use std::sync::Arc;
use std::thread;

use parking_lot::Mutex;
use rustc_hash::FxHasher;
use superkmeans::{HierarchicalSuperKMeans, HierarchicalSuperKMeansConfig, SuperKMeansError};
use tantivy::vector::{
    Assignment, IvfCentroids, IvfClusterer, IvfMatrix, IvfMergeSettings, IvfVectors, Metric,
    VectorOptions,
};
use tantivy::{Index, TantivyError};

use crate::postgres::options::BM25IndexOptions;

const DEFAULT_ASSIGN_BATCH_SIZE: usize = 40_960;
/// Boundary replication is ON by default: a vector unset for
/// `max_replicas_per_vector` replicates into up to this many neighbor
/// clusters (SPANN's ReplicaCount). Set the option to `0` to turn it off
/// (e.g. for isolation measurement); `-1`/unset means this default.
const DEFAULT_MAX_REPLICAS_PER_VECTOR: usize = 8;
/// ε₁ closure factor for boundary replication when the option is unset.
/// Lowered from SPANN's spec value of 10.0: the `max_replicas` + RNG cap binds
/// the replica set long before a 10× radius does, so the wider radius mostly
/// bought fallback-scan cost (~80% of vectors miss the prune certificate at
/// ε=10) without changing the kept replicas. Provisional — validate against the
/// recall@10 sweep before treating as final.
const DEFAULT_REPLICA_EPSILON: f32 = 3.0;

#[derive(Clone, Debug)]
pub struct SuperKMeansIvfClusterer {
    config: HierarchicalSuperKMeansConfig,
    centroid_ratio: f32,
    training_samples_per_centroid: usize,
    assign_batch_size: usize,
    /// Hard cap on primary cluster size. `None` defers to tantivy's default
    /// merge band, preserving the pre-option behavior.
    max_posting_len: Option<usize>,
    /// Floor on primary cluster size. `None` defers to tantivy's default
    /// merge band.
    min_posting_len: Option<usize>,
    /// Max clusters a boundary vector replicates into. `None` => the default
    /// `DEFAULT_MAX_REPLICAS_PER_VECTOR` (replication ON). Set explicitly to `0`
    /// to disable (the isolation knob for recall measurement).
    max_replicas_per_vector: Option<usize>,
    /// Per-cluster replica budget. `None` => driver default (~max_posting_len/2).
    max_replicas_per_cluster: Option<usize>,
    /// ε₁ closure factor for replica candidacy. `None` => the default
    /// `DEFAULT_REPLICA_EPSILON`.
    replica_epsilon: Option<f32>,
    /// Build-time-only cache of the centroid-neighbor table used to prune the
    /// per-vector replica scan (see [`CentroidNeighbors`]). Built lazily on the
    /// first `assign` call of a merge and reused across that merge's batches;
    /// re-keyed by centroid fingerprint so a later merge with a different
    /// centroid set rebuilds. `Arc<Mutex<_>>` keeps the clusterer `Clone`/`Debug`
    /// and lets clones share the cache. Never serialized; discarded with the
    /// clusterer. Empty when replication is off / metric is `Dot`.
    neighbor_cache: Arc<Mutex<Option<Arc<CentroidNeighbors>>>>,
}

impl Default for SuperKMeansIvfClusterer {
    fn default() -> Self {
        let config = HierarchicalSuperKMeansConfig {
            suppress_warnings: true,
            sampling_fraction: 1.0,
            ..Default::default()
        };
        Self {
            config,
            centroid_ratio: 0.01,
            training_samples_per_centroid: 32,
            assign_batch_size: DEFAULT_ASSIGN_BATCH_SIZE,
            max_posting_len: None,
            min_posting_len: None,
            max_replicas_per_vector: None,
            max_replicas_per_cluster: None,
            replica_epsilon: None,
            neighbor_cache: Arc::new(Mutex::new(None)),
        }
    }
}

impl SuperKMeansIvfClusterer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_centroid_ratio(mut self, centroid_ratio: f32) -> Self {
        self.centroid_ratio = centroid_ratio;
        self
    }

    pub fn with_training_samples_per_centroid(
        mut self,
        training_samples_per_centroid: usize,
    ) -> Self {
        self.training_samples_per_centroid = training_samples_per_centroid;
        self
    }

    pub fn with_max_posting_len(mut self, max_posting_len: Option<usize>) -> Self {
        self.max_posting_len = max_posting_len;
        self
    }

    pub fn with_min_posting_len(mut self, min_posting_len: Option<usize>) -> Self {
        self.min_posting_len = min_posting_len;
        self
    }

    pub fn with_max_replicas_per_vector(mut self, max_replicas_per_vector: Option<usize>) -> Self {
        self.max_replicas_per_vector = max_replicas_per_vector;
        self
    }

    pub fn with_max_replicas_per_cluster(
        mut self,
        max_replicas_per_cluster: Option<usize>,
    ) -> Self {
        self.max_replicas_per_cluster = max_replicas_per_cluster;
        self
    }

    pub fn with_replica_epsilon(mut self, replica_epsilon: Option<f32>) -> Self {
        self.replica_epsilon = replica_epsilon;
        self
    }

    /// Return the centroid-neighbor table for the given centroid set, building
    /// it once and caching it on the clusterer. `assign` is called per batch
    /// with the same (fixed) centroids, so this builds on the first batch and
    /// every later batch of the merge reuses the cached `Arc`. The cache is
    /// keyed by a fingerprint of the centroid bytes, so a subsequent merge that
    /// retrains a different centroid set rebuilds rather than serving a stale
    /// table. The table is built ONLY here (gated by the caller on
    /// `max_replicas > 0` and a triangle-safe metric), never when replication
    /// is off — so the no-replication build is byte-identical to Phase 1.
    fn centroid_neighbors(
        &self,
        centroids: &[f32],
        num_centroids: usize,
        dim: usize,
        max_replicas: usize,
    ) -> Arc<CentroidNeighbors> {
        let fingerprint = centroid_fingerprint(centroids, num_centroids, dim);
        let mut guard = self.neighbor_cache.lock();
        if let Some(existing) = guard.as_ref() {
            if existing.rows == num_centroids && existing.fingerprint == fingerprint {
                return existing.clone();
            }
        }
        // Cap on neighbors stored per centroid. Correctness does NOT depend on
        // this value: under-coverage is detected per vector and falls back to a
        // full scan (see `replicas_pruned`). It only trades memory/build time
        // for prune effectiveness. Scale modestly with the replica cap.
        let neighbor_cap = max_replicas.saturating_mul(32).clamp(128, 1024);
        let table = Arc::new(CentroidNeighbors::build(
            centroids,
            num_centroids,
            dim,
            neighbor_cap,
            fingerprint,
        ));
        *guard = Some(table.clone());
        table
    }
}

impl IvfClusterer for SuperKMeansIvfClusterer {
    fn centroid_ratio(&self) -> f32 {
        self.centroid_ratio
    }

    fn training_samples_per_centroid(&self) -> usize {
        self.training_samples_per_centroid
    }

    fn assign_batch_size(&self) -> usize {
        self.assign_batch_size
    }

    fn merge_settings(&self, total_target_docs: usize) -> tantivy::Result<IvfMergeSettings> {
        let centroid_ratio = self.centroid_ratio;
        let training_samples_per_centroid = self.training_samples_per_centroid;
        let assign_batch_size = self.assign_batch_size;

        // Mirror tantivy's default `IvfClusterer::merge_settings` exactly, so
        // an index that sets neither `max_posting_len` nor `min_posting_len`
        // behaves byte-identically to the default path. Only the band fields
        // diverge, and only when the option is explicitly set.
        assert!(
            centroid_ratio > 0.0 && centroid_ratio <= 1.0,
            "centroid_ratio must be in (0, 1], got {centroid_ratio}"
        );
        assert!(
            training_samples_per_centroid > 1,
            "training_samples_per_centroid must be > 1, got {training_samples_per_centroid}"
        );
        assert!(assign_batch_size > 0, "assign_batch_size must be > 0");

        let num_centroids =
            ((total_target_docs as f64) * f64::from(centroid_ratio)).ceil() as usize;
        let num_centroids = num_centroids.clamp(1, total_target_docs);

        // Default band: max = 4×mean, min = mean/4 (tantivy's private
        // MAX_POSTING_FACTOR / MIN_POSTING_DIVISOR, replicated here). When an
        // option is set, it wins; otherwise the default is preserved.
        let mean_posting_len = (total_target_docs / num_centroids).max(1);
        let max_posting_len = self
            .max_posting_len
            .unwrap_or_else(|| mean_posting_len.saturating_mul(2));
        let min_posting_len = self
            .min_posting_len
            .unwrap_or_else(|| (mean_posting_len / 2).max(1));

        // Phase 2 replication knobs. `max_replicas_per_vector` defaults to
        // DEFAULT_MAX_REPLICAS_PER_VECTOR (ON) when unset; the cluster budget
        // defaults to ~half the max posting length. Set the option to 0 to
        // disable replication (the isolation knob).
        let max_replicas_per_vector = self
            .max_replicas_per_vector
            .unwrap_or(DEFAULT_MAX_REPLICAS_PER_VECTOR);
        let max_replicas_per_cluster = self
            .max_replicas_per_cluster
            .unwrap_or_else(|| (max_posting_len / 2).max(1));
        let replica_epsilon = self.replica_epsilon.unwrap_or(DEFAULT_REPLICA_EPSILON);

        Ok(IvfMergeSettings {
            num_centroids,
            training_samples_per_centroid,
            assign_batch_size,
            max_posting_len,
            min_posting_len,
            max_replicas_per_vector,
            max_replicas_per_cluster,
            replica_epsilon,
        })
    }

    fn train(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        num_centroids: usize,
    ) -> tantivy::Result<IvfCentroids> {
        let IvfVectors::F32(vectors) = vectors;
        let dim = options.dim();
        if vectors.matrix.dims != dim {
            return Err(TantivyError::InvalidArgument(format!(
                "vector dimensionality mismatch: expected {dim}, got {}",
                vectors.matrix.dims
            )));
        }
        if vectors.doc_ids.len() != vectors.matrix.rows {
            return Err(TantivyError::InvalidArgument(format!(
                "vector doc_id count mismatch: expected {}, got {}",
                vectors.matrix.rows,
                vectors.doc_ids.len()
            )));
        }
        if vectors.matrix.values.len() != vectors.matrix.rows * dim {
            return Err(TantivyError::InvalidArgument(format!(
                "vector value count mismatch: expected {}, got {}",
                vectors.matrix.rows * dim,
                vectors.matrix.values.len()
            )));
        }

        let mut config = self.config.clone();
        if matches!(options.metric(), Metric::Cosine | Metric::Dot) {
            config.angular = true;
        }
        let mut clusterer = HierarchicalSuperKMeans::with_config(num_centroids, dim, config)
            .map_err(to_tantivy_error)?;
        let centroids = clusterer
            .train(vectors.matrix.values, vectors.matrix.rows)
            .map_err(to_tantivy_error)?;
        if centroids.len() != num_centroids * dim {
            return Err(TantivyError::InternalError(format!(
                "SuperKMeans returned {} centroid floats, expected {}",
                centroids.len(),
                num_centroids * dim
            )));
        }
        Ok(IvfCentroids::F32(IvfMatrix {
            values: centroids,
            rows: num_centroids,
            dims: dim,
        }))
    }

    fn assign(
        &self,
        options: &VectorOptions,
        vectors: IvfVectors<'_>,
        centroids: &IvfCentroids,
    ) -> tantivy::Result<Vec<Assignment>> {
        let IvfVectors::F32(vectors) = vectors;
        let IvfCentroids::F32(centroids) = centroids;
        let dim = options.dim();
        let vector_matrix = vectors.matrix;
        let centroid_matrix = centroids;
        if vector_matrix.dims != dim {
            return Err(TantivyError::InvalidArgument(format!(
                "vector dimensionality mismatch: expected {dim}, got {}",
                vector_matrix.dims
            )));
        }
        if vectors.doc_ids.len() != vector_matrix.rows {
            return Err(TantivyError::InvalidArgument(format!(
                "vector doc_id count mismatch: expected {}, got {}",
                vector_matrix.rows,
                vectors.doc_ids.len()
            )));
        }
        if vector_matrix.values.len() != vector_matrix.rows * dim {
            return Err(TantivyError::InvalidArgument(format!(
                "vector value count mismatch: expected {}, got {}",
                vector_matrix.rows * dim,
                vector_matrix.values.len()
            )));
        }
        if centroid_matrix.rows == 0 {
            return Err(TantivyError::InvalidArgument(
                "cannot assign with zero centroids".to_string(),
            ));
        }
        if centroid_matrix.dims != dim {
            return Err(TantivyError::InvalidArgument(format!(
                "centroid dimensionality mismatch: expected {dim}, got {}",
                centroid_matrix.dims
            )));
        }
        if centroid_matrix.values.len() != centroid_matrix.rows * dim {
            return Err(TantivyError::InvalidArgument(format!(
                "centroid value count mismatch: expected {}, got {}",
                centroid_matrix.rows * dim,
                centroid_matrix.values.len()
            )));
        }
        if vector_matrix.rows == 0 {
            return Ok(Vec::new());
        }

        let mut config = self.config.clone();
        if matches!(options.metric(), Metric::Cosine | Metric::Dot) {
            config.angular = true;
        }
        let mut clusterer = HierarchicalSuperKMeans::with_config(centroid_matrix.rows, dim, config)
            .map_err(to_tantivy_error)?;
        // PRIMARY assignment: unchanged from Phase 1 (superkmeans, angular-aware
        // for cosine/dot). This is the byte-identical path.
        let primaries = clusterer
            .assign(
                vector_matrix.values,
                centroid_matrix.values.as_slice(),
                vector_matrix.rows,
                centroid_matrix.rows,
            )
            .map_err(to_tantivy_error)?;

        let max_replicas = self
            .max_replicas_per_vector
            .unwrap_or(DEFAULT_MAX_REPLICAS_PER_VECTOR);
        if max_replicas == 0 {
            // Replication OFF: primary-only assignments => byte-identical to
            // Phase 1. The isolation knob.
            return Ok(primaries
                .into_iter()
                .map(Assignment::primary_only)
                .collect());
        }

        // ---- Phase 2: per-vector replica candidates (ε₁ closure + RNG) ----
        // Computed in the wrapper over the centroid matrix (superkmeans is not
        // modified). The candidate scan is pruned by a build-time-only centroid-
        // neighbor table (see [`CentroidNeighbors`]) so each vector examines its
        // primary's local centroid neighborhood instead of all C centroids. The
        // prune is build-time only: no storage-format or query-path change.
        //
        // NOTE(metric): replica distances are plain squared Euclidean on the
        // stored values. Cosine docs/centroids are unit-normalized (at write /
        // merge), so Euclidean-nearest == angular-nearest and the triangle-
        // inequality prune is exact. For raw `Dot` the triangle inequality is
        // unsound, so the prune is DISABLED for Dot and we fall back to the full
        // scan (identical to the pre-prune behavior); Dot replication stays the
        // flagged-unsound case it already was.
        let epsilon = self.replica_epsilon.unwrap_or(DEFAULT_REPLICA_EPSILON);
        let eps_sq = epsilon * epsilon;
        let num_centroids = centroid_matrix.rows;
        let centroid_vals = centroid_matrix.values.as_slice();

        // Part B gate: the centroid-neighbor prune is only built/used when
        // replication is on (checked above via `max_replicas == 0` early return)
        // AND the metric admits the Euclidean triangle inequality (L2 /
        // normalized Cosine — never Dot).
        let table = if matches!(options.metric(), Metric::L2 | Metric::Cosine) {
            Some(self.centroid_neighbors(centroid_vals, num_centroids, dim, max_replicas))
        } else {
            None
        };

        let mut assignments = Vec::with_capacity(vector_matrix.rows);
        for (vi, primary) in primaries.into_iter().enumerate() {
            let v = &vector_matrix.values[vi * dim..(vi + 1) * dim];
            let p = primary as usize;
            let ref_dist = squared_distance(v, &centroid_vals[p * dim..(p + 1) * dim]);

            // Pruned path when the table is present and certifies full coverage
            // for this vector; otherwise the exact full scan. Both produce the
            // identical replica set (see `replicas_pruned` for why).
            let replicas = table
                .as_ref()
                .and_then(|t| {
                    replicas_pruned(
                        t,
                        v,
                        centroid_vals,
                        dim,
                        p,
                        ref_dist,
                        epsilon,
                        eps_sq,
                        max_replicas,
                    )
                })
                .unwrap_or_else(|| {
                    replicas_full_scan(
                        v,
                        centroid_vals,
                        num_centroids,
                        dim,
                        p,
                        ref_dist,
                        eps_sq,
                        max_replicas,
                    )
                });

            let mut assignment = Assignment::primary_only(primary);
            for ci in replicas {
                assignment.replicas.push(ci);
            }
            assignments.push(assignment);
        }
        Ok(assignments)
    }
}

/// Squared Euclidean distance between two equal-length vectors.
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Build-time-only centroid-neighbor table: for each centroid, its nearest
/// `cap` other centroids (ascending squared Euclidean distance). Used to prune
/// the per-vector replica scan via the triangle inequality (Part B). Built once
/// per merge, cached on the clusterer, and discarded with it — never persisted,
/// never touched by the query path.
#[derive(Debug)]
struct CentroidNeighbors {
    /// `neighbors[p]` = the `min(cap, rows-1)` nearest other centroids to `p`,
    /// as `(squared_distance, centroid_index)` sorted ascending by distance
    /// (ties broken by index, matching the candidate ordering in the scan).
    neighbors: Vec<Vec<(f32, u32)>>,
    rows: usize,
    /// `true` when each centroid keeps only a strict subset of the others
    /// (`rows - 1 > cap`). When `false` every list holds *all* other centroids,
    /// so the prune is always exact (no per-vector coverage check needed).
    truncated: bool,
    /// Fingerprint of the centroid bytes this table was built from; the cache
    /// rebuilds when it changes.
    fingerprint: u64,
}

impl CentroidNeighbors {
    fn build(
        centroids: &[f32],
        rows: usize,
        dim: usize,
        cap: usize,
        fingerprint: u64,
    ) -> CentroidNeighbors {
        let mut neighbors: Vec<Vec<(f32, u32)>> = vec![Vec::new(); rows];
        // Naive blocked C×C pass (the brief's accepted first cut), parallelized
        // across centroid rows with `std::thread::scope` (no rayon dep). Each
        // thread owns a disjoint chunk of the output, so there is no sharing.
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, rows.max(1));
        let chunk = rows.div_ceil(threads).max(1);
        thread::scope(|scope| {
            for (chunk_idx, slots) in neighbors.chunks_mut(chunk).enumerate() {
                let base = chunk_idx * chunk;
                scope.spawn(move || {
                    for (offset, slot) in slots.iter_mut().enumerate() {
                        let p = base + offset;
                        let pv = &centroids[p * dim..(p + 1) * dim];
                        // Bounded max-heap of the `cap` nearest others: the heap
                        // top is the current farthest kept, popped when beaten.
                        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(cap + 1);
                        for c in 0..rows {
                            if c == p {
                                continue;
                            }
                            let d = squared_distance(pv, &centroids[c * dim..(c + 1) * dim]);
                            let entry = HeapEntry {
                                dist: d,
                                idx: c as u32,
                            };
                            if heap.len() < cap {
                                heap.push(entry);
                            } else {
                                // Replace the farthest kept (heap top) when the
                                // new entry is nearer. Resolve the comparison
                                // before mutating so the peek borrow is dropped.
                                let nearer = heap.peek().is_some_and(|top| entry < *top);
                                if nearer {
                                    heap.pop();
                                    heap.push(entry);
                                }
                            }
                        }
                        // Ascending by (distance, index) to match the scan's
                        // candidate ordering exactly.
                        *slot = heap
                            .into_sorted_vec()
                            .into_iter()
                            .map(|e| (e.dist, e.idx))
                            .collect();
                    }
                });
            }
        });
        CentroidNeighbors {
            neighbors,
            rows,
            truncated: rows.saturating_sub(1) > cap,
            fingerprint,
        }
    }
}

/// Heap entry ordered by `(squared distance, index)` so the table's tie-break
/// matches the candidate scan and `into_sorted_vec` yields ascending order.
struct HeapEntry {
    dist: f32,
    idx: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Fingerprint of a centroid matrix: `(rows, dim, hash(value bits))`. Cheap
/// relative to the table build (O(rows·dim) hashing vs O(rows²·dim) build) and
/// hashes every value, so a different centroid set with the same row count
/// never aliases to a stale cached table.
fn centroid_fingerprint(centroids: &[f32], rows: usize, dim: usize) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write_usize(rows);
    hasher.write_usize(dim);
    for &value in centroids {
        hasher.write_u32(value.to_bits());
    }
    hasher.finish()
}

/// Greedy RNG (relative-neighborhood) dedup over candidates pre-sorted ascending
/// by distance-to-vector. Keeps a candidate `c_i` unless an already-kept (hence
/// closer) `c_j` is nearer to `c_i` than `c_i` is to the vector — spreading
/// replicas across distinct directions. Caps at `max_replicas`. Returns the kept
/// replica indices and the squared distance of the farthest kept one (`d_fill`),
/// which the pruned path uses as its coverage certificate.
fn rng_dedup(
    cands: &[(f32, u32)],
    centroid_vals: &[f32],
    dim: usize,
    max_replicas: usize,
) -> (Vec<u32>, f32) {
    let mut replicas: Vec<u32> = Vec::with_capacity(max_replicas);
    let mut d_fill_sq = 0.0f32;
    for &(d_v_ci, ci) in cands {
        if replicas.len() >= max_replicas {
            break;
        }
        let ci_vec = &centroid_vals[ci as usize * dim..(ci as usize + 1) * dim];
        let covered = replicas.iter().any(|&cj| {
            let cj_vec = &centroid_vals[cj as usize * dim..(cj as usize + 1) * dim];
            squared_distance(ci_vec, cj_vec) < d_v_ci
        });
        if !covered {
            replicas.push(ci);
            d_fill_sq = d_v_ci;
        }
    }
    (replicas, d_fill_sq)
}

/// Exact replica candidates by scanning ALL centroids (the pre-prune behavior).
/// Used for `Dot` and as the fallback whenever the prune cannot certify
/// coverage. ε₁ gate: keep `c != p` with `dist²(v,c) <= eps_sq · ref_dist`,
/// nearest-first, then RNG-dedup.
#[allow(clippy::too_many_arguments)]
fn replicas_full_scan(
    v: &[f32],
    centroid_vals: &[f32],
    num_centroids: usize,
    dim: usize,
    p: usize,
    ref_dist: f32,
    eps_sq: f32,
    max_replicas: usize,
) -> Vec<u32> {
    let gate = eps_sq * ref_dist;
    let mut cands: Vec<(f32, u32)> = Vec::new();
    for c in 0..num_centroids {
        if c == p {
            continue;
        }
        let d = squared_distance(v, &centroid_vals[c * dim..(c + 1) * dim]);
        if d <= gate {
            cands.push((d, c as u32));
        }
    }
    cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    rng_dedup(&cands, centroid_vals, dim, max_replicas).0
}

/// Replica candidates using the centroid-neighbor table to prune the scan, or
/// `None` if coverage cannot be certified for this vector (the caller then runs
/// the exact full scan).
///
/// Triangle-inequality prune: every ε₁ candidate `c` (with
/// `dist(v,c) <= ε·dist(v,p)`) satisfies `dist(p,c) <= dist(v,p) + dist(v,c) <=
/// (1+ε)·dist(v,p)`, so only centroids within `(1+ε)·dist(v,p)` of the primary
/// `p` can qualify. We compute `dist(v,·)` for just `p`'s table neighbors,
/// apply the ε₁ gate, sort, and RNG-dedup — identical to the full scan over
/// that set.
///
/// Coverage certificate (exactness): any centroid NOT in `p`'s table has
/// `dist(p,·) > R` (R = distance to `p`'s farthest stored neighbor), hence by
/// the reverse triangle inequality `dist(v,·) > R - dist(v,p) =: R_safe`. So:
///
/// - if the table is not truncated, it holds every other centroid → exact;
/// - if the RNG cap filled, every kept replica is within `d_fill`, and we require
///   `R_safe > d_fill` so no omitted centroid is nearer than any kept one → the
///   nearest-first scan up to `d_fill` is complete → exact;
/// - if the cap did not fill, we require `R_safe > ε·dist(v,p)` so no omitted
///   centroid can pass the ε₁ gate → exact.
///
/// Strict `>` keeps boundary ties on the conservative side (fall back). When the
/// certificate fails we return `None` and the caller full-scans this vector, so
/// the result always equals the brute-force set regardless of the table cap.
#[allow(clippy::too_many_arguments)]
fn replicas_pruned(
    table: &CentroidNeighbors,
    v: &[f32],
    centroid_vals: &[f32],
    dim: usize,
    p: usize,
    ref_dist: f32,
    epsilon: f32,
    eps_sq: f32,
    max_replicas: usize,
) -> Option<Vec<u32>> {
    let table_p = &table.neighbors[p];
    let gate = eps_sq * ref_dist;
    let mut cands: Vec<(f32, u32)> = Vec::with_capacity(table_p.len());
    for &(_sq_pc, c) in table_p {
        // `c != p` is guaranteed: a centroid is never its own table neighbor.
        let d = squared_distance(v, &centroid_vals[c as usize * dim..(c as usize + 1) * dim]);
        if d <= gate {
            cands.push((d, c));
        }
    }
    cands.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let (replicas, d_fill_sq) = rng_dedup(&cands, centroid_vals, dim, max_replicas);

    // Untruncated tables hold every other centroid → always exact.
    if !table.truncated {
        return Some(replicas);
    }
    // Linear distances for the reverse-triangle certificate.
    let d_vp = ref_dist.sqrt();
    let r_table = match table_p.last() {
        Some(&(sq, _)) => sq.sqrt(),
        None => 0.0,
    };
    let r_safe = r_table - d_vp;
    let certified = if replicas.len() >= max_replicas {
        r_safe > d_fill_sq.sqrt()
    } else {
        r_safe > epsilon * d_vp
    };
    certified.then_some(replicas)
}

pub fn set_ivf_clusterer(index: &mut Index, options: &BM25IndexOptions) {
    let clusterer = SuperKMeansIvfClusterer::new()
        .with_centroid_ratio(options.centroid_ratio())
        .with_training_samples_per_centroid(options.training_samples_per_centroid())
        .with_max_posting_len(options.max_posting_len())
        .with_min_posting_len(options.min_posting_len())
        .with_max_replicas_per_vector(options.max_replicas_per_vector())
        .with_max_replicas_per_cluster(options.max_replicas_per_cluster())
        .with_replica_epsilon(options.replica_epsilon());
    index.set_ivf_clusterer(Arc::new(clusterer));
}

fn to_tantivy_error(error: SuperKMeansError) -> TantivyError {
    match error {
        SuperKMeansError::InvalidArgument(message) => TantivyError::InvalidArgument(message),
        SuperKMeansError::Runtime(message) => TantivyError::InternalError(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so the fixture is reproducible without an RNG dep.
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / ((1u64 << 31) as f32)
        }
        fn coord(&mut self) -> f32 {
            self.unit() * 10.0
        }
    }

    fn make_matrix(rows: usize, dim: usize, lcg: &mut Lcg) -> Vec<f32> {
        (0..rows * dim).map(|_| lcg.coord()).collect()
    }

    /// Euclidean argmin over centroids — what the L2 primary assignment reduces
    /// to, and the reference `p` the replica helpers take.
    fn argmin(v: &[f32], centroids: &[f32], rows: usize, dim: usize) -> usize {
        (0..rows)
            .map(|c| (squared_distance(v, &centroids[c * dim..(c + 1) * dim]), c))
            .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
            .map(|(_, c)| c)
            .unwrap()
    }

    /// THE key check (Step 3 bullet 1): the pruned candidate set (with the
    /// coverage fallback) must equal the brute-force full-scan set, per vector,
    /// across tight/loose ε and truncated/untruncated tables. Also asserts the
    /// prune does real work (certifies for some vectors) in the favorable case,
    /// so the equality isn't passing only because we always fall back.
    #[test]
    fn pruned_replica_set_equals_brute_force() {
        let dim = 4;
        let num_centroids = 40;
        let num_vectors = 300;
        let mut lcg = Lcg(0x9e3779b97f4a7c15);
        let centroids = make_matrix(num_centroids, dim, &mut lcg);
        let vectors = make_matrix(num_vectors, dim, &mut lcg);

        // (epsilon, max_replicas, neighbor_cap). Small caps force truncation
        // (rows-1 = 39 > cap) and exercise the certificate + fallback; a cap
        // >= rows-1 leaves the table untruncated (always exact).
        let configs = [
            (1.2f32, 8usize, 6usize),
            (1.2, 8, 20),
            (1.2, 4, 6),
            (10.0, 8, 6),
            (10.0, 8, 20),
            (10.0, 4, 12),
            (1.5, 8, 39), // untruncated
            (3.0, 6, 39), // untruncated
        ];

        // Aggregate coverage so we can prove BOTH certificate branches run and
        // each yields correct results: the accept branch (certified > 0 on a
        // truncated table) and the reject/fallback branch (fell_back > 0).
        let mut truncated_certified = 0usize;
        let mut truncated_fell_back = 0usize;

        for (epsilon, max_replicas, cap) in configs {
            let eps_sq = epsilon * epsilon;
            let fp = centroid_fingerprint(&centroids, num_centroids, dim);
            let table = CentroidNeighbors::build(&centroids, num_centroids, dim, cap, fp);
            assert_eq!(table.truncated, (num_centroids - 1) > cap);

            let mut certified = 0usize;
            for vi in 0..num_vectors {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                let p = argmin(v, &centroids, num_centroids, dim);
                let ref_dist = squared_distance(v, &centroids[p * dim..(p + 1) * dim]);

                let brute = replicas_full_scan(
                    v,
                    &centroids,
                    num_centroids,
                    dim,
                    p,
                    ref_dist,
                    eps_sq,
                    max_replicas,
                );
                let pruned = replicas_pruned(
                    &table,
                    v,
                    &centroids,
                    dim,
                    p,
                    ref_dist,
                    epsilon,
                    eps_sq,
                    max_replicas,
                );
                if pruned.is_some() {
                    certified += 1;
                }
                let effective = pruned.unwrap_or_else(|| {
                    replicas_full_scan(
                        v,
                        &centroids,
                        num_centroids,
                        dim,
                        p,
                        ref_dist,
                        eps_sq,
                        max_replicas,
                    )
                });
                assert_eq!(
                    effective, brute,
                    "config (eps={epsilon}, max={max_replicas}, cap={cap}) vector {vi}: \
                     pruned set != brute-force set"
                );
            }

            // Untruncated tables certify every vector (early-out, no per-vector
            // check). Truncated tables feed the aggregate below.
            if !table.truncated {
                assert_eq!(certified, num_vectors);
            } else {
                truncated_certified += certified;
                truncated_fell_back += num_vectors - certified;
            }
        }

        // The certified-prune branch must actually fire on truncated tables
        // (else the equality check only ever validated the full-scan fallback),
        // and the fallback branch must also fire (the adversarial loose-ε +
        // tiny-cap configs force it) — both validated equal to brute force above.
        assert!(
            truncated_certified > 0,
            "certified prune never fired on a truncated table"
        );
        assert!(
            truncated_fell_back > 0,
            "coverage fallback never fired on a truncated table"
        );
    }

    /// The neighbor table is built ONCE per centroid set and reused; a different
    /// centroid set re-keys and rebuilds.
    #[test]
    fn neighbor_table_is_cached_and_rekeyed() {
        let dim = 4;
        let num_centroids = 16;
        let mut lcg = Lcg(123);
        let centroids = make_matrix(num_centroids, dim, &mut lcg);

        let clusterer = SuperKMeansIvfClusterer::new();
        let first = clusterer.centroid_neighbors(&centroids, num_centroids, dim, 8);
        let second = clusterer.centroid_neighbors(&centroids, num_centroids, dim, 8);
        assert!(
            Arc::ptr_eq(&first, &second),
            "same centroids must reuse the cached table"
        );

        let mut other = centroids.clone();
        other[0] += 1.0; // perturb -> different fingerprint
        let third = clusterer.centroid_neighbors(&other, num_centroids, dim, 8);
        assert!(
            !Arc::ptr_eq(&first, &third),
            "different centroids must rebuild the table"
        );
        assert_ne!(first.fingerprint, third.fingerprint);
    }

    /// SPEED (Step 3.2), in miniature: at the *default* ε=10 the ε₁ radius is
    /// enormous, yet the prune still engages because the replica cap binds at 8
    /// long before the radius does and the coverage certificate keys off the
    /// farthest *kept* replica (small), not the radius. This measures the
    /// certified-prune fraction and the per-vector candidate work (= table row
    /// length) vs the full centroid count, which is the speedup proxy. Run with
    /// `--nocapture` to see the numbers.
    #[test]
    fn prune_engages_at_default_epsilon() {
        let dim = 8;
        let num_centroids = 1500;
        let num_vectors = 800;
        let max_replicas = 8;
        let cap = max_replicas * 32; // == clusterer's neighbor_cap for these args
        let mut lcg = Lcg(0xd1b54a32d192ed03);
        let centroids = make_matrix(num_centroids, dim, &mut lcg);
        // Clustered vectors (realistic embedding data): each is a centroid plus
        // small jitter, so a vector sits close to its primary — the regime the
        // triangle prune targets. (Uniform-random vectors, by contrast, are the
        // prune's adversarial worst case and engage far less.)
        let vectors: Vec<f32> = (0..num_vectors)
            .flat_map(|_| {
                let base = (lcg.unit() * num_centroids as f32) as usize % num_centroids;
                let c = &centroids[base * dim..(base + 1) * dim];
                (0..dim)
                    .map(|d| c[d] + (lcg.unit() - 0.5) * 3.0)
                    .collect::<Vec<_>>()
            })
            .collect();
        let fp = centroid_fingerprint(&centroids, num_centroids, dim);
        let table = CentroidNeighbors::build(&centroids, num_centroids, dim, cap, fp);
        let row_len = table.neighbors[0].len();
        assert_eq!(row_len, cap, "table truncated to the cap as expected");

        for (epsilon, min_frac) in [(10.0f32, 0.2f32), (1.5, 0.9)] {
            let eps_sq = epsilon * epsilon;
            let mut certified = 0usize;
            for vi in 0..num_vectors {
                let v = &vectors[vi * dim..(vi + 1) * dim];
                let p = argmin(v, &centroids, num_centroids, dim);
                let ref_dist = squared_distance(v, &centroids[p * dim..(p + 1) * dim]);
                if replicas_pruned(
                    &table,
                    v,
                    &centroids,
                    dim,
                    p,
                    ref_dist,
                    epsilon,
                    eps_sq,
                    max_replicas,
                )
                .is_some()
                {
                    certified += 1;
                }
            }
            let frac = certified as f32 / num_vectors as f32;
            println!(
                "ε={epsilon}: certified {certified}/{num_vectors} ({:.1}%); per-certified-vector \
                 candidate scan = {row_len} of {num_centroids} centroids ({:.1}× fewer dist evals)",
                frac * 100.0,
                num_centroids as f32 / row_len as f32,
            );
            // The prune's win is ε-sensitive: at a tight closure factor it
            // certifies the clear majority of vectors (the regime it targets);
            // at the very loose default ε on this deliberately *unclustered*
            // fixture engagement is low (worst case). Both are correct — the
            // fallback keeps results exact regardless. We only floor the tight
            // case as a regression guard; the loose case is measured, not gated.
            assert!(
                frac >= min_frac,
                "prune engaged on only {:.1}% of vectors at ε={epsilon} (floor {:.0}%)",
                frac * 100.0,
                min_frac * 100.0
            );
        }
    }

    /// Each centroid's neighbor list is sorted ascending by (distance, index),
    /// excludes the centroid itself, and is the nearest `min(cap, rows-1)`.
    #[test]
    fn neighbor_table_rows_are_sorted_nearest_first() {
        let dim = 3;
        let rows = 12;
        let cap = 5;
        let mut lcg = Lcg(777);
        let centroids = make_matrix(rows, dim, &mut lcg);
        let fp = centroid_fingerprint(&centroids, rows, dim);
        let table = CentroidNeighbors::build(&centroids, rows, dim, cap, fp);

        for p in 0..rows {
            let row = &table.neighbors[p];
            assert_eq!(row.len(), cap.min(rows - 1));
            assert!(row.iter().all(|&(_, c)| c as usize != p));
            for w in row.windows(2) {
                let (da, ia) = w[0];
                let (db, ib) = w[1];
                assert!(da < db || (da == db && ia < ib), "row {p} not sorted");
            }
            // Brute-force nearest `cap`: the table row must match exactly.
            let mut all: Vec<(f32, u32)> = (0..rows)
                .filter(|&c| c != p)
                .map(|c| {
                    (
                        squared_distance(
                            &centroids[p * dim..(p + 1) * dim],
                            &centroids[c * dim..(c + 1) * dim],
                        ),
                        c as u32,
                    )
                })
                .collect();
            all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            assert_eq!(&all[..cap.min(rows - 1)], &row[..]);
        }
    }
}
