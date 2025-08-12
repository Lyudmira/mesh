


---


you will need to create a conda/minimanba environment and start from there. all dependencies shall be installed via minimanba or conda.

### Quick Start

Install [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) and create the environment:

```
micromamba env create -f environment.yml
micromamba activate mesh-env
```

The `environment.yml` specifies the core dependencies used by this project:

- `openvdb` – sparse voxel and distance field processing
- `pcl` – point cloud utilities
- `cgal-cpp` – computational geometry algorithms
- `igl` – libigl geometry processing bindings
- `open3d` – visualization and Python helpers
- `pymeshlab` – additional mesh filters
- `pytest` – test runner

All packages are sourced from the `conda-forge` channel and installed without superuser privileges.

### Pipeline Demo

A minimal end-to-end pipeline is provided in `recon/src/pipeline.cpp`. It leverages PCL for point cloud preprocessing, OpenVDB for level-set construction and dual contouring, CGAL for surface metrics, and libigl for mesh export. Compile and run:

```
g++ recon/src/pipeline.cpp -std=c++17 $(pkg-config --cflags --libs openvdb) $(pcl_config --cflags) $(pcl_config --libs) $(pkg-config --cflags --libs cgal) -o pipeline
./pipeline input.ply output.obj
```

The Python wrapper `recon/scripts/pipeline.py` invokes the binary and loads the resulting mesh using Open3D and PyMeshLab for verification.



---



# Dual-Layer Hybrid: A Complete Pipeline Design for Robust Indoor Point Cloud Reconstruction to a Visualizable Mesh

> **One-Sentence Goal:** **First, use an Unsigned Distance Field (UDF) and Graph Cut to create a robust, flat, and closed "building shell" representing only the interior space. Then, separately reconstruct the "clutter and details" within a normal-offset band outside the shell and stitch them together.**
>
> **Applicable Conditions:** **No ground truth poses**, **no reliance on deep learning**, **poor quality normals**, and **uneven sampling density**.
> **Hardware:** RTX 4090 + 64 GB RAM.
> **Scene Size:** Approx. 40×10×3 meters.
> **Input:** High-density colored point cloud (multiple dense versions: 2/4/6/14 million points).

---

## 0. Team Preamble (Scope, Terminology, and Deliverables)

*   **Deliverables**
    1.  **Main Model (Shell + Details Merged):** A 2-manifold, non-self-intersecting mesh containing only interior surfaces. **Walls, floors, and ceilings must be planar**, while **small objects should be preserved with high fidelity**.
    2.  **Level of Detail (LOD) for Large-Scene Visualization:** GLB/PLY/OBJ formats with vertex colors. Also export partition labels for "planar domain" and "detail domain" to facilitate downstream shading and interaction.
    3.  **Quality Control (QC) Report:** Metrics including geometric error, planarity residuals, orthogonality deviation, detail recall, topological validity, and free-space correctness (see Section 7).

*   **Core Design Philosophy**
    *   **Divide and Conquer:** The architectural "shell" (globally stable, topologically controlled) and "details" (locally faithful) are processed in separate tracks and then merged.
    *   **Explicit Geometry First:** Construct implicit representations (like UDF) only when necessary. **Strictly avoid extrapolating exterior walls out of thin air.**
    *   **Strong Finishing:** A finalization pipeline of Alpha Wrapping, plane snapping, and intersection line re-projection to achieve **engineering-grade** topology and appearance.

---

## 1. Toolchain and Code Organization (Recommendations)

This pipeline can be implemented by combining several robust open-source libraries, minimizing the need to reinvent the wheel.

*   **Core Geometry Libraries:**
    *   **OpenVDB:** The primary tool for sparse voxel grids, distance field construction, and Dual Contouring mesh generation.
    *   **PCL (Point Cloud Library):** A complete toolchain for point cloud I/O, filtering, normal estimation, MLS/RIMLS smoothing, region growing, RANSAC plane segmentation, and Greedy Projection Triangulation (GP3).
    *   **CGAL (Computational Geometry Algorithms Library):** Provides high-quality algorithms for Alpha Wrapping, Polygon Mesh Processing (PMP) for topological repair and Booleans, Shape Detection, Surface Mesh Simplification, and WLOP simplification.
    *   **libigl:** Offers easy-to-use wrappers (often depending on CGAL) for `mesh_boolean` operations and other geometric processing, ideal for mesh merging and welding.
    *   **Open3D / PyMeshLab:** Useful for visualization, Python bindings, and supplementary filters like Poisson-disk sampling and RIMLS-based implicit reconstruction.

*   **Language and Framework:**
    *   **C++:** For the core reconstruction logic and graph structures (performance-critical).
    *   **Python:** For pipeline orchestration, visualization, and metrics calculation.

*   **Directory Structure:**
    ```
    recon/
      configs/           # YAML parameter sets (global / shell / detail / lod / qc)
      scripts/           # Python entry points: pipeline.py, qc.py, lod.py ...
      src/               # C++ core: udf_builder/, graph_cut/, dual_contouring/, wrap/
      modules/           # Bindings and utilities for PCL/OpenVDB/CGAL
      data/              # Input point clouds, caches, intermediate chunks
      outputs/           # mesh_final/, mesh_shell/, mesh_detail/, reports/
    ```

---

## 2. Stage 0: Data Auditing and Cleaning

**Goal:** Calibrate the data, remove obvious noise, and gather statistics for subsequent adaptive parameters.

1.  **Coordinate and Unit Standardization:** Unify all coordinates to meters (m).
2.  **Coarse Cropping:** Based on a known indoor bounding box plus a 5–10 cm buffer, remove sparse outdoor points. Be conservative to avoid clipping window-frame edges.
3.  **Density Statistics:** For each point, compute the k-NN distance distribution ($d_{kNN}$ for k=16/32), including the median and 95th percentile. This provides a baseline for adaptive radii and voxel refinement thresholds.
4.  **Outlier Removal:**
    *   **Implementation:** Use PCL's `StatisticalOutlierRemoval` (mean ± 2–3σ) and `RadiusOutlierRemoval` (radius r = 2·median($d_{kNN}$), min neighbors = 5–10).
5.  **Normal Estimation:**
    *   **Initial Estimation:** Use PCA on k-NN (k=64–96; adaptive radius = 2–3·median($d_{kNN}$)) to output a normal and curvature for each point.
    *   **Global Orientation:** Propagate orientation along a Minimum Spanning Tree (MST) of the k-NN graph with minimal flips. For difficult regions, use a least-squares sign field based on a Delaunay graph as a fallback.
    *   **Implementation:** This can be achieved using PCL's `NormalEstimation` / `NormalEstimationOMP`.
6.  **Feature-Preserving Denoising:**
    *   **Planar Regions** (low curvature, low color variance): Use a lightweight MLS.
    *   **Detail/Edge Regions** (high curvature or color gradient): Switch to **RIMLS, WLOP, or Bilateral filtering** to avoid blurring sharp edges and thin structures.
    *   **Implementation:**
        *   **MLS:** PCL's `MovingLeastSquares` can perform smoothing and normal re-estimation.¹
        *   **Bilateral:** PCL's `BilateralFilter` preserves edges while smoothing.²
        *   **WLOP:** CGAL's `wlop_simplify_and_regularize_point_set` produces a "denoised, outlier-free, and evenly distributed set of particles."⁴
7.  **Density Balancing:**
    *   **Planar Regions:** Voxel downsample to 5–10 mm.
    *   **Detail Regions:** Preserve original density or use Poisson-disk sampling to 3–5 mm.
    *   **Goal:** Control the final triangle count and avoid overfitting ripples in flat areas.
    *   **Implementation:** Use Open3D's `sample_points_poisson_disk` for balanced sampling.⁵

> **Note:** Color (RGB) **participates** in boundary/feature detection (e.g., walls typically have low color variance) and can be used as a joint weight in denoising and partitioning.

---

## 3. Stage 1: Building Shell (UDF + Graph Cut → DC → Wrap → Plane Snapping)

### 3.1 Sparse Voxelization and UDF Construction

*   **OpenVDB Grid:**
    *   **Coarse Voxel Size:** 30–50 mm.
    *   **Refinement Triggers:** Refine to 10–15 mm if any of the following are met:
        *   High local curvature / high normal variance.
        *   Large color gradient (potential detail/boundary).
        *   Near a candidate plane (< 5 cm away from a fitted plane).
        *   Distance to the nearest point is small (< 2·median($d_{kNN}$)).
    *   Activate voxels only within a **"point cloud dilation band"** (e.g., 30–50 cm) and the **indoor reachable zone** to avoid unnecessary computation.

*   **Unsigned Distance Field (UDF) and Confidence ($w_v$):**
    *   $d_v=$ Distance from voxel center to the nearest point (via k-D tree/k-NN).
    *   Confidence $w_v$: A composite of **point density**, **color consistency**, and **local planarity** (low plane-fitting residual → high $w_v$; high color noise → low $w_v$).
    *   **Truncation:** $d_v \leftarrow \min(d_v, d_{\text{max}})$, where $d_{\text{max}}$ = 3·voxel size (to suppress influence from distant points).

### 3.2 Energy Model (Binary Labels: `inside` / `free`)

$$
E(L)=\sum_v D_v(L_v) + \lambda \sum_{(u,v)\in \mathcal N} w_{uv} \,[L_u \neq L_v]
$$

*   **Data Term $D_v$:**
    *   $D_v(\text{inside}) = w_v \cdot \phi(d_v)$, with $\phi(d)=\min(d/\tau,1)$ and $\tau=8-15$ mm.
    *   $D_v(\text{free}) = \alpha \cdot \mathbf{1}\{v\in \mathcal R_{\text{free}}\} + \gamma \cdot C_v$
        *   $\mathcal R_{\text{free}}$: **Free-space reachable domain** (see 3.3).
        *   $C_v$: **Visibility/penetration cost** (if a ray from an indoor seed passes through many points to reach voxel *v*, *v* is more likely `inside`).
    *   **Region-Adaptive Parameters (Crucial):**
        *   **Planar Regions:** Higher $\lambda$ (0.8–1.2 relative scale), moderate $\alpha$ (0.5–0.8).
        *   **Detail Regions:** Lower $\lambda$ (0.3–0.6), slightly lower $\alpha$ (0.3–0.6) to avoid "eating" free space around fine details.

*   **Smoothness Term $w_{uv}$ (Anisotropic):**
    *   Default 6-connectivity.
    *   Within the same candidate plane: $w_{uv}$ × 2 (**preserves thin walls**).
    *   Across different planes: $w_{uv}$ × 0.5 (reduces over-smoothing).
    *   Density-adaptive: Lower $w_{uv}$ in sparse areas (avoids "bridging"); higher in dense areas (suppresses noise).

### 3.3 Free-Space Seeds and Visibility Mask

*   **Automated Indoor Seed Selection:**
    *   Sample voxel centers at 0.5–1.0 m intervals within the **point cloud's convex hull**. Reject candidates closer than 20–30 cm to any point.
    *   Perform a Breadth-First Search (BFS) flood-fill through **empty voxels**. The **largest connected component** becomes $\mathcal R_{\text{free}}$.
*   **Height/Color Priors:**
    *   Apply a strong `free` bias to voxels in floor/ceiling Z-ranges; apply an `inside` bias to regions with consistent wall colors.
*   **Door/Window Opening Stabilization:** Reinforce the `free` reward at boundaries of openings (near plane intersections + adjacent to free space) to prevent them from being sealed.

### 3.4 Graph Construction and Min-Cut

*   **Nodes:** Active voxels; **Edges:** 6-connectivity (can be extended to 18/26).
*   **Memory Estimation (for intuition):**
    *   Scene surface area ≈ 1100 m² (floor/ceiling + walls). With 5 cm voxels, a single layer is ≈ 440k voxels. A 6-layer band results in ≈ **2.6M** nodes.
    *   6-connectivity edges ≈ 3 × nodes ≈ 7–8M arcs. A standard max-flow solver will use a few hundred MB of RAM. **64 GB is ample.**
*   **Solver Implementation:**
    *   Use a standalone implementation of the **Boykov-Kolmogorov algorithm** (e.g., `BK-Maxflow`⁷) or the version in **Boost.Graph**⁸. Both are stable, with BK often being faster on sparse graphs.

### 3.5 Surface Extraction (Dual Contouring with Sparse Octree Support)

*   **Intersection Point & Normal:**
    *   Find the intersection point by root-finding on the cut edge.
    *   Use the normal from the local fitted plane first; otherwise, use the gradient of the distance field ($\nabla d$) as a fallback.
*   **QEF Solution:** Add **anisotropic regularization** (weighting the normal direction 2–3× higher) to prevent sharp edges from being rounded.
*   **Implementation:**
    *   **OpenVDB's `VolumeToMesh`** function has a built-in Dual Contouring implementation and can be called directly.⁹
    *   Alternatively, standalone C++ code like **Minimal Dual Contouring**¹⁰ or concepts from **Manifold/Occupancy-based Dual Contouring**¹¹ can be used for reference, especially for extracting the QEF solver.
*   **Output:** A watertight shell mesh (topology not yet fully legalized).

### 3.6 Legalization and Geometric Regularization

*   **Alpha Wrapping:**
    *   Apply a small offset (1–3 mm; 0.5–1 mm in detail areas) to create a unified 2-manifold, removing self-intersections and slivers.
    *   **Implementation:** Use **CGAL's `3D Alpha Wrapping`** package.¹²
*   **Plane Snapping (only for large, truly flat regions):**
    *   Fit plane clusters using RANSAC/MLESAC (area > 0.5–1.0 m²).
    *   Project only vertices with `|distance| < 8–15 mm` and **avoid** color/curvature boundaries to protect features like baseboards, door frames, and window sills.
    *   **Implementation:** Use **CGAL's `Shape Detection`** (Efficient RANSAC)¹³ or PCL's `SACSegmentation`¹⁴ to identify planes.
*   **Intersection Line Re-projection:**
    *   For adjacent planes, find their intersection line using least-squares fitting. Re-project nearby edges/vertices onto this line to create **perfectly straight right angles**.
    *   **Implementation:** Use **CGAL's `Line_3`** API for quick implementation.

> The result is a **Building Shell**: topologically robust, with straight walls and right angles, and existing only on the interior (the exterior is empty). This serves as the global skeleton.

---

## 4. Stage 2: Detail Layer (Faithful Reconstruction in a Normal-Offset Band)

**Concept:** Reconstruct details from the original points **only within a thin band** (5–10 cm) along the shell's outward normal. The shell acts as a hard boundary; details only add to it, never modifying it.

### 4.1 Geometry Extraction

*   **Band Definition:** Sample points on the shell mesh and perform a nearest neighbor query along their outward normals. Extract all original points where "distance to shell ≤ 5–10 cm and the point is on the outside of the shell."
*   **Denoising:** A light RIMLS/Bilateral filter (radius ≤ 1.5·median($d_{kNN}$)) can be applied within this thin band.

### 4.2 Detail Reconstruction Routes (Choose one, A is the default)

**A. Adaptive GP3 (Greedy Projection Triangulation)**

*   **Points and Normals:** Use the oriented normals from Stage 0, which are of sufficient quality for GP3.
*   **Adaptive Parameters:**
    *   **Local Max Edge Length:** $R_i = c \cdot d_{kNN}(i)$, with c = 2.8–3.2.
    *   **μ:** 2.5–3 (controls the tightness of candidate edges).
    *   **Angle Thresholds:** 45° for planar regions; 25–35° for edges/high-curvature areas.
    *   **Triangle Angles:** Min 10–15°, Max 110–120°.
    *   Enable **normal consistency**.
*   **Implementation:** Use **PCL's `GreedyProjectionTriangulation`**, which exposes all these parameters.¹⁶
*   **Post-Cleaning:** Trim dangling edges and filter small, isolated components (noise clusters).

**B. RIMLS → Dual Contouring (if local normals are unstable or thin structures dominate)**

*   **Local Implicit Field:** Use RIMLS with a narrow bandwidth (β=1.4–1.8) and 10–12 mm voxels.
*   **Surface Extraction:** Use Dual Contouring (normals from $\nabla f$).
*   **Implementation:** **PyMeshLab** provides filters for RIMLS implicit surface fitting (`compute_curvature_and_color_rimls_per_vertex`) and Marching Cubes/DC reconstruction (`generate_marching_cubes_rimls`).¹⁷ ¹⁸
*   **Style:** Smoother than GP3, thin features may be slightly blunted, but more robust in areas with bad normals.

### 4.3 Detail Layer Legalization

*   **PMP Repair:** Use **CGAL's Polygon Mesh Processing** (PMP) to fix self-intersections and non-manifold geometry.¹⁵
*   **Simplification:** Use Quadric Edge Collapse (QEM) to moderately reduce triangles while **preserving sharp edges** (protected based on normal angle and curvature thresholds).
    *   **Implementation:** Use **CGAL's `Surface_mesh_simplification`** module, which offers strategies like Garland-Heckbert.¹⁹

---

## 5. Stage 3: Fusion (Shell-dominated, Detail-supplemented)

*   **Priority:** The shell is dominant. Details are preserved only in regions where they are `> δ` (3–6 mm) away from the shell, preventing small, co-planar items from being "flattened" into the wall.
*   **Fusion Strategy:**
    *   **Boolean Operations (Union/Difference):**
        *   `Shell ∪ Detail`. If a detail intersects the shell, the shell's surface is prioritized, and only the protruding part of the detail is kept.
        *   **Implementation:** Use **libigl's `igl/boolean/mesh_boolean`**, which internally calls CGAL for robust operations.²⁰
    *   **Alternative: Second Alpha Wrap:** Wrap the combined "Shell + Detail" mesh with a small offset (1–2 mm). Follow up with another plane snapping pass to pull any slightly shrunken wall vertices back to their planes.
*   **Welding and Cleanup:** After merging, perform **vertex welding** (threshold = 1–2 mm) and **re-estimate normals** for consistent orientation.
    *   **Implementation:** Use PMP's `remove_duplicated_vertices` or libigl's `unique_rows`.
*   **Color Assignment:**
    *   **Shell Vertices:** Sample colors from their neighborhood within the same plane, weighted by distance.
    *   **Detail Vertices:** Inherit color directly from the original points.
    *   **Junctions:** Blend colors (e.g., 70% shell + 30% detail) to prevent harsh transitions.
    *   **Implementation:** This requires custom logic, but **MeshLab's "Transfer Vertex Color"** filter can serve as a reference.

---

## 6. Stage 4: LOD, Export, and Performance

*   **LOD Generation:**
    *   **Region-Aware Simplification:** Heavily simplify planar regions (potentially into quads) while lightly simplifying detail regions.
    *   **Target Triangle Counts:** Full model: 3–8M; LOD1: 1–2M; LOD2: 300–600K.
    *   **Implementation:** Use **CGAL's `Surface_mesh_simplification`**, which allows flexible policies to set different simplification weights for different regions.²¹
*   **Export Formats:** GLB (with vertex colors), PLY (for development), OBJ (for interoperability).
    *   **Implementation:** PCL, Assimp, or Open3D can handle these formats.
*   **Performance Tips:**
    *   **Chunking:** Use 5m³ chunks with a 0.2m overlap. Run the full shell+detail pipeline within each chunk, then use Wrap/Boolean operations to stitch the chunks.
    *   **Memory:** The overall process should fit within < 32 GB of RAM, making 64 GB comfortable.

---

## 7. Stage 7: Quality Control (Required Metrics and Thresholds)

| Metric | Threshold | Implementation Tool |
| :--- | :--- | :--- |
| **Planarity Residual (RMS/90th %)** | ≤ 3–6 mm / ≤ 8–12 mm | Custom script after plane fitting. |
| **Orthogonality Deviation (wall-floor/wall-wall)** | 90° ± 1.0–1.5° | Custom script on face normals from detected planes. |
| **Detail Recall** | Hausdorff dist to GT ≤ 10–15 mm | **CGAL's** `approximate_Hausdorff_distance()`. |
| | Area/Volume deviation ≤ 10–15% | Custom geometry script. |
| **Topological Validity** | Self-intersections=0; Non-manifold=0 | Use checks from **CGAL PMP** or **libigl**. |
| **Fidelity (Mesh-to-Point)** | Median dist ≤ 5–8 mm; 95% ≤ 20 mm | k-D tree nearest neighbor search. |
| **Interior-Only Correctness** | Voxel coverage ratio: interior ≈ 1, exterior ≈ 0 | Custom check using seed-based flood fill. |
| **Rendering Health** | Consistent normals, no large flipped patches | **libigl's** `is_vertex_manifold`, `orient_outward_ao`. |

---

## 8. Failure Modes & Quick Fixes (Runbook)

*   **Thin wall squeezed out:** Increase co-planar neighbor weight (×3); refine voxels to 10–12 mm; decrease $\lambda$ in detail regions.
*   **Exterior "scum" / sealed openings:** Increase $\alpha$; deepen free-seed flood-fill; increase visibility penalty.
*   **Features (baseboards) flattened:** Decrease plane snapping distance threshold (15→8 mm) and add color-boundary protection.
*   **Double walls / ghosting:** Add a `free` bias to voxels between the two layers in the graph cut.
*   **Dull details:** Refine voxels in detail areas; decrease $\lambda$; use anisotropic QEF; switch to detail reconstruction Route B (RIMLS→DC).
*   **Self-intersections / non-manifold:** Increase Alpha Wrap offset (1→2-3 mm). If details are lost, re-apply plane snapping.

---

## 9. Execution Steps (Team Field Manual)

> **Recommendation:** First run and tune parameters on a low-resolution version (2–4M points) before processing the full 14M point cloud.

**Step 0: Audit & Pre-processing**
`scripts/pipeline.py --stage 0 --in data/room_all.ply --cfg configs/global.yaml`
*   **Outputs:** `data/clean.ply`, `data/normals.ply`, `reports/audit.json`

**Step 1: Shell (UDF+Cut→DC→Wrap→Snap)**
`--stage shell_udf` (Writes UDF/weights to `.vdb`)
`--stage shell_cut` (Solves graph cut, outputs labels to `.vdb`)
`--stage shell_dc` (Extracts mesh: `outputs/mesh_shell_raw.ply`)
`--stage shell_wrap_snap` (Finalizes shell: `outputs/mesh_shell.ply`)

**Step 2: Detail Layer**
`--stage detail_extract` (Selects points: `data/detail_band.ply`)
`--stage detail_recon --method gp3` (or `rimls_dc`) → `outputs/mesh_detail.ply`

**Step 3: Fusion & LOD**
`--stage fuse` (Boolean/Wrap) → `outputs/mesh_final.ply`
`--stage lod --targets LOD1,LOD2` → `outputs/mesh_final_lod*.glb`

**Step 4: Quality Control**
`scripts/qc.py --gt data/normals.ply --mesh outputs/mesh_final.ply --cfg configs/qc.yaml`
*   **Output:** `reports/qc.html` (tables, heatmaps, histograms)

---

## 10. Parameter "Cookbook" (How to Tune)

*   **Voxels:**
    *   *Symptom:* "Details are sharp, but walls are wavy." → **Coarsen** wall voxels (30→40–50 mm) & increase $\lambda$.
    *   *Symptom:* "Thin features disappear." → **Refine** detail voxels (15→10–12 mm), increase co-planar weight (×3), & decrease $\lambda$.
*   **$\alpha$ (free-space reward):**
    *   *Symptom:* "Exterior scum." → Increase $\alpha$.
    *   *Symptom:* "Doorways are sealed." → Decrease $\alpha$ & increase visibility penalty at openings.
*   **$\lambda$ (smoothness):**
    *   *Symptom:* "Walls are not flat enough/still wavy." → Increase $\lambda$ in planar regions.
    *   *Symptom:* "Small objects are blurred." → Decrease $\lambda$ in detail regions.
*   **Plane Snapping:**
    *   *Symptom:* "Baseboard is erased." → Decrease snapping distance threshold (15→8 mm) & add color boundary protection.
    *   *Symptom:* "Corners are not straight." → Increase intersection line re-projection radius (2→3 ring neighbors).

---

## 11. Why This Method is "Robust + Truthful" (Technical Endorsement for the Team)

*   **Robust:** The shell layer establishes its topology and inside/outside division via **global optimization (graph cut)**, which does **not** require good normals. Alpha Wrapping strictly enforces a 2-manifold result. Snapping and re-projection ensure that "what should be flat is flat, and what should be straight is straight."
*   **Truthful:** The detail layer is reconstructed from **original points** in a thin band outside the shell (using GP3 or RIMLS→DC), **avoiding over-smoothing**. The shell-dominant fusion prevents "detail-polluting-the-wall," achieving a balance of regularity and fidelity.
*   **Interior-Only:** The combination of free-space seeds and a visibility penalty explicitly encodes in the energy function that the "outside is free," preventing the algorithm from hallucinating exterior walls.

---


---

# Open Source Tools and Reusable Code Report for an Indoor Point Cloud Reconstruction Project

Based on the provided "dual-layer hybrid indoor point cloud reconstruction to a visualizable mesh pipeline," this report outlines available open-source libraries and existing implementations to reduce redundant development efforts. The report is organized according to the main tasks in each stage of the pipeline, categorized as follows:

1.  **Mature libraries that can be directly integrated as dependencies** (widely used, well-documented, and generally require no modification).
2.  **Open-source implementations suitable for full or partial copying into the project** (e.g., standalone algorithm implementations or header-only libraries).
3.  **Approaches or reference code that can be consulted for parts that still require custom development** (existing similar research or implementations that can be used for rapid prototyping and validation).

The report also provides links to relevant documentation or papers for further reading and verification.

---

## 1. Data Auditing and Cleaning (Stage 0)

| Functionality | Recommended Dependency/Open Source Implementation | Description |
| :--- | :--- | :--- |
| **Coordinate Unification, Cropping, and Statistical Filtering** | PCL (Point Cloud Library) | PCL provides data reading, unit handling, and common filters like `StatisticalOutlierRemoval` and `RadiusOutlierRemoval` for removing outliers and performing sparse filtering. |
| **Normal Estimation and Orientation** | PCL | PCL's `MovingLeastSquares` implements MLS smoothing and normal re-estimation. The documentation states it "implements the MLS algorithm for data smoothing and improving normal estimation"<sup>1</sup>. It can be combined with `NormalEstimation` / `NormalEstimationOMP` to complete normal estimation and global orientation. |
| **Feature-Preserving Denoising (MLS / Bilateral / WLOP / RIMLS)** | PCL `BilateralFilter`, CGAL Bilateral & WLOP | PCL provides `BilateralFilter` for smoothing noise while preserving edges; its documentation describes it as "a bilateral filter implementation for point cloud data"<sup>2</sup>. CGAL's Point Set Processing module offers `bilateral_smooth_point_set`<sup>3</sup> for normal-sensitive bilateral projection. The same module also implements the `wlop_simplify_and_regularize_point_set` function, which realizes the Weighted Locally Optimal Projection (WLOP) simplification algorithm, capable of producing a "denoised, outlier-free, and evenly distributed set of particles"<sup>4</sup>. These algorithms can be used directly or adapted. |
| **Poisson-disk / Local Density-Balancing Sampling** | Open3D or header-only library | Open3D's `sample_points_poisson_disk` function, based on Yuksel 2015's sample elimination method, samples a blue-noise point set with uniform spacing from a mesh or point cloud<sup>5</sup>. A header-only library (e.g., `thinks/poisson`) can also be used for quick implementation. |
| **Local Clustering and Region Growing** | PCL `RegionGrowing` | PCL's `RegionGrowing` implements the classic region growing segmentation algorithm. Its documentation indicates it "implements the famous region growing segmentation algorithm"<sup>6</sup>. It can be used for partitioning based on a combination of curvature and color. |

## 2. Building Shell Reconstruction (Stage 1)

### 2.1 Sparse Voxelization and Unsigned Distance Field (UDF)

- **OpenVDB** - Provides tools for sparse voxel grids and distance fields. OpenVDB's `ParticlesToLevelSet` can convert a point cloud into a distance field. After generation, its `tools::NanFilter` and `FastSweeping` can be used to compute an unsigned distance field. Since this pipeline only requires an unsigned distance, OpenVDB's existing functions can be used directly with custom confidence weights.

### 2.2 Graph Cut Optimization (inside/free labels)

| Functionality | Dependency/Reference Implementation | Description |
| :--- | :--- | :--- |
| **Max-Flow/Min-Cut Solving** | BK Maxflow (Kolmogorov & Zabih) or Boost | The `maxflow-v3.01` library is a standalone implementation of the Boykov-Kolmogorov algorithm for finding the max-flow/min-cut on any graph<sup>7</sup>. `Boost.Graph` also provides the `boykov_kolmogorov_max_flow` function<sup>8</sup>. Either can be chosen to avoid implementing the complex graph-cut algorithm from scratch. |
| **Adjacency Graph Construction and Weighting** | Self-developed, based on OpenVDB's active voxel indexing | Based on the pipeline, a 6/18/26-connectivity adjacency graph needs to be constructed, with weights assigned to data and smoothness terms. The max-flow libraries mentioned above are only responsible for the solving part. |
| **Free-Space Flooding and Visibility Estimation** | Self-developed | This requires implementation based on indoor seed voxels and ray-casting statistics, a part for which there is no general-purpose library. Existing visibility analysis code (e.g., PCL's `ray_cast` module) can be referenced to reduce the workload. |

### 2.3 Surface Extraction (Dual Contouring)

| Functionality | Available Implementation | Description |
| :--- | :--- | :--- |
| **Dual Contouring** | 1. **OpenVDB `VolumeToMesh`** – OpenVDB's `volumeToMesh` function uses the Dual Contouring algorithm to generate quad meshes from a voxel grid. The OpenVDB author confirmed in a mailing list that this tool has a built-in Dual Contouring implementation<sup>9</sup>. It can be called directly to generate the shell mesh. <br> 2. **`libigl` or `Minimal Dual Contouring`** – Several standalone implementations are available for reference, such as the C++ Dual Contouring implementation published by Emil Ernerfeldt, whose README states it provides a simple dual contouring mesh generation function and is in the public domain<sup>10</sup>. You can also consult the GitHub projects for Manifold Dual Contouring and Self-supervised Dual Contouring. These implementations provide extractable code for QEF solving and feature preservation. |
| **QEF Solving** | `Manifold Dual Contouring` / `Occupancy-based Dual Contouring` | A standalone QEF solver can be extracted from the libraries mentioned above. The `Occupancy-based Dual Contouring` paper notes that its method solves the QEF with the help of auxiliary 1D/2D sample points, and "only uses 1D and 2D points to solve the quadratic error function, then meshes"<sup>11</sup>. This can serve as a reference for an optimized QEF. |

### 2.4 Topological Repair and Geometric Regularization

| Functionality | Recommended Dependency/Implementation | Description |
| :--- | :--- | :--- |
| **Alpha Wrapping (Shrink-Wrapping)** | CGAL 3D Alpha Wrapping | CGAL's `Alpha Wrapping` package provides a conservative shrink-wrapping algorithm that can wrap an input point set or triangle mesh into a "watertight, non-self-intersecting, and orientable 2-manifold mesh"<sup>12</sup>. It is suitable for legalizing the topology of the shell mesh. |
| **Plane Detection and Snapping** | CGAL Shape Detection (Efficient RANSAC) | CGAL's `Shape Detection` module uses Efficient RANSAC to detect planes, cylinders, and other shapes from point clouds and their normals<sup>13</sup>. The detected planes can be used for subsequent vertex projection and intersection line re-projection. PCL's `RANSAC` (`SACSegmentation`) can also be used for plane model fitting<sup>14</sup>. |
| **Intersection Line Re-projection** | Self-developed or based on CGAL | Based on the detected intersection lines, use least-squares fitting to find the line equation and re-project nearby vertices. The `CGAL::Line_3` related API can be used for rapid implementation. |
| **Mesh Stitching and Repair** | CGAL Polygon Mesh Processing (PMP) | PMP provides the `stitch_borders` function to repair duplicated edges or vertices, ensuring each edge is connected to exactly two faces<sup>15</sup>. PMP's `remove_self_intersections` and `orient_polygon_soup` tools can also be used to repair the shell mesh. |

## 3. Detail Layer Reconstruction (Stage 2)

### 3.1 Detail-Band Point Extraction

This part mainly involves selecting points along the outward normal direction of the shell mesh. It can be implemented from scratch using basic operations like kd-tree nearest neighbor queries and normal projection. The nearest neighbor search interfaces in PCL or libigl can reduce the development burden.

### 3.2 Detail Reconstruction

| Method | Available Implementation | Description |
| :--- | :--- | :--- |
| **A. Greedy Projection Triangulation (GP3)** | PCL `GreedyProjectionTriangulation` | PCL provides the `GreedyProjectionTriangulation` class for greedy triangulation based on local 2D projections. The documentation describes it as "an implementation of a greedy triangulation algorithm for 3D points based on local 2D projections"<sup>16</sup>. The class exposes interfaces to set search radius, max angle, normal consistency, etc., which align with the adaptive parameters in the proposed pipeline. |
| **B. RIMLS → Dual Contouring** | MeshLab / PyMeshLab RIMLS filters | PyMeshLab provides filters like `compute_curvature_and_color_rimls_per_vertex` and `generate_marching_cubes_rimls`, which implement RIMLS implicit surface fitting and RIMLS-based Marching Cubes reconstruction<sup>17, 18</sup>. These filters can be called directly to obtain a detail mesh within the thin band, avoiding the need to write RIMLS and the implicit solver from scratch. |
| **C. WLOP Simplification/Sampling (Optional)** | CGAL WLOP | CGAL's `wlop_simplify_and_regularize_point_set` function provides Weighted Locally Optimal Projection simplification. The algorithm can produce a "denoised, outlier-free, and evenly distributed" set of particles<sup>4</sup>; a parallel version is also available. It can be used to simplify or regularize the point set of the detail layer. |

### 3.3 Detail Layer Legalization and Simplification

- **PMP Repair** - CGAL PMP provides functions for repairing self-intersections and non-manifold geometry.
- **QEM Simplification** - CGAL's `Surface_mesh_simplification` module implements an edge-collapse-based triangle mesh simplification algorithm and offers strategies like Lindstrom-Turk and Garland-Heckbert<sup>19</sup>. It can be used to moderately reduce the face count in detail regions while preserving sharp edges.

## 4. Merging and Boolean Operations (Stage 3)

| Functionality | Available Implementation | Description |
| :--- | :--- | :--- |
| **Boolean Union/Difference** | `libigl` `mesh_boolean` (depends on CGAL) | The libigl FAQ suggests directly calling `igl/boolean/mesh_boolean` for union operations on closed manifold meshes<sup>20</sup>. This function internally calls CGAL for robust Boolean operations and self-intersection repair, and can handle multiple components. Alternatively, CGAL's `Polygon_mesh_processing::corefinement::polygon_mesh_boolean()` can be used. |
| **Second Alpha Wrapping** | CGAL Alpha Wrapping | After merging the shell and detail layers, perform another Alpha Wrapping with a small offset to unify the topology and smooth the boundaries. This method was already introduced in the shell reconstruction stage. |
| **Welding/Vertex Merging** | PMP or `libigl` | PMP's `remove_duplicated_vertices`, `merge_vertices`, and libigl's `unique_rows` can be used for vertex welding. Afterwards, normals should be re-estimated to ensure consistent orientation. |
| **Color Baking and Blending** | Self-developed or Meshlab Filter | PCL/CGAL mainly handle geometry; color blending requires custom weight-based sampling. MeshLab's "Transfer Vertex Color" filter can also be referenced for color projection. |

## 5. LOD Generation and Export (Stage 4)

| Functionality | Available Implementation | Description |
| :--- | :--- | :--- |
| **Region-Aware Simplification** | CGAL `Surface mesh simplification` | This package provides flexible edge-collapse strategies and stopping conditions, allowing different weights to be set based on regions (planes vs. details) to achieve adaptive simplification<sup>21</sup>. |
| **Mesh Format Export** | PCL / Assimp / Open3D | PCL can read/write PLY/OBJ formats. Open3D can export GLB/PLY and provides a relevant API. Alternatively, the Assimp library can be used to uniformly handle reading and writing of various 3D formats. |

## 6. Quality Control Metrics and Evaluation

The quality control report needs to compute metrics such as geometric error, planarity residuals, orthogonality deviation, detail recall, and topological validity. The following are reusable tools:

- **Hausdorff Distance** - CGAL provides `Polygon_mesh_processing::approximate_Hausdorff_distance()` to compute the approximate Hausdorff distance between two meshes.
- **Normal Consistency and Flipped-Face Detection** – `libigl` provides functions like `is_vertex_manifold`, `orient_outward_ao`, and `bw_unwind` to check if a mesh is 2-manifold and to unify normal directions.
- **Curvature and Right-Angle Measurement** - In PCL, `PrincipalCurvaturesEstimation` can be used. In CGAL, angles can be computed after calculating face normals.

## 7. Parts Still Requiring Custom Development but with References

Although many tools can be used directly, some key steps still need to be implemented or customized according to the project's specific requirements:

1.  **Adaptive Voxel Refinement and Weighting** - OpenVDB provides a sparse voxel framework, but the logic for dynamically refining voxels based on curvature, color gradients, etc., and for assigning different confidence levels needs to be custom-developed. However, OpenVDB example code and its mesh decimation example (which uses voxel tagging) can be referenced.
2.  **Free-Space Seed Selection and Visibility Penalty** - This requires performing flooding and ray-tracing in the voxel space. Modules for ray casting from OpenVDB, Embree, or PCL can be combined for implementation. There is no single library that can directly generate this energy term.
3.  **Plane Snapping and Intersection Line Re-projection** - Although CGAL can detect planes, protecting features like color boundaries and baseboards during projection is an engineering detail that requires custom implementation combining point cloud color and curvature features.
4.  **Color Blending and Texture Synthesis** - This requires defining custom weights and performing neighborhood color sampling. MeshLab's color transfer filter can serve as a reference.

## 8. Summary

Leveraging existing open-source tools can significantly reduce development effort:

- **OpenVDB** is responsible for sparse voxels, distance field construction, and Dual Contouring mesh generation.
- **PCL** provides a complete point cloud processing toolchain, including filtering, normal estimation, MLS/RIMLS smoothing, region growing, RANSAC plane segmentation, and Greedy Projection Triangulation.
- **CGAL** offers high-quality algorithms like Alpha Wrapping, Shape Detection (plane detection), Polygon Mesh Processing (topological repair/Boolean operations), Surface Mesh Simplification, and WLOP simplification.
- **libigl** (which depends on CGAL) provides an easy-to-use `mesh_boolean` and other geometric processing functions, suitable for mesh Booleans and welding.
- **PyMeshLab/Open3D** supplement with practical filters like RIMLS implicit reconstruction and Poisson-disk sampling.

By combining these components, the development can focus on the unique energy design and engineering details of the proposed pipeline. For parts requiring custom development, referencing related papers and open-source code (like Manifold Dual Contouring, Occupancy-based Dual Contouring) can reduce research and implementation costs.

---

### References

1.  Point Cloud Library (PCL): `pcl::MovingLeastSquares< PointInT, PointOutT >` Class Template Reference
    <https://pointclouds.org/documentation/classpcl_1_1_moving_least__squares.html>
2.  Point Cloud Library (PCL): `pcl::BilateralFilter< PointT >` Class Template Reference
    <https://pointclouds.org/documentation/classpcl_1_1_bilateral__filter.html>
3.  CGAL 6.0.1 - Point Set Processing: Algorithms
    <https://doc.cgal.org/latest/Point_set_processing_3/group__PkgPointSetProcessing3Algorithms.html>
4.  CGAL 6.0.1 - Point Set Processing: Algorithms
    <https://doc.cgal.org/latest/Point_set_processing_3/group__PkgPointSetProcessing3Algorithms.html>
5.  `open3d.geometry.sample_points_poisson_disk` — Open3D 0.7.0 documentation
    <https://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.sample_points_poisson_disk.html>
6.  Point Cloud Library (PCL): `pcl::RegionGrowing< PointT, NormalT >` Class Template Reference
    <https://pointclouds.org/documentation/classpcl_1_1_region__growing.html>
7.  Computer Vision at Waterloo - Code
    <https://vision.cs.uwaterloo.ca/code/>
8.  Boost Graph Library: Boykov-Kolmogorov Maximum Flow
    <https://www.boost.org/doc/libs/1_88_0/libs/graph/doc/boykov_kolmogorov_max_flow.html>
9.  VolumeToMesh generating Quads or Triangles
    <https://groups.google.com/g/openvdb-forum/c/uvZzhnNa4kQ>
10. raw.githubusercontent.com
    <https://raw.githubusercontent.com/emilk/Dual-Contouring/master/README.md>
11. Occupancy-Based Dual Contouring
    <https://occupancy-based-dual-contouring.github.io/>
12. CGAL 6.0.1 - 3D Alpha Wrapping: User Manual
    <https://doc.cgal.org/latest/Alpha_wrap_3/index.html>
13. CGAL 6.0.1 - Shape Detection: User Manual
    <https://doc.cgal.org/latest/Shape_detection/index.html>
14. Point Cloud Library (PCL): `pcl::SACSegmentation< PointT >` Class Template Reference
    <https://pointclouds.org/documentation/classpcl_1_1_s__a__c__segmentation.html>
15. CGAL 4.11.3 - Polygon Mesh Processing: User Manual
    <https://doc.cgal.org/4.11.3/Polygon_mesh_processing/index.html>
16. Point Cloud Library (PCL): `pcl::GreedyProjectionTriangulation< PointInT >` Class Template Reference
    <https://pointclouds.org/documentation/classpcl_1_1_greedy__projection__triangulation.html>
17. List of Filters — PyMeshLab documentation
    <https://pymeshlab.readthedocs.io/en/latest/filter_list.html>
18. List of Filters — PyMeshLab documentation
    <https://pymeshlab.readthedocs.io/en/latest/filter_list.html>
19. CGAL 6.0.1 - Triangulated Surface Mesh Simplification: User Manual
    <https://doc.cgal.org/latest/Surface_mesh_simplification/index.html>
20. FAQ - libigl
    <https://libigl.github.io/faq/>
21. CGAL 6.0.1 - Triangulated Surface Mesh Simplification: User Manual
    <https://doc.cgal.org/latest/Surface_mesh_simplification/index.html> 

