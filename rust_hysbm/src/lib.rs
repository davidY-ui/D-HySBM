use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;


struct PsiPart {
    hye_idx: Vec<i64>,
    node_idx: Vec<i64>,
    psi_vals: Vec<f64>,
}

struct PsiPartData {
    // Flat positions in SharedCSCMessages.data
    positions: Vec<i64>,

    // Internal storage order:
    // K values for position0,
    // K values for position1,
    // ...
    // stored in this order
    psi_vals: Vec<f64>,
}

fn i64_to_usize(x: i64, name: &str) -> Result<usize, String> {
    if x < 0 {
        return Err(format!("{name} contains negative value: {x}"));
    }

    Ok(x as usize)
}


fn weighted_logsumexp(values: &[(f64, f64)]) -> f64 {
    let mut max_v = f64::NEG_INFINITY;

    for &(log_v, weight) in values.iter() {
        if weight > 0.0 && log_v.is_finite() && log_v > max_v {
            max_v = log_v;
        }
    }

    if !max_v.is_finite() {
        return f64::NEG_INFINITY;
    }

    let mut sum = 0.0;

    for &(log_v, weight) in values.iter() {
        if weight > 0.0 && log_v.is_finite() {
            sum += weight * (log_v - max_v).exp();
        }
    }

    if sum <= 0.0 {
        f64::NEG_INFINITY
    } else {
        max_v + sum.ln()
    }
}


fn read_log_message(
    log_messages: &[f64],
    nnz: usize,
    k: usize,
    data_pos: usize,
    out: &mut [f64],
) {
    // log_messages shape = (K, nnz), C-contiguous.
    for a in 0..k {
        out[a] = log_messages[a * nnz + data_pos];
    }
}


fn update_eta_non_tilde(
    eta_old: &[f64],
    log_s_old: &[f64],
    log_message: &[f64],
    p: &[f64],
    k: usize,
    eta_new: &mut [f64],
) {
    for a in 0..k {
        let mut values: Vec<(f64, f64)> = Vec::with_capacity(2 * k + 1);

        // addends[:, 0] = eta
        values.push((eta_old[a], 1.0));

        // addends[:, 1:K+1] = log_message + log_s
        for b in 0..k {
            values.push((log_message[b] + log_s_old[b], 1.0));
        }

        // addends[:, K+1:] = log_message, weights = p[a, b]
        for b in 0..k {
            values.push((log_message[b], p[a * k + b]));
        }

        eta_new[a] = weighted_logsumexp(&values);
    }
}


fn update_eta_tilde(
    eta_old: f64,
    log_s_old: &[f64],
    log_message: &[f64],
    k: usize,
) -> f64 {
    let mut values: Vec<(f64, f64)> = Vec::with_capacity(k + 1);

    values.push((eta_old, 1.0));

    for b in 0..k {
        values.push((log_message[b] + log_s_old[b], 1.0));
    }

    weighted_logsumexp(&values)
}


fn update_log_s(
    log_s_old: &[f64],
    log_message: &[f64],
    p: &[f64],
    k: usize,
    log_s_new: &mut [f64],
) {
    for a in 0..k {
        let mut values: Vec<(f64, f64)> = Vec::with_capacity(k + 1);

        // addends[:, 0] = log_s
        values.push((log_s_old[a], 1.0));

        // addends[:, 1:] = log_message, weights = p[a, b]
        for b in 0..k {
            values.push((log_message[b], p[a * k + b]));
        }

        log_s_new[a] = weighted_logsumexp(&values);
    }
}


fn compute_eta_skip(
    p: &[f64],
    log_messages: &[f64],
    nnz: usize,
    k: usize,
    start: usize,
    d: usize,
    skip_i: usize,
) -> Vec<f64> {
    let mut eta = vec![f64::NEG_INFINITY; k];
    let mut log_s = vec![f64::NEG_INFINITY; k];

    let mut eta_new = vec![f64::NEG_INFINITY; k];
    let mut log_s_new = vec![f64::NEG_INFINITY; k];
    let mut log_message = vec![0.0; k];

    let f = d - 1;
    let mut idx = 0usize;

    for local_pos in 0..d {
        if local_pos == skip_i {
            continue;
        }

        let data_pos = start + local_pos;

        read_log_message(
            log_messages,
            nnz,
            k,
            data_pos,
            &mut log_message,
        );

        update_eta_non_tilde(
            &eta,
            &log_s,
            &log_message,
            p,
            k,
            &mut eta_new,
        );

        eta.copy_from_slice(&eta_new);

        if idx < f - 1 {
            update_log_s(
                &log_s,
                &log_message,
                p,
                k,
                &mut log_s_new,
            );

            log_s.copy_from_slice(&log_s_new);
        }

        idx += 1;
    }

    eta
}


fn compute_eta_tilde_one(
    p: &[f64],
    log_messages: &[f64],
    nnz: usize,
    k: usize,
    start: usize,
    d: usize,
) -> f64 {
    let mut eta = f64::NEG_INFINITY;
    let mut log_s = vec![f64::NEG_INFINITY; k];

    let mut log_s_new = vec![f64::NEG_INFINITY; k];
    let mut log_message = vec![0.0; k];

    for local_pos in 0..d {
        let data_pos = start + local_pos;

        read_log_message(
            log_messages,
            nnz,
            k,
            data_pos,
            &mut log_message,
        );

        eta = update_eta_tilde(
            eta,
            &log_s,
            &log_message,
            k,
        );

        if local_pos < d - 1 {
            update_log_s(
                &log_s,
                &log_message,
                p,
                k,
                &mut log_s_new,
            );

            log_s.copy_from_slice(&log_s_new);
        }
    }

    eta
}


fn selected_size_index(
    selected_hye_sizes: &[i64],
    d: usize,
) -> Result<usize, String> {
    for (idx, &size_value) in selected_hye_sizes.iter().enumerate() {
        if size_value == d as i64 {
            return Ok(idx);
        }
    }

    Err(format!(
        "hyperedge size {d} is not in selected_hye_sizes"
    ))
}


fn p_layer_slice<'a>(
    p_layers: &'a [f64],
    n_layers: usize,
    k: usize,
    layer_idx: usize,
) -> Result<&'a [f64], String> {
    if layer_idx >= n_layers {
        return Err(format!(
            "layer_idx {layer_idx} out of range n_layers={n_layers}"
        ));
    }

    let start = layer_idx * k * k;
    let end = start + k * k;

    Ok(&p_layers[start..end])
}


fn process_one_hye_single_p(
    hye_value: i64,
    indptr: &[i64],
    indices: &[i64],
    log_messages: &[f64],
    nnz: usize,
    p: &[f64],
    k: usize,
    mask: Option<&[bool]>,
) -> Result<PsiPart, String> {
    let hye = i64_to_usize(hye_value, "hye")?;

    if hye + 1 >= indptr.len() {
        return Err(format!(
            "hye index {hye} out of range, indptr len={}",
            indptr.len()
        ));
    }

    let start = i64_to_usize(indptr[hye], "indptr")?;
    let end = i64_to_usize(indptr[hye + 1], "indptr")?;

    if end < start || end > indices.len() {
        return Err(format!(
            "invalid CSC range for hye={hye}: start={start}, end={end}"
        ));
    }

    let d = end - start;

    if d < 2 {
        return Err(format!(
            "hyperedge size must be at least 2, got d={d} for hye={hye}"
        ));
    }

    let mut hye_idx = Vec::new();
    let mut node_idx = Vec::new();
    let mut psi_vals = Vec::new();

    for local_i in 0..d {
        let data_pos = start + local_i;

        if let Some(mask_data) = mask {
            if !mask_data[data_pos] {
                continue;
            }
        }

        let eta = compute_eta_skip(
            p,
            log_messages,
            nnz,
            k,
            start,
            d,
            local_i,
        );

        hye_idx.push(hye_value);
        node_idx.push(indices[data_pos]);
        psi_vals.extend_from_slice(&eta);
    }

    Ok(PsiPart {
        hye_idx,
        node_idx,
        psi_vals,
    })
}


fn process_one_hye_loop_p(
    hye_value: i64,
    indptr: &[i64],
    indices: &[i64],
    log_messages: &[f64],
    nnz: usize,
    p_layers: &[f64],
    n_layers: usize,
    selected_hye_sizes: &[i64],
    k: usize,
    mask: Option<&[bool]>,
) -> Result<PsiPart, String> {
    let hye = i64_to_usize(hye_value, "hye")?;

    if hye + 1 >= indptr.len() {
        return Err(format!(
            "hye index {hye} out of range, indptr len={}",
            indptr.len()
        ));
    }

    let start = i64_to_usize(indptr[hye], "indptr")?;
    let end = i64_to_usize(indptr[hye + 1], "indptr")?;

    if end < start || end > indices.len() {
        return Err(format!(
            "invalid CSC range for hye={hye}: start={start}, end={end}"
        ));
    }

    let d = end - start;

    if d < 2 {
        return Err(format!(
            "hyperedge size must be at least 2, got d={d} for hye={hye}"
        ));
    }

    let layer_idx = selected_size_index(selected_hye_sizes, d)?;

    let p = p_layer_slice(
        p_layers,
        n_layers,
        k,
        layer_idx,
    )?;

    process_one_hye_single_p(
        hye_value,
        indptr,
        indices,
        log_messages,
        nnz,
        p,
        k,
        mask,
    )
}

//new 
fn process_one_hye_single_p_data(
    hye_value: i64,
    indptr: &[i64],
    log_messages: &[f64],
    nnz: usize,
    p: &[f64],
    k: usize,
    mask: Option<&[bool]>,
) -> Result<PsiPartData, String> {
    let hye = i64_to_usize(hye_value, "hye")?;

    if hye + 1 >= indptr.len() {
        return Err(format!(
            "hye index {hye} out of range, indptr len={}",
            indptr.len()
        ));
    }

    let start = i64_to_usize(
        indptr[hye],
        "indptr",
    )?;

    let end = i64_to_usize(
        indptr[hye + 1],
        "indptr",
    )?;

    if end < start || end > nnz {
        return Err(format!(
            "invalid CSC range for hye={hye}: \
             start={start}, end={end}, nnz={nnz}"
        ));
    }

    let d = end - start;

    if d < 2 {
        return Err(format!(
            "hyperedge size must be at least 2, \
             got d={d} for hye={hye}"
        ));
    }

    // Estimate how many messages in this hyperedge actually need updating.
    let selected_count = match mask {
        Some(mask_data) => {
            (start..end)
                .filter(|&pos| mask_data[pos])
                .count()
        }
        None => d,
    };

    let mut positions =
        Vec::with_capacity(selected_count);

    let mut psi_vals =
        Vec::with_capacity(selected_count * k);

    for local_i in 0..d {
        let data_pos = start + local_i;

        if let Some(mask_data) = mask {
            if !mask_data[data_pos] {
                continue;
            }
        }

        let eta = compute_eta_skip(
            p,
            log_messages,
            nnz,
            k,
            start,
            d,
            local_i,
        );

        // Key detail:
        // Store neither hye_idx nor node_idx;
        // store positions directly in the CSC data array.
        positions.push(data_pos as i64);

        psi_vals.extend_from_slice(&eta);
    }

    Ok(PsiPartData {
        positions,
        psi_vals,
    })
}

fn process_one_hye_loop_p_data(
    hye_value: i64,
    indptr: &[i64],
    log_messages: &[f64],
    nnz: usize,
    p_layers: &[f64],
    n_layers: usize,
    selected_hye_sizes: &[i64],
    k: usize,
    mask: Option<&[bool]>,
) -> Result<PsiPartData, String> {
    let hye = i64_to_usize(
        hye_value,
        "hye",
    )?;

    if hye + 1 >= indptr.len() {
        return Err(format!(
            "hye index {hye} out of range, indptr len={}",
            indptr.len()
        ));
    }

    let start = i64_to_usize(
        indptr[hye],
        "indptr",
    )?;

    let end = i64_to_usize(
        indptr[hye + 1],
        "indptr",
    )?;

    if end < start || end > nnz {
        return Err(format!(
            "invalid CSC range for hye={hye}: \
             start={start}, end={end}, nnz={nnz}"
        ));
    }

    let d = end - start;

    if d < 2 {
        return Err(format!(
            "hyperedge size must be at least 2, \
             got d={d} for hye={hye}"
        ));
    }

    let layer_idx = selected_size_index(
        selected_hye_sizes,
        d,
    )?;

    let p = p_layer_slice(
        p_layers,
        n_layers,
        k,
        layer_idx,
    )?;

    process_one_hye_single_p_data(
        hye_value,
        indptr,
        log_messages,
        nnz,
        p,
        k,
        mask,
    )
}

fn concat_psi_parts<'py>(
    py: Python<'py>,
    parts: Vec<PsiPart>,
    k: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let total_rows: usize = parts.iter().map(|x| x.hye_idx.len()).sum();

    let mut hye_out: Vec<i64> = Vec::with_capacity(total_rows);
    let mut node_out: Vec<i64> = Vec::with_capacity(total_rows);
    let mut psi_out: Vec<f64> = Vec::with_capacity(total_rows * k);

    for part in parts {
        hye_out.extend(part.hye_idx);
        node_out.extend(part.node_idx);
        psi_out.extend(part.psi_vals);
    }

    let psi_array = Array2::from_shape_vec(
        (total_rows, k),
        psi_out,
    )
    .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    Ok((
        Array1::from_vec(hye_out).into_pyarray(py),
        Array1::from_vec(node_out).into_pyarray(py),
        psi_array.into_pyarray(py),
    ))
}

fn concat_psi_parts_data<'py>(
    py: Python<'py>,
    parts: Vec<PsiPartData>,
    k: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let total_rows: usize = parts
        .iter()
        .map(|x| x.positions.len())
        .sum();

    let mut positions_out: Vec<i64> =
        Vec::with_capacity(total_rows);

    // Python ultimately expects:
    //
    // shape = (K, R)
    //
    // Allocate the K * R array directly.
    let mut psi_out =
        vec![0.0_f64; k * total_rows];

    let mut row_offset = 0usize;

    for part in parts {
        let rows = part.positions.len();

        if part.psi_vals.len() != rows * k {
            return Err(
                PyValueError::new_err(
                    format!(
                        "internal psi size mismatch: \
                         rows={rows}, k={k}, psi_len={}",
                        part.psi_vals.len()
                    )
                )
            );
        }

        positions_out.extend_from_slice(
            &part.positions
        );

        // Internal order of part.psi_vals:
        //
        // [r0_a0, r0_a1, ...,
        //  r1_a0, r1_a1, ...]
        //
        // Convert to the final layout:
        //
        // psi_out[a, r]
        for r in 0..rows {
            for a in 0..k {
                psi_out[
                    a * total_rows
                        + row_offset
                        + r
                ] =
                    part.psi_vals[
                        r * k + a
                    ];
            }
        }

        row_offset += rows;
    }

    let psi_array = Array2::from_shape_vec(
        (k, total_rows),
        psi_out,
    )
    .map_err(
        |e| PyValueError::new_err(
            format!("{e}")
        )
    )?;

    Ok((
        Array1::from_vec(
            positions_out
        ).into_pyarray(py),

        psi_array.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, log_messages, p, mask=None))]
fn compute_psi_all_single_p<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let log_messages_slice = log_messages.as_slice()?;
    let p_slice = p.as_slice()?;

    let log_shape = log_messages.shape();
    let p_shape = p.shape();

    if indptr_slice.is_empty() {
        return Err(PyValueError::new_err("indptr is empty"));
    }

    if log_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "log_messages must have shape (K, nnz)",
        ));
    }

    if p_shape.len() != 2 || p_shape[0] != p_shape[1] {
        return Err(PyValueError::new_err(
            "p must have shape (K, K)",
        ));
    }

    let k = p_shape[0];
    let nnz = log_shape[1];

    if log_shape[0] != k {
        return Err(PyValueError::new_err(
            "log_messages.shape[0] must equal K",
        ));
    }

    if indices_slice.len() != nnz {
        return Err(PyValueError::new_err(
            "indices length must equal nnz",
        ));
    }

    let mask_slice = match &mask {
        Some(m) => {
            let s = m.as_slice()?;
            if s.len() != nnz {
                return Err(PyValueError::new_err(
                    "mask length must equal nnz",
                ));
            }
            Some(s)
        }
        None => None,
    };

    let e = indptr_slice.len() - 1;

    let results: Vec<Result<PsiPart, String>> = py.detach(|| {
        (0..e)
            .into_par_iter()
            .map(|hye| {
                process_one_hye_single_p(
                    hye as i64,
                    indptr_slice,
                    indices_slice,
                    log_messages_slice,
                    nnz,
                    p_slice,
                    k,
                    mask_slice,
                )
            })
            .collect()
    });

    let mut parts = Vec::with_capacity(results.len());

    for r in results {
        parts.push(r.map_err(PyValueError::new_err)?);
    }

    concat_psi_parts(py, parts, k)
}


#[pyfunction]
#[pyo3(signature = (indptr, indices, log_messages, p_layers, selected_hye_sizes, mask=None))]
fn compute_psi_all_loop_p<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p_layers: PyReadonlyArray3<'py, f64>,
    selected_hye_sizes: PyReadonlyArray1<'py, i64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let log_messages_slice = log_messages.as_slice()?;
    let p_layers_slice = p_layers.as_slice()?;
    let selected_slice = selected_hye_sizes.as_slice()?;

    let log_shape = log_messages.shape();
    let p_shape = p_layers.shape();

    if indptr_slice.is_empty() {
        return Err(PyValueError::new_err("indptr is empty"));
    }

    if log_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "log_messages must have shape (K, nnz)",
        ));
    }

    if p_shape.len() != 3 || p_shape[1] != p_shape[2] {
        return Err(PyValueError::new_err(
            "p_layers must have shape (L, K, K)",
        ));
    }

    let n_layers = p_shape[0];
    let k = p_shape[1];
    let nnz = log_shape[1];

    if log_shape[0] != k {
        return Err(PyValueError::new_err(
            "log_messages.shape[0] must equal K",
        ));
    }

    if selected_slice.len() != n_layers {
        return Err(PyValueError::new_err(
            "selected_hye_sizes length must equal p_layers.shape[0]",
        ));
    }

    if indices_slice.len() != nnz {
        return Err(PyValueError::new_err(
            "indices length must equal nnz",
        ));
    }

    let mask_slice = match &mask {
        Some(m) => {
            let s = m.as_slice()?;
            if s.len() != nnz {
                return Err(PyValueError::new_err(
                    "mask length must equal nnz",
                ));
            }
            Some(s)
        }
        None => None,
    };

    let e = indptr_slice.len() - 1;

    let results: Vec<Result<PsiPart, String>> = py.detach(|| {
        (0..e)
            .into_par_iter()
            .map(|hye| {
                process_one_hye_loop_p(
                    hye as i64,
                    indptr_slice,
                    indices_slice,
                    log_messages_slice,
                    nnz,
                    p_layers_slice,
                    n_layers,
                    selected_slice,
                    k,
                    mask_slice,
                )
            })
            .collect()
    });

    let mut parts = Vec::with_capacity(results.len());

    for r in results {
        parts.push(r.map_err(PyValueError::new_err)?);
    }

    concat_psi_parts(py, parts, k)
}


//single-p data API
#[pyfunction]
#[pyo3(
    signature = (
        indptr,
        log_messages,
        p,
        mask=None
    )
)]
fn compute_psi_all_single_p_data<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let indptr_slice =
        indptr.as_slice()?;

    let log_messages_slice =
        log_messages.as_slice()?;

    let p_slice =
        p.as_slice()?;

    let log_shape =
        log_messages.shape();

    let p_shape =
        p.shape();

    if indptr_slice.is_empty() {
        return Err(
            PyValueError::new_err(
                "indptr is empty"
            )
        );
    }

    if log_shape.len() != 2 {
        return Err(
            PyValueError::new_err(
                "log_messages must have shape (K, nnz)"
            )
        );
    }

    if p_shape.len() != 2
        || p_shape[0] != p_shape[1]
    {
        return Err(
            PyValueError::new_err(
                "p must have shape (K, K)"
            )
        );
    }

    let k = p_shape[0];
    let nnz = log_shape[1];

    if log_shape[0] != k {
        return Err(
            PyValueError::new_err(
                "log_messages.shape[0] must equal K"
            )
        );
    }

    if indptr_slice[0] != 0 {
        return Err(
            PyValueError::new_err(
                "indptr[0] must be 0"
            )
        );
    }

    if *indptr_slice.last().unwrap()
        != nnz as i64
    {
        return Err(
            PyValueError::new_err(
                format!(
                    "indptr[-1] must equal nnz: \
                     got {}, expected {}",
                    indptr_slice
                        .last()
                        .unwrap(),
                    nnz
                )
            )
        );
    }

    let mask_slice =
        match &mask {
            Some(m) => {
                let s = m.as_slice()?;

                if s.len() != nnz {
                    return Err(
                        PyValueError::new_err(
                            "mask length must equal nnz"
                        )
                    );
                }

                Some(s)
            }

            None => None,
        };

    let e =
        indptr_slice.len() - 1;

    let results:
        Vec<Result<PsiPartData, String>> =
        py.detach(|| {
            (0..e)
                .into_par_iter()
                .map(|hye| {
                    process_one_hye_single_p_data(
                        hye as i64,
                        indptr_slice,
                        log_messages_slice,
                        nnz,
                        p_slice,
                        k,
                        mask_slice,
                    )
                })
                .collect()
        });

    let mut parts =
        Vec::with_capacity(
            results.len()
        );

    for r in results {
        parts.push(
            r.map_err(
                PyValueError::new_err
            )?
        );
    }

    concat_psi_parts_data(
        py,
        parts,
        k,
    )
}

// loop / multilayer data API

#[pyfunction]
#[pyo3(
    signature = (
        indptr,
        log_messages,
        p_layers,
        selected_hye_sizes,
        mask=None
    )
)]
fn compute_psi_all_loop_p_data<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p_layers: PyReadonlyArray3<'py, f64>,
    selected_hye_sizes: PyReadonlyArray1<'py, i64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let indptr_slice =
        indptr.as_slice()?;

    let log_messages_slice =
        log_messages.as_slice()?;

    let p_layers_slice =
        p_layers.as_slice()?;

    let selected_slice =
        selected_hye_sizes.as_slice()?;

    let log_shape =
        log_messages.shape();

    let p_shape =
        p_layers.shape();

    if indptr_slice.is_empty() {
        return Err(
            PyValueError::new_err(
                "indptr is empty"
            )
        );
    }

    if log_shape.len() != 2 {
        return Err(
            PyValueError::new_err(
                "log_messages must have shape (K, nnz)"
            )
        );
    }

    if p_shape.len() != 3
        || p_shape[1] != p_shape[2]
    {
        return Err(
            PyValueError::new_err(
                "p_layers must have shape (L, K, K)"
            )
        );
    }

    let n_layers =
        p_shape[0];

    let k =
        p_shape[1];

    let nnz =
        log_shape[1];

    if log_shape[0] != k {
        return Err(
            PyValueError::new_err(
                "log_messages.shape[0] must equal K"
            )
        );
    }

    if selected_slice.len()
        != n_layers
    {
        return Err(
            PyValueError::new_err(
                "selected_hye_sizes length \
                 must equal p_layers.shape[0]"
            )
        );
    }

    if indptr_slice[0] != 0 {
        return Err(
            PyValueError::new_err(
                "indptr[0] must be 0"
            )
        );
    }

    if *indptr_slice.last().unwrap()
        != nnz as i64
    {
        return Err(
            PyValueError::new_err(
                format!(
                    "indptr[-1] must equal nnz: \
                     got {}, expected {}",
                    indptr_slice
                        .last()
                        .unwrap(),
                    nnz
                )
            )
        );
    }

    let mask_slice =
        match &mask {
            Some(m) => {
                let s =
                    m.as_slice()?;

                if s.len() != nnz {
                    return Err(
                        PyValueError::new_err(
                            "mask length must equal nnz"
                        )
                    );
                }

                Some(s)
            }

            None => None,
        };

    let e =
        indptr_slice.len() - 1;

    let results:
        Vec<Result<PsiPartData, String>> =
        py.detach(|| {
            (0..e)
                .into_par_iter()
                .map(|hye| {
                    process_one_hye_loop_p_data(
                        hye as i64,
                        indptr_slice,
                        log_messages_slice,
                        nnz,
                        p_layers_slice,
                        n_layers,
                        selected_slice,
                        k,
                        mask_slice,
                    )
                })
                .collect()
        });

    let mut parts =
        Vec::with_capacity(
            results.len()
        );

    for r in results {
        parts.push(
            r.map_err(
                PyValueError::new_err
            )?
        );
    }

    concat_psi_parts_data(
        py,
        parts,
        k,
    )
}



#[pyfunction]
fn compute_psi_tilde_all_single_p<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let log_messages_slice = log_messages.as_slice()?;
    let p_slice = p.as_slice()?;

    let log_shape = log_messages.shape();
    let p_shape = p.shape();

    if indptr_slice.is_empty() {
        return Err(PyValueError::new_err("indptr is empty"));
    }

    if p_shape.len() != 2 || p_shape[0] != p_shape[1] {
        return Err(PyValueError::new_err(
            "p must have shape (K, K)",
        ));
    }

    let k = p_shape[0];
    let nnz = log_shape[1];

    if log_shape.len() != 2 || log_shape[0] != k {
        return Err(PyValueError::new_err(
            "log_messages must have shape (K, nnz)",
        ));
    }

    if indices_slice.len() != nnz {
        return Err(PyValueError::new_err(
            "indices length must equal nnz",
        ));
    }

    let e = indptr_slice.len() - 1;

    let results: Vec<Result<f64, String>> = py.detach(|| {
        (0..e)
            .into_par_iter()
            .map(|hye| {
                let start = i64_to_usize(indptr_slice[hye], "indptr")?;
                let end = i64_to_usize(indptr_slice[hye + 1], "indptr")?;
                let d = end - start;

                Ok(compute_eta_tilde_one(
                    p_slice,
                    log_messages_slice,
                    nnz,
                    k,
                    start,
                    d,
                ))
            })
            .collect()
    });

    let mut out = Vec::with_capacity(e);

    for r in results {
        out.push(r.map_err(PyValueError::new_err)?);
    }

    Ok(Array1::from_vec(out).into_pyarray(py))
}


#[pyfunction]
fn compute_psi_tilde_all_loop_p<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    log_messages: PyReadonlyArray2<'py, f64>,
    p_layers: PyReadonlyArray3<'py, f64>,
    selected_hye_sizes: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let log_messages_slice = log_messages.as_slice()?;
    let p_layers_slice = p_layers.as_slice()?;
    let selected_slice = selected_hye_sizes.as_slice()?;

    let log_shape = log_messages.shape();
    let p_shape = p_layers.shape();

    if indptr_slice.is_empty() {
        return Err(PyValueError::new_err("indptr is empty"));
    }

    if p_shape.len() != 3 || p_shape[1] != p_shape[2] {
        return Err(PyValueError::new_err(
            "p_layers must have shape (L, K, K)",
        ));
    }

    let n_layers = p_shape[0];
    let k = p_shape[1];
    let nnz = log_shape[1];

    if log_shape.len() != 2 || log_shape[0] != k {
        return Err(PyValueError::new_err(
            "log_messages must have shape (K, nnz)",
        ));
    }

    if selected_slice.len() != n_layers {
        return Err(PyValueError::new_err(
            "selected_hye_sizes length must equal p_layers.shape[0]",
        ));
    }

    if indices_slice.len() != nnz {
        return Err(PyValueError::new_err(
            "indices length must equal nnz",
        ));
    }

    let e = indptr_slice.len() - 1;

    let results: Vec<Result<f64, String>> = py.detach(|| {
        (0..e)
            .into_par_iter()
            .map(|hye| {
                let start = i64_to_usize(indptr_slice[hye], "indptr")?;
                let end = i64_to_usize(indptr_slice[hye + 1], "indptr")?;
                let d = end - start;

                let layer_idx = selected_size_index(selected_slice, d)?;

                let p = p_layer_slice(
                    p_layers_slice,
                    n_layers,
                    k,
                    layer_idx,
                )?;

                Ok(compute_eta_tilde_one(
                    p,
                    log_messages_slice,
                    nnz,
                    k,
                    start,
                    d,
                ))
            })
            .collect()
    });

    let mut out = Vec::with_capacity(e);

    for r in results {
        out.push(r.map_err(PyValueError::new_err)?);
    }

    Ok(Array1::from_vec(out).into_pyarray(py))
}


// #[pymodule]
// fn rust_hysbm<'py>(
//     _py: Python<'py>,
//     m: &Bound<'py, PyModule>,
// ) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(compute_psi_all_single_p, m)?)?;
//     m.add_function(wrap_pyfunction!(compute_psi_all_loop_p, m)?)?;
//     m.add_function(wrap_pyfunction!(compute_psi_tilde_all_single_p, m)?)?;
//     m.add_function(wrap_pyfunction!(compute_psi_tilde_all_loop_p, m)?)?;

//     Ok(())
// }


#[pymodule]
fn rust_hysbm<'py>(
    _py: Python<'py>,
    m: &Bound<'py, PyModule>,
) -> PyResult<()> {

    // ===============================
    // Legacy interface
    // ===============================
    m.add_function(
        wrap_pyfunction!(
            compute_psi_all_single_p,
            m
        )?
    )?;

    m.add_function(
        wrap_pyfunction!(
            compute_psi_all_loop_p,
            m
        )?
    )?;

    // ===============================
    // New SharedCSCMessages interface
    // ===============================
    m.add_function(
        wrap_pyfunction!(
            compute_psi_all_single_p_data,
            m
        )?
    )?;

    m.add_function(
        wrap_pyfunction!(
            compute_psi_all_loop_p_data,
            m
        )?
    )?;

    // ===============================
    // tilde
    // ===============================
    m.add_function(
        wrap_pyfunction!(
            compute_psi_tilde_all_single_p,
            m
        )?
    )?;

    m.add_function(
        wrap_pyfunction!(
            compute_psi_tilde_all_loop_p,
            m
        )?
    )?;

    Ok(())
}