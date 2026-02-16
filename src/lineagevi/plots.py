import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Optional, Tuple, Union
from anndata import AnnData
from scipy import sparse
import seaborn as sns

def top_features_table(
    adata,
    groupby_key: str,
    categories="all",
    layer: Optional[str] = None,
    n: Optional[int] = 10,
):
    """
    Return a pandas DataFrame of features (genes or gene programs) ranked by absolute mean activation,
    with per-category mean activation columns (signed).

    Parameters
    ----------
    adata : AnnData
        AnnData where rows (obs) are cells and columns (var) are features (genes or gene programs).
        Feature activations/expressions are in .X or in a specified .layers[layer].
    groupby_key : str
        Key in adata.obs with the categorical variable to group by (e.g., 'cell_type', 'cluster', 'condition').
    categories : list[str] | str, default "all"
        Which categories (levels of `groupby_key`) to include in the per-category stats.
        If "all", include all categories present in adata.obs[groupby_key].
        If a single string (not "all"), it's treated as a single category.
    layer : Optional[str], default None
        Use adata.layers[layer] instead of adata.X if provided.
    n : Optional[int], default 10
        Number of top features to return. If None, return all.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - 'feature' : feature name (from var_names)
        - 'mean_activation' : mean activation/expression across the *selected cells* (all if categories="all")
        - 'abs_mean_activation' : absolute value of mean_activation
        - Per-category columns with mean activation/expression (signed) for each category
        - 'n_cells' : number of cells used in the overall mean/statistic
    """
    # Validate obs key
    if groupby_key not in adata.obs:
        raise KeyError(f"'{groupby_key}' not found in adata.obs")

    # Resolve which categories to include in per-category stats
    obs_vals_all = adata.obs[groupby_key]
    if isinstance(categories, str) and categories != "all":
        categories = [categories]

    if categories == "all":
        overall_mask = np.ones(adata.n_obs, dtype=bool)
        include_cats = pd.Index(obs_vals_all.dropna().unique()).tolist()
    else:
        include_cats = list(categories)
        overall_mask = obs_vals_all.isin(include_cats).to_numpy()

    if not np.any(overall_mask):
        raise ValueError(
            f"No cells match {groupby_key} in {include_cats!r}."
        )

    # Get data matrix
    X_full = adata.layers[layer] if layer is not None else adata.X
    X_overall = X_full[overall_mask]

    # Compute overall mean across selected cells
    if sparse.issparse(X_overall):
        mean_act = np.asarray(X_overall.mean(axis=0)).ravel()
    else:
        mean_act = X_overall.mean(axis=0)

    abs_mean = np.abs(mean_act)

    # Sort by magnitude (descending)
    order = np.argsort(abs_mean)[::-1]
    if n is not None:
        order = order[:n]

    # Base dataframe (overall stats)
    df = pd.DataFrame({
        "feature": np.array(adata.var_names)[order],
        "mean_activation": mean_act[order],
    })

    # Per-category *mean activation* columns (signed)
    for cat in include_cats:
        cat_mask = (obs_vals_all == cat).to_numpy()
        if not np.any(cat_mask):
            df[f"{cat} mean"] = np.nan
            continue

        X_cat = X_full[cat_mask]
        if sparse.issparse(X_cat):
            cat_mean = np.asarray(X_cat.mean(axis=0)).ravel()
        else:
            cat_mean = X_cat.mean(axis=0)

        df[f"{cat} mean"] = cat_mean[order]

    # Nice index 1..N
    df.index = np.arange(1, len(df) + 1)
    return df

def plot_phase_plane(
    adata: AnnData,
    gene_name: str,
    u_scale: float = 0.1,
    s_scale: float = 0.1,
    alpha: float = 1,
    head_width: float = 0.02,
    head_length: float = 0.03,
    length_includes_head: bool = False,
    log: bool = False,
    norm_velocity: bool = True,
    filter_cells: bool = False,
    smooth_expr: bool = True,
    show_plot: bool = True,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    cluster_key: str = "clusters",
    title_fontsize: int = 16,
    axis_fontsize: int = 14,
    legend_fontsize: int = 12,
    tick_fontsize: int = 11,
    ax: Optional[plt.Axes] = None,
    # Configurable layer keys
    unspliced_key: str = "Mu",
    spliced_key: str = "Ms", 
    velocity_u_key: str = "velocity_u",
    velocity_s_key: str = "velocity",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a phase plane (spliced vs. unspliced) with RNA velocity vectors for one gene.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing layers and velocities.
        Expected layers:
        - If smooth_expr=True: "Mu" (unspliced smoothed), "Ms" (spliced smoothed)
        - Else: "unspliced", "spliced"
        Expected velocity layers:
        - "velocity_u" (unspliced velocity), "velocity" (spliced velocity)
    gene_name : str
        Gene to plot (must be in `adata.var_names`).
    u_scale, s_scale : float, default 0.1
        Multiplicative scaling of velocity arrows in unspliced/spliced directions.
        Reduce these values (e.g., 0.05) to make arrows shorter.
    alpha : float
        Arrow alpha.
    head_width, head_length : float
        Arrow head geometry for quiver.
    length_includes_head : bool
        Whether arrow length includes head length.
    log : bool
        If True, apply log1p to velocities (NOT expressions) to compress dynamic range.
    norm_velocity : bool
        If True, divide velocity components by their max absolute values (per component).
    filter_cells : bool
        If True, keep only cells with both expressions > 0 before normalization.
    smooth_expr : bool
        If True, use "Mu"/"Ms"; else "unspliced"/"spliced".
    show_plot : bool
        If True, calls plt.show() at the end.
    save_plot : bool
        If True, saves the figure to `save_path` (PNG). If `save_path` is None, uses
        "phase_plane_{gene_name}.png" in the current directory.
    save_path : Optional[str]
        Path to save PNG. Only used if `save_plot` is True.
    cluster_key : str
        Categorical obs column used for coloring points/arrows. Requires corresponding
        colors in `adata.uns[f"{cluster_key}_colors"]` when available.
    *_fontsize : int
        Font sizes for title, axes, legend, and ticks.
    ax : Optional[plt.Axes]
        If provided, plot into this axes; otherwise create a new figure/axes.
    unspliced_key : str, default "Mu"
        Key for unspliced expression in adata.layers.
    spliced_key : str, default "Ms"
        Key for spliced expression in adata.layers.
    velocity_u_key : str, default "velocity_u"
        Key for unspliced velocity in adata.layers.
    velocity_s_key : str, default "velocity"
        Key for spliced velocity in adata.layers.

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib Figure and Axes.
    """

    # --- Basic checks
    if gene_name not in adata.var_names:
        raise KeyError(f"Gene '{gene_name}' not found in adata.var_names")

    gidx = adata.var_names.get_loc(gene_name)

    expr_u_layer = unspliced_key if smooth_expr else "unspliced"
    expr_s_layer = spliced_key if smooth_expr else "spliced"

    for layer in (expr_u_layer, expr_s_layer, velocity_u_key, velocity_s_key):
        if layer not in adata.layers:
            raise KeyError(f"Required layer '{layer}' is missing from adata.layers")

    # --- Fetch raw vectors (handle sparse matrices)
    def to_dense_array(data):
        """Convert sparse matrix to dense numpy array if needed."""
        if sparse.issparse(data):
            return data.toarray()
        return np.asarray(data)
    
    u_expr = to_dense_array(adata.layers[expr_u_layer][:, gidx]).ravel()
    s_expr = to_dense_array(adata.layers[expr_s_layer][:, gidx]).ravel()
    u_vel  = to_dense_array(adata.layers[velocity_u_key][:, gidx]).ravel()
    s_vel  = to_dense_array(adata.layers[velocity_s_key][:, gidx]).ravel()

    # --- Optional filtering (before normalization)
    if filter_cells:
        valid = (u_expr > 0) & (s_expr > 0)
    else:
        # keep all finite values
        valid = np.isfinite(u_expr) & np.isfinite(s_expr) & np.isfinite(u_vel) & np.isfinite(s_vel)

    if not np.any(valid):
        raise ValueError("No cells left after filtering; check `filter_cells` or data contents.")

    u_expr = u_expr[valid]
    s_expr = s_expr[valid]
    u_vel  = u_vel[valid]
    s_vel  = s_vel[valid]

    # --- Min-max normalize expressions safely to [0, 1]
    def safe_minmax(x: np.ndarray) -> np.ndarray:
        """
        Safely normalize array to [0, 1] range.
        
        Parameters
        ----------
        x : np.ndarray
            Input array to normalize.
        
        Returns
        -------
        np.ndarray
            Normalized array in [0, 1] range, or zeros if constant.
        """
        xmin, xmax = np.min(x), np.max(x)
        if xmax == xmin:
            # constant vector -> all zeros
            return np.zeros_like(x, dtype=float)
        return (x - xmin) / (xmax - xmin)

    u_expr = safe_minmax(u_expr)
    s_expr = safe_minmax(s_expr)

    # --- Normalize velocity components (independently) by max abs value if requested
    def safe_maxabs_norm(x: np.ndarray) -> np.ndarray:
        """
        Safely normalize array by maximum absolute value.
        
        Parameters
        ----------
        x : np.ndarray
            Input array to normalize.
        
        Returns
        -------
        np.ndarray
            Array normalized by max absolute value, or unchanged if all zeros.
        """
        m = np.max(np.abs(x))
        return x / m if m > 0 else x

    if norm_velocity:
        u_vel = safe_maxabs_norm(u_vel)
        s_vel = safe_maxabs_norm(s_vel)

    # --- Optional log compression for velocities (not expressions)
    if log:
        # log1p keeps sign if we transform magnitude and reapply sign
        # (log1p on negatives is invalid). Use signed log transform.
        def signed_log1p(v: np.ndarray) -> np.ndarray:
            """
            Apply signed log1p transformation to preserve sign.
            
            Parameters
            ----------
            v : np.ndarray
                Input array to transform.
            
            Returns
            -------
            np.ndarray
                Array with signed log1p transformation applied.
            """
            return np.sign(v) * np.log1p(np.abs(v))
        u_vel = signed_log1p(u_vel)
        s_vel = signed_log1p(s_vel)

    # --- Colors per cell type
    # Default to gray if mapping not available.
    colors = np.full(u_expr.shape, "#888888", dtype=object)

    if cluster_key in adata.obs:
        # Slice obs to the valid cells
        cell_types = adata.obs[cluster_key].iloc[np.where(valid)[0]]

        # Try mapping from adata.uns[f"{cluster_key}_colors"]
        mapping_available = False
        if f"{cluster_key}_colors" in adata.uns:
            try:
                unique_ct = list(adata.obs[cluster_key].cat.categories)  # requires categorical
                ct_colors = list(adata.uns[f"{cluster_key}_colors"])
                if len(unique_ct) == len(ct_colors):
                    color_map = dict(zip(unique_ct, ct_colors))
                    colors = cell_types.map(color_map).astype(object).to_numpy()
                    mapping_available = True
            except Exception:
                mapping_available = False

        # If no mapping, try matplotlib auto color cycle per observed category
        if not mapping_available:
            observed_ct = cell_types.astype(str).to_numpy()
            uniq_obs = np.unique(observed_ct)
            prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            palette = {ct: prop_cycle[i % len(prop_cycle)] for i, ct in enumerate(uniq_obs)}
            colors = np.array([palette[c] for c in observed_ct], dtype=object)
    else:
        cell_types = None

    # --- Prepare figure/axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=120)
        created_fig = True
    else:
        fig = ax.figure

    # --- Scatter points (background) and velocity quiver
    ax.scatter(s_expr, u_expr, c=colors, s=12, alpha=0.6, linewidths=0)

    # Compute scaled velocities for axis limit calculation
    s_vel_scaled = s_vel * s_scale
    u_vel_scaled = u_vel * u_scale

    # Quiver expects U,V as x- and y- components respectively
    quiv = ax.quiver(
        s_expr, u_expr,                   # starting points (x=spliced, y=unspliced)
        s_vel_scaled, u_vel_scaled,       # components (dx, dy)
        angles="xy",
        scale_units="xy",
        scale=1.0,  # keep raw scale; we already scaled velocities
        width=0.002,
        headwidth=head_width / 0.02 * 3.0,   # normalize to quiver's units
        headlength=head_length / 0.03 * 5.0, # normalize to quiver's units
        minlength=0,
        color=colors,
        alpha=alpha,
        pivot="tail",
        clip_on=True,  # Clip arrows to axis limits
    )

    # --- Set axis limits with padding to accommodate arrows
    # Calculate maximum arrow extension to ensure arrows stay within bounds
    s_max_ext = np.max(s_expr + s_vel_scaled) if len(s_expr) > 0 else 1.0
    s_min_ext = np.min(s_expr + s_vel_scaled) if len(s_expr) > 0 else 0.0
    u_max_ext = np.max(u_expr + u_vel_scaled) if len(u_expr) > 0 else 1.0
    u_min_ext = np.min(u_expr + u_vel_scaled) if len(u_expr) > 0 else 0.0
    
    # Add padding (5% of range or fixed minimum)
    s_range = max(s_expr.max() - s_expr.min(), 0.01) if len(s_expr) > 0 else 0.01
    u_range = max(u_expr.max() - u_expr.min(), 0.01) if len(u_expr) > 0 else 0.01
    s_pad = max(s_range * 0.05, 0.02)
    u_pad = max(u_range * 0.05, 0.02)
    
    ax.set_xlim(
        max(-0.05, min(s_expr.min(), s_min_ext) - s_pad),
        min(1.05, max(s_expr.max(), s_max_ext) + s_pad)
    )
    ax.set_ylim(
        max(-0.05, min(u_expr.min(), u_min_ext) - u_pad),
        min(1.05, max(u_expr.max(), u_max_ext) + u_pad)
    )

    # --- Labels & title
    ax.set_xlabel(f"Normalized Spliced Expression of {gene_name}", fontsize=axis_fontsize)
    ax.set_ylabel(f"Normalized Unspliced Expression of {gene_name}", fontsize=axis_fontsize)
    ax.set_title(f"Expression and Velocity of {gene_name} by Cell Type", fontsize=title_fontsize)

    # --- Ticks
    ax.tick_params(labelsize=tick_fontsize)

    # --- Legend (only for categories present post-filter)
    if cell_types is not None:
        observed_ct = np.array(cell_types.astype(str))
        uniq_obs = np.unique(observed_ct)
        # Build color mapping for observed_ct from the plotted colors
        ct_to_color = {}
        # Use the first occurrence color for each category
        for ct in uniq_obs:
            idx = np.where(observed_ct == ct)[0][0]
            ct_to_color[ct] = colors[idx]

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=ct_to_color[ct], markeredgecolor="none",
                       markersize=8, label=str(ct))
            for ct in uniq_obs
        ]
        leg = ax.legend(
            handles=handles,
            title="Cell Type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=False,
        )
        # Ensure legend markers are visible
        for h in handles:
            h.set_alpha(0.9)

    fig.tight_layout()

    # --- Save / show
    if save_plot:
        path = save_path or f"phase_plane_{gene_name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")

    if show_plot and created_fig:
        plt.show()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Sequence, Union
from anndata import AnnData

GP = str
Pair = Tuple[GP, GP]

def plot_gp_phase_planes(
    adata_gp: AnnData,
    program_pairs: Union[Pair, Sequence[Pair]],
    *,
    cluster_key: str = "clusters",
    title: Optional[str] = None,
    # visibility & styling (matching plot_phase_plane)
    alpha: float = 1,
    head_width: float = 0.02,
    head_length: float = 0.03,
    length_includes_head: bool = False,
    title_fontsize: int = 16,
    axis_fontsize: int = 14,
    legend_fontsize: int = 12,
    tick_fontsize: int = 11,
    # velocity transforms
    log: bool = False,
    norm_velocity: bool = True,
    # filtering
    filter_cells: bool = False,
    # arrow sizing (autoscaled; these act as multipliers)
    arrow_multiplier: float = 1.0,
    # layout
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (5.2, 4.4),
    # save/show
    show_plot: bool = True,
    save_plot: bool = False,
    save_path: Optional[str] = None,
    # Configurable layer keys
    latent_key: str = "Ms",
    velocity_key: str = "velocity",
):
    """
    Plot 2D phase planes for GP pairs with velocity overlays.
    Each subplot uses x = spliced[gp_x], y = spliced[gp_y], arrows = velocity components (dx, dy).

    Parameters
    ----------
    adata_gp : AnnData
        Gene program AnnData object with configurable layer keys.
    program_pairs : Union[Pair, Sequence[Pair]]
        Gene program pairs to plot.
    cluster_key : str, default "clusters"
        Key for cluster coloring.
    alpha : float, default 1
        Arrow alpha (same alpha used for both points and arrows, matching plot_phase_plane).
    head_width, head_length : float
        Arrow head geometry for quiver (matching plot_phase_plane defaults).
    length_includes_head : bool, default False
        Whether arrow length includes head length.
    title_fontsize : int, default 16
        Font size for subplot titles (matching plot_phase_plane).
    axis_fontsize : int, default 14
        Font size for axis labels (matching plot_phase_plane).
    legend_fontsize : int, default 12
        Font size for legend (matching plot_phase_plane).
    tick_fontsize : int, default 11
        Font size for tick labels (matching plot_phase_plane).
    log : bool, default False
        If True, apply log1p to velocities (matching plot_phase_plane parameter name).
    norm_velocity : bool, default True
        If True, normalize velocity components.
    filter_cells : bool, default False
        If True, keep only cells with both expressions > 0 (matching plot_phase_plane parameter name).
    arrow_multiplier : float, default 1.0
        Multiplier for arrow sizing (autoscaled). 
        Reduce this value (e.g., 0.5) to make arrows shorter.
    ncols : int, default 2
        Number of columns in subplot grid.
    figsize_per_panel : Tuple[float, float], default (5.2, 4.4)
        Figure size per panel.
    show_plot : bool, default True
        Whether to show the plot (matching plot_phase_plane parameter name).
    save_plot : bool, default False
        Whether to save the plot (matching plot_phase_plane parameter name).
    save_path : Optional[str], default None
        Path to save the plot.
    latent_key : str, default "Ms"
        Key for latent representations in adata_gp.layers (use "Ms" for mean latent state).
    velocity_key : str, default "velocity"
        Key for velocities in adata_gp.layers.
    
    Colors by `adata_gp.obs[cluster_key]`, using `adata_gp.uns[f"{cluster_key}_colors"]` if available.
    """

    # --- Basic checks
    required_layers = [latent_key, velocity_key]
    for layer in required_layers:
        if layer not in adata_gp.layers:
            raise KeyError(f"Required layer '{layer}' missing from adata_gp.layers.")

    # Normalize input pairs to a list
    if isinstance(program_pairs, tuple) and len(program_pairs) == 2 and isinstance(program_pairs[0], str):
        pairs = [program_pairs]  # single pair to list
    else:
        pairs = list(program_pairs)

    # Validate GPs exist
    for gp_x, gp_y in pairs:
        if gp_x not in adata_gp.var_names:
            raise KeyError(f"Program '{gp_x}' not in adata_gp.var_names.")
        if gp_y not in adata_gp.var_names:
            raise KeyError(f"Program '{gp_y}' not in adata_gp.var_names.")

    # Check for required layers
    for layer in (latent_key, velocity_key):
        if layer not in adata_gp.layers:
            raise KeyError(f"Required layer '{layer}' is missing from adata_gp.layers")

    # Indices for fast slicing
    idx_map = {gp: adata_gp.var_names.get_loc(gp) for gp in set([g for p in pairs for g in p])}

    # Handle sparse matrices
    def to_dense_array(data):
        """Convert sparse matrix to dense numpy array if needed."""
        if sparse.issparse(data):
            return data.toarray()
        return np.asarray(data)
    
    S = to_dense_array(adata_gp.layers[latent_key])   # (cells × programs)
    V = to_dense_array(adata_gp.layers[velocity_key])  # (cells × programs)

    n_pairs = len(pairs)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_pairs / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    # Use constrained_layout to avoid oversized margins
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes_list = axes.ravel()

    # --- Colors by cluster
    if cluster_key in adata_gp.obs:
        cell_types_full = adata_gp.obs[cluster_key]
        colors_full = np.full(cell_types_full.shape[0], "#888888", dtype=object)

        mapping_available = False
        if f"{cluster_key}_colors" in adata_gp.uns:
            try:
                cats = list(cell_types_full.cat.categories)  # if categorical
                palette = list(adata_gp.uns[f"{cluster_key}_colors"])
                if len(cats) == len(palette):
                    color_map = dict(zip(cats, palette))
                    colors_full = cell_types_full.map(color_map).astype(object).to_numpy()
                    mapping_available = True
            except Exception:
                mapping_available = False

        if not mapping_available:
            observed = cell_types_full.astype(str).to_numpy()
            uniq = np.unique(observed)
            prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            pal = {ct: prop_cycle[i % len(prop_cycle)] for i, ct in enumerate(uniq)}
            colors_full = np.array([pal[c] for c in observed], dtype=object)
    else:
        cell_types_full = None
        colors_full = np.full(adata_gp.n_obs, "#888888", dtype=object)

    # --- Helpers
    def safe_minmax(x: np.ndarray) -> np.ndarray:
        """
        Safely normalize array to [0, 1] range.
        
        Parameters
        ----------
        x : np.ndarray
            Input array to normalize.
        
        Returns
        -------
        np.ndarray
            Normalized array in [0, 1] range, or zeros if constant.
        """
        xmin, xmax = np.min(x), np.max(x)
        return np.zeros_like(x) if xmax == xmin else (x - xmin) / (xmax - xmin)

    def signed_log1p(x: np.ndarray) -> np.ndarray:
        """
        Apply signed log1p transformation to preserve sign.
        
        Parameters
        ----------
        x : np.ndarray
            Input array to transform.
        
        Returns
        -------
        np.ndarray
            Array with signed log1p transformation applied.
        """
        return np.sign(x) * np.log1p(np.abs(x))

    # Precompute per-GP normalization constants for velocity if norm_velocity
    if norm_velocity:
        vel_norms = {gp: np.max(np.abs(V[:, idx_map[gp]])) for gp in idx_map}
    else:
        vel_norms = {gp: 1.0 for gp in idx_map}

    # --- Plot each pair
    for k, (gp_x, gp_y) in enumerate(pairs):
        ax = axes_list[k]
        ix, iy = idx_map[gp_x], idx_map[gp_y]

        x_raw = S[:, ix].astype(float).ravel()
        y_raw = S[:, iy].astype(float).ravel()
        dx_raw = V[:, ix].astype(float).ravel()
        dy_raw = V[:, iy].astype(float).ravel()

        # Per-subplot filtering
        if filter_cells:
            valid = (x_raw > 0) & (y_raw > 0) & np.isfinite(dx_raw) & np.isfinite(dy_raw)
        else:
            valid = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(dx_raw) & np.isfinite(dy_raw)

        if not np.any(valid):
            ax.text(0.5, 0.5, "No valid cells", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        x = safe_minmax(x_raw[valid])
        y = safe_minmax(y_raw[valid])
        dx = dx_raw[valid]
        dy = dy_raw[valid]
        cols = colors_full[valid]
        ct_sub = cell_types_full.iloc[np.where(valid)[0]] if cell_types_full is not None else None

        # Velocity transforms
        if norm_velocity:
            dx = dx / (vel_norms[gp_x] if vel_norms[gp_x] > 0 else 1.0)
            dy = dy / (vel_norms[gp_y] if vel_norms[gp_y] > 0 else 1.0)
        if log:
            dx = signed_log1p(dx)
            dy = signed_log1p(dy)

        # Auto-scale arrows so median magnitude is ~3% of axis span
        vmag = np.hypot(dx, dy)
        finite = np.isfinite(vmag) & (vmag > 0)
        if not np.any(finite):
            ax.scatter(x, y, c=cols, s=12, alpha=0.6, linewidths=0)
            ax.text(0.5, 0.5, "All velocities ~0", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            x_span = max(1e-12, np.max(x) - np.min(x))
            y_span = max(1e-12, np.max(y) - np.min(y))
            target = 0.03 * np.sqrt(x_span * y_span)
            med = np.median(vmag[finite])
            base_scale = target / max(med, 1e-12)
            dxp = dx * base_scale * arrow_multiplier
            dyp = dy * base_scale * arrow_multiplier

            ax.scatter(x, y, c=cols, s=12, alpha=0.6, linewidths=0, zorder=3)
            ax.quiver(
                x, y, dxp, dyp,
                angles="xy", scale_units="xy", scale=1.0,
                width=0.002,
                headwidth=head_width / 0.02 * 3.0,   # normalize to quiver's units (matching plot_phase_plane)
                headlength=head_length / 0.03 * 5.0, # normalize to quiver's units (matching plot_phase_plane)
                minlength=0,
                color=cols, alpha=alpha, pivot="tail", zorder=4,
                clip_on=True,  # Clip arrows to axis limits
            )
            
            # Set axis limits with padding to accommodate arrows
            x_max_ext = np.max(x + dxp) if len(x) > 0 else 1.0
            x_min_ext = np.min(x + dxp) if len(x) > 0 else 0.0
            y_max_ext = np.max(y + dyp) if len(y) > 0 else 1.0
            y_min_ext = np.min(y + dyp) if len(y) > 0 else 0.0
            
            # Add padding (5% of range or fixed minimum)
            x_range = max(x.max() - x.min(), 0.01) if len(x) > 0 else 0.01
            y_range = max(y.max() - y.min(), 0.01) if len(y) > 0 else 0.01
            x_pad = max(x_range * 0.05, 0.02)
            y_pad = max(y_range * 0.05, 0.02)
            
            ax.set_xlim(
                max(-0.05, min(x.min(), x_min_ext) - x_pad),
                min(1.05, max(x.max(), x_max_ext) + x_pad)
            )
            ax.set_ylim(
                max(-0.05, min(y.min(), y_min_ext) - y_pad),
                min(1.05, max(y.max(), y_max_ext) + y_pad)
            )

        # Labels and cosmetics
        ax.set_xlabel(f"Spliced (norm) {gp_x}", fontsize=axis_fontsize)
        ax.set_ylabel(f"Spliced (norm) {gp_y}", fontsize=axis_fontsize)
        ax.set_title(f"{gp_x} vs {gp_y}", fontsize=title_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        # Legend info per subplot for a figure-level legend
        if ct_sub is not None:
            obs_ct = np.array(ct_sub.astype(str))
            uniq = np.unique(obs_ct)
            ct_to_color = {}
            for ct in uniq:
                first_idx = np.where(obs_ct == ct)[0][0]
                ct_to_color[ct] = cols[first_idx]
            ax._ct_legend_map = ct_to_color
        else:
            ax._ct_legend_map = {}

    # Hide unused axes
    for j in range(n_pairs, len(axes_list)):
        axes_list[j].set_visible(False)

    # Global title
    if title:
        fig.suptitle(title, fontsize=title_fontsize + 2)
        # leave a bit of room for suptitle without creating huge right margin
        fig.subplots_adjust(top=0.90)

    # --- Build a compact legend that hugs the grid (TOP RIGHT, minimal gap)
    legend_map = {}
    for ax in axes_list[:n_pairs]:
        legend_map.update(getattr(ax, "_ct_legend_map", {}))

    if legend_map:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=col, markeredgecolor="none",
                       markersize=6, label=str(ct))
            for ct, col in sorted(legend_map.items(), key=lambda x: x[0])
        ]

        # Make sure axes positions are finalized
        fig.canvas.draw_idle()

        # Right edge of the visible axes in figure coordinates
        right_edge = max(a.get_position().x1 for a in axes_list[:n_pairs] if a.get_visible())
        top_edge = max(a.get_position().y1 for a in axes_list[:n_pairs] if a.get_visible())
        pad_x = 0.012  # small gap between grid and legend
        pad_y = 0.012

        # Anchor legend just outside the grid's top-right corner
        fig.legend(
            handles=handles,
            title=cluster_key,
            loc="upper left",
            bbox_to_anchor=(right_edge + pad_x, top_edge - pad_y),
            bbox_transform=fig.transFigure,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=False,
        )

        # If needed, gently shrink the right margin so legend is close but not overlapping
        fig.subplots_adjust(right=min(0.95, right_edge + 0.08))

    # Save/show
    if save_plot:
        out = save_path or "gp_phase_planes.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
    if show_plot:
        plt.show()

    return fig, axes

def _is_differential_format(scores):
    """True if scores are differential() output: dict of DataFrames with 'padj' column."""
    val = next(iter(scores.values()), None)
    return isinstance(val, pd.DataFrame) and "padj" in val.columns


def plot_abs_bfs_key(scores, terms, key, n_points=30, lim_val=2.3, fontsize=8, scale_y=2, yt_step=0.3,
                    title=None, ax=None):
    """
    Plot top features by significance for one group.
    Supports (1) legacy bf format: scores[key]['bf'] array, terms array;
    (2) differential format: scores[key] is DataFrame with 'padj' and index = feature names.
    For differential format, y-axis is -log10(padj) (capped); lim_val is -log10(p-threshold).
    """
    txt_args = dict(
        rotation='vertical',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontsize=fontsize,
    )

    ax = ax if ax is not None else plt.axes()
    ax.grid(False)

    if isinstance(scores[key], pd.DataFrame) and "padj" in scores[key].columns:
        df = scores[key]
        padj = df["padj"].values
        padj = np.where(padj <= 0, np.nan, padj)  # avoid log(0)
        y_vals = -np.log10(padj)
        y_vals = np.clip(y_vals, 0, 20)  # cap inf
        terms_arr = np.asarray(df.index)
        # already sorted by padj ascending, so first n_points are top
        n_show = min(n_points, len(terms_arr))
        y_vals = y_vals[:n_show]
        terms_arr = terms_arr[:n_show]
        y_label = "-log10(adj p-value)"
    else:
        bfs = np.abs(scores[key]["bf"])
        srt = np.argsort(bfs)[::-1][:n_points]
        y_vals = bfs[srt]
        terms_arr = terms[srt]
        n_show = len(y_vals)
        y_label = "Absolute log bayes factors"

    top = np.nanmax(y_vals) if len(y_vals) else 1.0
    if not np.isfinite(top) or top <= 0:
        top = 1.0

    y_max = top * scale_y
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))

    ax.set_xlim(0.1, n_show + 0.9)
    xt = np.arange(0, n_show + 1, 5)
    if len(xt) > 0:
        xt[0] = 1
    ax.set_xticks(xt)

    for i, (y, term) in enumerate(zip(y_vals, terms_arr)):
        if np.isfinite(y):
            ax.text(i + 1, y, str(term), **txt_args)

    ax.axhline(y=lim_val, color='red', linestyle='--', label='')

    ax.set_xlabel("Rank")
    ax.set_ylabel(y_label)
    ax.set_title(key if title is None else title)

    return ax.figure

def plot_differential(
    adata,
    scores_key: str,
    *,
    terms: Union[str, list, None] = "terms",
    keys=None,
    n_cols=3,
    n_points=30,
    lim_val=1.3,
    figsize=None,
    dpi=None,
    fontsize=8,
    scale_y=2,
    yt_step=0.3,
    title=None,
    **kwargs,
):
    """\
    Plot top features by significance per group (ranked bar chart).

    Use with differential() results: store the returned dict in adata.uns[scores_key],
    then call this to plot -log10(adj p-value) vs rank for each group.

    Parameters
    ----------
    adata : AnnData
        AnnData with adata.uns[scores_key] = dict of DataFrames (from differential()).
    scores_key : str
        Key in adata.uns where the differential results are stored (e.g. 'differential_latent').
    terms : str, list, or None
        For legacy bf format only: key in adata.uns for feature names, or list. Ignored for differential format.
    keys : list, optional
        Subset of groups to plot. Default: all groups.
    n_cols : int
        Number of columns in the subplot grid.
    n_points : int
        Number of top features to show per group.
    lim_val : float
        Horizontal line threshold; for differential format this is -log10(p), e.g. 1.3 for p=0.05.
    """
    from itertools import product

    scores = adata.uns[scores_key]

    if _is_differential_format(scores):
        terms = None
    else:
        if terms is None:
            terms = "terms"
        if isinstance(terms, str):
            terms = np.asarray(adata.uns[terms])
        else:
            terms = np.asarray(terms)
        if len(terms) != len(next(iter(scores.values()))["bf"]):
            raise ValueError("Incorrect length of terms.")

    if keys is None:
        keys = list(scores.keys())

    if len(keys) == 1:
        keys = keys[0]

    kwargs.setdefault("n_points", n_points)
    kwargs.setdefault("lim_val", lim_val)
    kwargs.setdefault("fontsize", fontsize)
    kwargs.setdefault("scale_y", scale_y)
    kwargs.setdefault("yt_step", yt_step)
    kwargs.setdefault("title", title)

    if isinstance(keys, str):
        return plot_abs_bfs_key(scores, terms, keys, **kwargs)

    n_keys = len(keys)

    if n_keys <= n_cols:
        n_cols = n_keys
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_keys / n_cols))

    if figsize is None:
        figsize = (4 * n_cols, 8 * n_rows)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize, dpi=dpi,
        gridspec_kw={"wspace": 0.4},
    )

    for key, ix in zip(keys, product(range(n_rows), range(n_cols))):
        if n_rows == 1:
            ix = ix[1]
        elif n_cols == 1:
            ix = ix[0]
        plot_abs_bfs_key(scores, terms, key, ax=axs[ix], **kwargs)

    n_inactive = n_rows * n_cols - n_keys
    if n_inactive > 0:
        for i in range(n_inactive):
            axs[n_rows - 1, -(i + 1)].axis("off")

    plt.close(fig)
    return fig


def plot_abs_bfs(
    adata,
    scores_key="bf_scores",
    terms: Union[str, list, None] = "terms",
    keys=None,
    n_cols=3,
    figsize=None,
    dpi=None,
    **kwargs,
):
    """\
    Legacy alias for plot_differential. Prefer plot_differential() for differential() results.
    """
    return plot_differential(
        adata,
        scores_key=scores_key,
        terms=terms,
        keys=keys,
        n_cols=n_cols,
        figsize=figsize,
        dpi=dpi,
        **kwargs,
    )


def _to_list(x):
    """Return [x] if x is not a list, else x."""
    return x if isinstance(x, list) else [x]


def heatmap(
    adata: AnnData,
    var_names: Union[str, list],
    sortby: str = "latent_time",
    layer: Optional[str] = "Ms",
    color_map: str = "viridis",
    col_color: Optional[Union[str, list]] = None,
    palette: Union[str, list] = "viridis",
    n_convolve: Optional[int] = 30,
    standard_scale: Optional[int] = 0,
    sort: bool = True,
    colorbar: Optional[bool] = None,
    col_cluster: bool = False,
    row_cluster: bool = False,
    figsize: Tuple[float, float] = (8, 4),
    show: Optional[bool] = None,
    save: Optional[Union[bool, str]] = None,
    **kwargs,
):
    """Plot time series for genes (or gene programs) as heatmap.

    Similar to scvelo.pl.plot_heatmap: cells are ordered by sortby (e.g. latent time),
    optional smoothing along the x-axis, optional row sort by peak position.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    var_names : str or list of str
        Names of variables (genes or gene program names) to plot.
    sortby : str, default 'latent_time'
        Key in adata.obs to order cells by (e.g. latent time).
    layer : str or None, default 'Ms'
        Layer key for expression. If None, use adata.X.
    color_map : str, default 'viridis'
        Matplotlib colormap name for the heatmap.
    col_color : str or list of str, optional
        Keys in adata.obs to color the columns (cells) by.
    palette : str or list, default 'viridis'
        Palette for categorical column annotations.
    n_convolve : int or None, default 30
        If set, smooth data along the x-axis with a rolling mean of this size.
    standard_scale : int or None, default 0
        0 = scale each row (gene) to [0,1]; 1 = scale each column (cell).
    sort : bool, default True
        If True, sort rows (genes) by position of maximum value (peak order).
    colorbar : bool or None
        Whether to show the heatmap colorbar.
    col_cluster : bool, default False
        If True, cluster columns (cells).
    row_cluster : bool, default False
        If True, cluster rows (genes).
    figsize : tuple, default (8, 4)
        Figure size (width, height).
    show : bool or None
        If False, return the clustermap object without showing. If True or None, show.
    save : bool or str or None
        If True or a path, save the figure.
    **kwargs
        Passed to seaborn.clustermap (e.g. yticklabels=True to show all gene names).

    Returns
    -------
    seaborn.matrix.ClusterMap or None
        If show is False, returns the clustermap; otherwise shows and returns None.
    """
    var_names = _to_list(var_names)
    var_names = [n for n in var_names if n in adata.var_names]

    if len(var_names) == 0:
        raise ValueError("No var_names found in adata.var_names")

    tkey = sortby
    xkey = layer if layer is not None else "X"
    time = adata.obs[tkey].values.copy()
    valid = np.isfinite(time)
    if not np.any(valid):
        raise ValueError(f"All values in adata.obs[{tkey!r}] are non-finite.")
    time = time[valid]

    if layer and layer in adata.layers:
        X = adata[valid, var_names].layers[layer]
    else:
        X = adata[valid, var_names].X

    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    order = np.argsort(time)
    df = pd.DataFrame(X[order], columns=var_names)

    if n_convolve is not None and n_convolve > 1:
        weights = np.ones(n_convolve) / n_convolve
        for gene in var_names:
            try:
                df[gene] = np.convolve(df[gene].values.astype(float), weights, mode="same")
            except (ValueError, TypeError):
                pass

    if sort:
        try:
            max_sort = np.argsort(np.argmax(df.values, axis=0))
            df = pd.DataFrame(df.values[:, max_sort], columns=df.columns[max_sort])
        except (ValueError, np.AxisError):
            pass

    col_color_arrays = None
    if col_color is not None:
        col_color_list = _to_list(col_color)
        col_color_arrays = []
        for col in col_color_list:
            if col not in adata.obs.columns:
                continue
            vals = adata.obs[col].values[valid][order]
            if pd.api.types.is_categorical_dtype(adata.obs[col]):
                cats = adata.obs[col].cat.categories
                n_cat = len(cats)
                # Use adata.uns[f"{col}_colors"] when available (scanpy convention)
                palette_colors = None
                colors_key = f"{col}_colors"
                if colors_key in adata.uns:
                    stored = adata.uns[colors_key]
                    if hasattr(stored, "__len__") and len(stored) >= n_cat:
                        from matplotlib.colors import to_rgba
                        palette_colors = []
                        for c in list(stored)[:n_cat]:
                            try:
                                rgba = to_rgba(c)
                                palette_colors.append(rgba[:3])  # rgb for seaborn
                            except (ValueError, TypeError):
                                break
                        if len(palette_colors) != n_cat:
                            palette_colors = None
                if palette_colors is None:
                    palette_colors = (
                        sns.color_palette(palette, n_cat)
                        if isinstance(palette, str)
                        else list(palette)[:n_cat]
                    )
                color_map_cat = dict(zip(cats, palette_colors))
                col_color_arrays.append([color_map_cat.get(v, (0.5, 0.5, 0.5)) for v in vals])
            else:
                v = np.asarray(vals, dtype=float)
                v = np.ma.masked_invalid(v)
                if np.ma.is_masked(v) or not np.any(np.isfinite(v)):
                    col_color_arrays.append([(0.8, 0.8, 0.8)] * len(vals))
                else:
                    norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-8)
                    cmap = plt.get_cmap(palette if isinstance(palette, str) else "viridis")
                    col_color_arrays.append([cmap(x) for x in norm])
        if len(col_color_arrays) == 0:
            col_color_arrays = None

    if "dendrogram_ratio" not in kwargs:
        kwargs["dendrogram_ratio"] = (
            0.1 if row_cluster else 0,
            0.2 if col_cluster else 0,
        )
    if colorbar is False or (colorbar is None and col_color_arrays is not None):
        kwargs["cbar_pos"] = None
    else:
        kwargs.setdefault("cbar_pos", (0.02, 0.8, 0.03, 0.18))

    kwargs.update(
        {
            "col_colors": col_color_arrays,
            "col_cluster": col_cluster,
            "row_cluster": row_cluster,
            "cmap": color_map,
            "xticklabels": False,
            "standard_scale": standard_scale,
            "figsize": figsize,
        }
    )

    try:
        cm = sns.clustermap(df.T, **kwargs)
    except (TypeError, ImportError):
        kwargs.pop("dendrogram_ratio", None)
        kwargs.pop("cbar_pos", None)
        cm = sns.clustermap(df.T, **kwargs)

    if save not in (None, False):
        path = save if isinstance(save, str) else "heatmap.pdf"
        cm.savefig(path, dpi=150, bbox_inches="tight")

    if show is False:
        plt.close(cm.fig)
        return cm
    plt.show()
    return None
