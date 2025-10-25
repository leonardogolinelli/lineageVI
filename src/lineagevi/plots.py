import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from anndata import AnnData
from scipy import sparse

def top_gps_table(
    gp_adata,
    celltype_key: str,
    categories="all",
    layer: str | None = None,
    n: int | None = 10,
):
    """
    Return a pandas DataFrame of gene programs ranked by absolute mean activation,
    with per-category mean activation columns (signed).

    Parameters
    ----------
    gp_adata : AnnData
        AnnData where rows (obs) are cells and columns (var) are gene programs.
        GP activations are in .X or in a specified .layers[layer].
    celltype_key : str
        Key in gp_adata.obs with the categorical variable to filter on (e.g., 'cell_type').
    categories : list[str] | str, default "all"
        Which categories (levels of `celltype_key`) to include in the per-category stats.
        If "all", include all categories present in gp_adata.obs[celltype_key].
        If a single string (not "all"), it's treated as a single category.
    layer : str | None, default None
        Use gp_adata.layers[layer] instead of gp_adata.X if provided.
    n : int | None, default 10
        Number of top programs to return. If None, return all.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - 'gp' : gene program name (from var_names)
        - 'mean_activation' : mean activation across the *selected cells* (all if categories="all")
        - 'abs_mean_activation' : absolute value of mean_activation
        - Per-category columns with mean activation (signed) for each category
        - 'n_cells' : number of cells used in the overall mean/statistic
    """
    # Validate obs key
    if celltype_key not in gp_adata.obs:
        raise KeyError(f"'{celltype_key}' not found in gp_adata.obs")

    # Resolve which categories to include in per-category stats
    obs_vals_all = gp_adata.obs[celltype_key]
    if isinstance(categories, str) and categories != "all":
        categories = [categories]

    if categories == "all":
        overall_mask = np.ones(gp_adata.n_obs, dtype=bool)
        include_cats = pd.Index(obs_vals_all.dropna().unique()).tolist()
    else:
        include_cats = list(categories)
        overall_mask = obs_vals_all.isin(include_cats).to_numpy()

    if not np.any(overall_mask):
        raise ValueError(
            f"No cells match {celltype_key} in {include_cats!r}."
        )

    # Get data matrix
    X_full = gp_adata.layers[layer] if layer is not None else gp_adata.X
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
        "gp": np.array(gp_adata.var_names)[order],
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
    cell_type_key: str = "clusters",
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
    u_scale, s_scale : float
        Multiplicative scaling of velocity arrows in unspliced/spliced directions.
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
    cell_type_key : str
        Categorical obs column used for coloring points/arrows. Requires corresponding
        colors in `adata.uns[f"{cell_type_key}_colors"]` when available.
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

    # --- Fetch raw vectors
    u_expr = np.asarray(adata.layers[expr_u_layer][:, gidx]).ravel()
    s_expr = np.asarray(adata.layers[expr_s_layer][:, gidx]).ravel()
    u_vel  = np.asarray(adata.layers[velocity_u_key][:, gidx]).ravel()
    s_vel  = np.asarray(adata.layers[velocity_s_key][:, gidx]).ravel()

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

    if cell_type_key in adata.obs:
        # Slice obs to the valid cells
        cell_types = adata.obs[cell_type_key].iloc[np.where(valid)[0]]

        # Try mapping from adata.uns[f"{cell_type_key}_colors"]
        mapping_available = False
        if f"{cell_type_key}_colors" in adata.uns:
            try:
                unique_ct = list(adata.obs[cell_type_key].cat.categories)  # requires categorical
                ct_colors = list(adata.uns[f"{cell_type_key}_colors"])
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

    # Quiver expects U,V as x- and y- components respectively
    quiv = ax.quiver(
        s_expr, u_expr,                   # starting points (x=spliced, y=unspliced)
        s_vel * s_scale, u_vel * u_scale,  # components (dx, dy)
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
    cell_type_key: str = "clusters",
    title: Optional[str] = None,
    # visibility & styling
    point_size: float = 10.0,
    alpha_points: float = 0.6,
    alpha_arrows: float = 0.7,
    tick_fontsize: int = 10,
    axis_fontsize: int = 12,
    title_fontsize: int = 14,
    legend_fontsize: int = 10,
    # velocity transforms
    norm_velocity: bool = True,
    log_velocity: bool = False,
    # filtering
    filter_cells_positive: bool = False,  # keep only cells with both gp spliced > 0 for each pair
    # arrow sizing (autoscaled; these act as multipliers)
    arrow_multiplier: float = 1.0,
    # layout
    ncols: int = 2,
    figsize_per_panel: Tuple[float, float] = (5.2, 4.4),
    # save/show
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    # Configurable layer keys
    latent_key: str = "z",
    velocity_key: str = "velocity_gp",
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
    cell_type_key : str, default "clusters"
        Key for cell type coloring.
    latent_key : str, default "z"
        Key for latent representations in adata_gp.layers.
    velocity_key : str, default "velocity_gp"
        Key for velocities in adata_gp.layers.
    Other parameters control plotting appearance and behavior.
    
    Colors by `adata_gp.obs[cell_type_key]`, using `adata_gp.uns[f"{cell_type_key}_colors"]` if available.
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

    # Indices for fast slicing
    idx_map = {gp: adata_gp.var_names.get_loc(gp) for gp in set([g for p in pairs for g in p])}

    S = np.asarray(adata_gp.layers[latent_key])   # (cells × programs)
    V = np.asarray(adata_gp.layers[velocity_key])  # (cells × programs)

    n_pairs = len(pairs)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_pairs / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)

    # Use constrained_layout to avoid oversized margins
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes_list = axes.ravel()

    # --- Colors by cell type
    if cell_type_key in adata_gp.obs:
        cell_types_full = adata_gp.obs[cell_type_key]
        colors_full = np.full(cell_types_full.shape[0], "#888888", dtype=object)

        mapping_available = False
        if f"{cell_type_key}_colors" in adata_gp.uns:
            try:
                cats = list(cell_types_full.cat.categories)  # if categorical
                palette = list(adata_gp.uns[f"{cell_type_key}_colors"])
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
        if filter_cells_positive:
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
        if log_velocity:
            dx = signed_log1p(dx)
            dy = signed_log1p(dy)

        # Auto-scale arrows so median magnitude is ~3% of axis span
        vmag = np.hypot(dx, dy)
        finite = np.isfinite(vmag) & (vmag > 0)
        if not np.any(finite):
            ax.scatter(x, y, c=cols, s=point_size, alpha=alpha_points, linewidths=0)
            ax.text(0.5, 0.5, "All velocities ~0", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            x_span = max(1e-12, np.max(x) - np.min(x))
            y_span = max(1e-12, np.max(y) - np.min(y))
            target = 0.03 * np.sqrt(x_span * y_span)
            med = np.median(vmag[finite])
            base_scale = target / max(med, 1e-12)
            dxp = dx * base_scale * arrow_multiplier
            dyp = dy * base_scale * arrow_multiplier

            ax.scatter(x, y, c=cols, s=point_size, alpha=alpha_points, linewidths=0, zorder=3)
            ax.quiver(
                x, y, dxp, dyp,
                angles="xy", scale_units="xy", scale=1.0,
                width=0.004, minlength=1.5, headwidth=4.5, headlength=6.5,
                color=cols, alpha=alpha_arrows, pivot="tail", zorder=4
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
            title=cell_type_key,
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
    if save:
        out = save_path or "gp_phase_planes.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, axes

def plot_abs_bfs_key(scores, terms, key, n_points=30, lim_val=2.3, fontsize=8, scale_y=2, yt_step=0.3,
                    title=None, ax=None):
    txt_args = dict(
        rotation='vertical',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontsize=fontsize,
    )

    ax = ax if ax is not None else plt.axes()
    ax.grid(False)

    bfs = np.abs(scores[key]['bf'])
    srt = np.argsort(bfs)[::-1][:n_points]
    top = bfs.max()

    ax.set_ylim(top=top * scale_y)
    yt = np.arange(0, top * 1.1, yt_step)
    ax.set_yticks(yt)

    ax.set_xlim(0.1, n_points + 0.9)
    xt = np.arange(0, n_points + 1, 5)
    xt[0] = 1
    ax.set_xticks(xt)

    for i, (bf, term) in enumerate(zip(bfs[srt], terms[srt])):
        ax.text(i+1, bf, term, **txt_args)

    ax.axhline(y=lim_val, color='red', linestyle='--', label='')

    ax.set_xlabel("Rank")
    ax.set_ylabel("Absolute log bayes factors")
    ax.set_title(key if title is None else title)

    return ax.figure

def plot_abs_bfs(
    adata,
    scores_key="bf_scores",
    terms: Union[str, list] = "terms",
    keys=None,
    n_cols=3,
    figsize=None,   # 👈 new
    dpi=None,       # 👈 new
    **kwargs,
):
    """\
    Plot the absolute bayes scores rankings.
    """

    from itertools import product

    scores = adata.uns[scores_key]

    if isinstance(terms, str):
        terms = np.asarray(adata.uns[terms])
    else:
        terms = np.asarray(terms)

    if len(terms) != len(next(iter(scores.values()))["bf"]):
        raise ValueError('Incorrect length of terms.')

    if keys is None:
        keys = list(scores.keys())

    if len(keys) == 1:
        keys = keys[0]

    if isinstance(keys, str):
        return plot_abs_bfs_key(scores, terms, keys, **kwargs)

    n_keys = len(keys)

    if n_keys <= n_cols:
        n_cols = n_keys
        n_rows = 1
    else:
        n_rows = int(np.ceil(n_keys / n_cols))

    # default scaling if figsize not provided
    if figsize is None:
        figsize = (4 * n_cols, 8 * n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)

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

    plt.close(fig)  # prevent auto-display
    return fig
