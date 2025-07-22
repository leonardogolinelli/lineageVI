import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union

def add_annotations(adata, files, min_genes=0, max_genes=None, varm_key='I', uns_key='terms',
                clean=True, genes_use_upper=True):
    """\
    Add annotations to an AnnData object from files.

    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Paths to text files with annotations. The function considers rows to be gene sets
        with name of a gene set in the first column followed by names of genes.
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
    uns_key
        Sore gene sets' names in `adata.uns[uns_key]`.
    clean
        If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
        and cuts the name to the first thirty symbols.
    genes_use_upper
        if 'True', converts genes' names from files and adata to uppercase for comparison.
    """
    
    files = [files] if isinstance(files, str) else files
    annot = []

    for file in files:
        with open(file) as f:
            p_f = [l.upper() for l in f] if genes_use_upper else f
            terms = [l.strip('\n').split() for l in p_f]

        if clean:
            terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
        annot+=terms

    var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]


def plot_top_gps_activation(adata, latent_key="X_cvae", terms_key="terms", n=10):
    latent_means = adata.obsm[latent_key].mean(0)
    sorted_idxs = np.argsort(np.abs(latent_means))[::-1][:n]

    # Retrieve corresponding term names
    gp_names = np.array(adata.uns[terms_key])[sorted_idxs]
    activations = latent_means[sorted_idxs]
    colors = ['blue' if val > 0 else 'red' for val in activations]

    plt.figure(figsize=(10, 6))
    plt.barh(gp_names[::-1], activations[::-1], color=colors[::-1])  # flip for top-down
    plt.xlabel('Activation')
    plt.title('Top {} Gene Programs by Absolute Activation'.format(n))
    plt.tight_layout()
    plt.show()

def plot_top_gps_per_celltype(adata,
                            groupby="cell_type", 
                            latent_key="X_cvae",
                            term_key="terms",
                            n=10,
                            target_group=None):
    """
    Plot barplots of top absolute gene program activations for a specific or all cell types.
    
    Parameters:
    - adata: AnnData object
    - groupby: Column in adata.obs to group cells by (e.g., "cell_type")
    - latent_key: Key in adata.obsm where gene program activations are stored
    - term_key: Key in adata.uns with gene program names
    - n: Number of top absolute gene programs to show per group
    - target_group: If specified, only plot this specific group (e.g., a single cell type)
    """

    groups = [target_group] if target_group else adata.obs[groupby].unique()
    gp_names = np.array(adata.uns[term_key])

    for group in groups:
        idx = adata.obs[groupby] == group
        group_activations = adata.obsm[latent_key][idx].mean(axis=0)
        
        top_idx = np.argsort(np.abs(group_activations))[::-1][:n]
        top_gps = gp_names[top_idx]
        top_vals = group_activations[top_idx]
        colors = ['blue' if val > 0 else 'red' for val in top_vals]

        plt.figure(figsize=(10, 6))
        plt.barh(top_gps[::-1], top_vals[::-1], color=colors[::-1])  # reverse for descending top-to-bottom
        plt.xlabel("Activation")
        plt.ylabel("Gene Programs")
        plt.title(f"Top {n} Absolute Gene Programs in {group}")
        plt.tight_layout()
        plt.show()

def scatter_terms(adata,
                term_x, 
                term_y, 
                latent_key="X_cvae", 
                term_key="terms", 
                groupby="clusters",
                s=10,
                alpha=0.8):
    """
    Scatter plot of cells in space of two gene programs using matplotlib,
    respecting Scanpy's color-to-group mapping.
    """
    
    gp_names = list(adata.uns[term_key])
    try:
        idx_x = gp_names.index(term_x)
        idx_y = gp_names.index(term_y)
    except ValueError as e:
        raise ValueError(f"Term not found in {term_key}: {e}")

    X = adata.obsm[latent_key][:, [idx_x, idx_y]]
    groups = adata.obs[groupby]

    # Use categorical order if available
    if pd.api.types.is_categorical_dtype(groups):
        group_order = list(groups.cat.categories)
    else:
        group_order = sorted(groups.unique())

    if f"{groupby}_colors" in adata.uns:
        color_list = adata.uns[f"{groupby}_colors"]
        if len(color_list) != len(group_order):
            raise ValueError("Mismatch between number of colors and number of categories.")
        color_map = dict(zip(group_order, color_list))
    else:
        raise ValueError(f"Expected colors in adata.uns['{groupby}_colors']")

    plt.figure(figsize=(8, 6))
    for group in group_order:
        idx = groups == group
        plt.scatter(X[idx, 0], X[idx, 1], 
                    c=color_map[group], 
                    label=group, 
                    s=s, alpha=alpha, edgecolors='none')

    plt.xlabel(term_x)
    plt.ylabel(term_y)
    plt.title(f"{term_x} vs {term_y}")
    plt.legend(title=groupby, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

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

def plot_abs_bfs(adata, scores_key="bf_scores", terms: Union[str, list]="terms",
                keys=None, n_cols=3, **kwargs):
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

    fig, axs = plt.subplots(n_rows, n_cols)
    for key, ix in zip(keys, product(range(n_rows), range(n_cols))):
        if n_rows == 1:
            ix = ix[1]
        elif n_cols == 1:
            ix = ix[0]
        plot_abs_bfs_key(scores, terms, key, ax=axs[ix], **kwargs)

    n_inactive = n_rows * n_cols - n_keys
    if n_inactive > 0:
        for i in range(n_inactive):
            axs[n_rows-1, -(i+1)].axis('off')

    return fig

def plot_phase_plane(adata, gene_name, u_scale=.01, s_scale=0.01, alpha=0.5, head_width=0.02, head_length=0.03, length_includes_head=False, log=False,
                        norm_velocity=True, filter_cells=False, smooth_expr=True, show_plot=True, save_plot=True, save_path=".",
                        cell_type_key="clusters",title_fontsize=16, axis_fontsize=14, legend_fontsize=14, tick_fontsize=12):

    if smooth_expr:
        unspliced_expression = adata.layers["Mu"][:, adata.var_names.get_loc(gene_name)].flatten() 
        spliced_expression = adata.layers["Ms"][:, adata.var_names.get_loc(gene_name)].flatten() 
    else:
        unspliced_expression = adata.layers["unspliced"][:, adata.var_names.get_loc(gene_name)].flatten()
        spliced_expression = adata.layers["spliced"][:, adata.var_names.get_loc(gene_name)].flatten()

    # Normalize the expression data
    unspliced_expression_min, unspliced_expression_max = np.min(unspliced_expression), np.max(unspliced_expression)
    spliced_expression_min, spliced_expression_max = np.min(spliced_expression), np.max(spliced_expression)

    # Min-Max normalization
    unspliced_expression = (unspliced_expression - unspliced_expression_min) / (unspliced_expression_max - unspliced_expression_min)
    spliced_expression = (spliced_expression - spliced_expression_min) / (spliced_expression_max - spliced_expression_min)

    # Extract the velocity data
    unspliced_velocity = adata.layers['velocity_u'][:, adata.var_names.get_loc(gene_name)].flatten()
    spliced_velocity = adata.layers['velocity'][:, adata.var_names.get_loc(gene_name)].flatten()

    def custom_scale(data):
        max_abs_value = np.max(np.abs(data))  # Find the maximum absolute value
        scaled_data = data / max_abs_value  # Scale by the maximum absolute value
        return scaled_data

    if norm_velocity:
        unspliced_velocity = custom_scale(unspliced_velocity)
        spliced_velocity = custom_scale(spliced_velocity)


    # Apply any desired transformations (e.g., log) here
    if log:
        # Apply log transformation safely, ensuring no log(0)
        unspliced_velocity = np.log1p(unspliced_velocity)
        spliced_velocity = np.log1p(spliced_velocity)

    # Generate boolean masks for conditions and apply them
    if filter_cells:
        valid_idx = (unspliced_expression > 0) & (spliced_expression > 0)
    else:
        valid_idx = (unspliced_expression >= 0) & (spliced_expression >= 0)

    # Filter data based on valid_idx
    unspliced_expression_filtered = unspliced_expression[valid_idx]
    spliced_expression_filtered = spliced_expression[valid_idx]
    unspliced_velocity_filtered = unspliced_velocity[valid_idx]
    spliced_velocity_filtered = spliced_velocity[valid_idx]

    # Also filter cell type information to match the filtered expressions
    # First, get unique cell types and their corresponding colors
    unique_cell_types = adata.obs[cell_type_key].cat.categories
    celltype_colors = adata.uns[f"{cell_type_key}_colors"]
    
    # Create a mapping of cell type to its color
    celltype_to_color = dict(zip(unique_cell_types, celltype_colors))

    # Filter cell types from the data to get a list of colors for the filtered data points
    cell_types_filtered = adata.obs[cell_type_key][valid_idx]
    colors = cell_types_filtered.map(celltype_to_color).to_numpy()
    plt.figure(figsize=(9, 6.5), dpi=100)
    # Lower dpi here if the file is still too large    scatter = plt.scatter(unspliced_expression_filtered, spliced_expression_filtered, c=colors, alpha=0.6)

    """# Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            unspliced_expression_filtered[i], spliced_expression_filtered[i], 
            unspliced_velocity_filtered[i] * u_scale, spliced_velocity_filtered[i] * s_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )"""

    # Plot velocity vectors
    for i in range(len(unspliced_expression_filtered)):
        cell_type_index = np.where(unique_cell_types == cell_types_filtered[i])[0][0]
        arrow_color = celltype_to_color[cell_types_filtered[i]]  # Use the color corresponding to the cell type
        plt.arrow(
            spliced_expression_filtered[i], unspliced_expression_filtered[i], 
            spliced_velocity_filtered[i] * s_scale, unspliced_velocity_filtered[i] * u_scale, 
            color=arrow_color, alpha=alpha, head_width=head_width, head_length=head_length, length_includes_head=length_includes_head
        )

    plt.ylabel(f'Normalized Unspliced Expression of {gene_name}', fontsize=axis_fontsize)
    plt.xlabel(f'Normalized Spliced Expression of {gene_name}', fontsize=axis_fontsize)
    plt.title(f'Expression and Velocity of {gene_name} by Cell Type', fontsize=title_fontsize)

    # Increase the font size of the tick labels
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Create a legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=celltype_to_color[celltype], markersize=10, label=celltype) 
            for celltype in unique_cell_types]
    plt.legend(handles=patches, title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize, title_fontsize=title_fontsize)

    plt.show()

    '''if save_plot:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Plot saved to {save_path}")'''



