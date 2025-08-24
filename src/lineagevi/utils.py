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

def forward_encoder(model, x):
    z, mean, logvar = model.encoder(x)
    return z, mean, logvar

#z, mean, logvar = forward_encoder(model, x)

'''def sample_z(model, x, n_samples):
    _, mean, logvar = forward_encoder(x)
    for sample in range(n_samples):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std'''

def forward_gene_decoder(model, z):
    x_rec = model.gene_decoder(z)
    return x_rec

#x_rec = forward_gene_decoder(model, z)

def forward_velocity_decoder(model, z, x):
    velocity, velocity_u = model.velocity_decoder(z, x)
    return velocity, velocity_u

#velocity, velocity_u = forward_velocity_decoder(model, z, x)

def latent_enrich(
        adata,
        model,
        groups,
        comparison='rest',
        n_sample=5000,
        use_directions=False,
        directions_key='directions',
        select_terms=None,
        exact=True,
        key_added='bf_scores'
    ):
        """Gene set enrichment test for the latent space. Test the hypothesis that latent scores
           for each term in one group (z_1) is bigger than in the other group (z_2).

           Puts results to `adata.uns[key_added]`. Results are a dictionary with
           `p_h0` - probability that z_1 > z_2, `p_h1 = 1-p_h0` and `bf` - bayes factors equal to `log(p_h0/p_h1)`.

           Parameters
           ----------
           groups: String or Dict
                A string with the key in `adata.obs` to look for categories or a dictionary
                with categories as keys and lists of cell names as values.
           comparison: String
                The category name to compare against. If 'rest', then compares each category against all others.
           n_sample: Integer
                Number of random samples to draw for each category.
           use_directions: Boolean
                If 'True', multiplies the latent scores by directions in `adata`.
           directions_key: String
                The key in `adata.uns` for directions.
           select_terms: Array
                If not 'None', then an index of terms to select for the test. Only does the test
                for these terms.
           adata: AnnData
                An AnnData object to use. If 'None', uses `self.adata`.
           exact: Boolean
                Use exact probabilities for comparisons.
           key_added: String
                key of adata.uns where to put the results of the test.
        """

        if isinstance(groups, str):
            cats_col = adata.obs[groups]
            cats = cats_col.unique()
        elif isinstance(groups, dict):
            cats = []
            all_cells = []
            for group, cells in groups.items():
                cats.append(group)
                all_cells += cells
            adata = adata[all_cells]
            cats_col = pd.Series(index=adata.obs_names, dtype=str)
            for group, cells in groups.items():
                cats_col[cells] = group
        else:
            raise ValueError("groups should be a string or a dict.")

        if comparison != "rest" and isinstance(comparison, str):
            comparison = [comparison]

        if comparison != "rest" and not set(comparison).issubset(cats):
            raise ValueError("comparison should be 'rest' or among the passed groups")

        scores = {}

        for cat in cats:
            if cat in comparison:
                continue

            cat_mask = cats_col == cat
            if comparison == "rest":
                others_mask = ~cat_mask
            else:
                others_mask = cats_col.isin(comparison)

            choice_1 = np.random.choice(cat_mask.sum(), n_sample)
            choice_2 = np.random.choice(others_mask.sum(), n_sample)

            adata_cat = adata[cat_mask][choice_1]
            adata_others = adata[others_mask][choice_2]

            if use_directions:
                directions = adata.uns[directions_key]
            else:
                directions = None

            z0, means0, vars0 = forward_encoder(model, adata_cat.X)
            z1, means1, vars1 = forward_encoder(model, adata_others.X)

            if not exact:
                if directions is not None:
                    z0 *= directions
                    z1 *= directions

                if select_terms is not None:
                    z0 = z0[:, select_terms]
                    z1 = z1[:, select_terms]

                to_reduce = z0 > z1

                zeros_mask = (np.abs(z0).sum(0) == 0) | (np.abs(z1).sum(0) == 0)
            else:
                from scipy.special import erfc

                means0, vars0 = z0
                means1, vars1 = z1

                if directions is not None:
                    means0 *= directions
                    means1 *= directions

                if select_terms is not None:
                    means0 = means0[:, select_terms]
                    means1 = means1[:, select_terms]
                    vars0 = vars0[:, select_terms]
                    vars1 = vars1[:, select_terms]

                to_reduce = (means1 - means0) / np.sqrt(2 * (vars0 + vars1))
                to_reduce = 0.5 * erfc(to_reduce)

                zeros_mask = (np.abs(means0).sum(0) == 0) | (np.abs(means1).sum(0) == 0)

            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            bf = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            bf[zeros_mask] = 0

            scores[cat] = dict(p_h0=p_h0, p_h1=p_h1, bf=bf)

        adata.uns[key_added] = scores


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


