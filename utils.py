import numpy as np

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



def compute_nn_from_connectivity(adata, key, n_neighbors):
        """
        Pick the top-n_neighbors nearest neighbors based on a precomputed connectivity matrix.

        Parameters
        ----------
        adata
            AnnData object with a connectivity matrix in adata.obsp[key].
        key : str
            Which connectivity matrix to use (e.g. 'connectivities').
        n_neighbors : int
            How many neighbors to pick per cell.

        Stores
        ------
        adata.uns['indices']
            Array of shape (n_cells, n_neighbors) with neighbor indices.
        adata.uns['weights']
            Array of shape (n_cells, n_neighbors) with their connectivity weights.
        """
        from scipy.sparse import issparse
        # 1) pull out dense matrix
        conn = adata.obsp[key]
        if issparse(conn):
            conn = conn.toarray()

        # 2) mask out self-connections so they won’t sort to the top
        #    (assumes diagonal was 1 or max; set to -inf to drop it)
        np.fill_diagonal(conn, -np.inf)

        # 3) sort each row descending, take first n_neighbors
        #    argsort gives ascending, so negate to get descending
        idx = np.argsort(-conn, axis=1)[:, :n_neighbors]
        weights = np.take_along_axis(conn, idx, axis=1)

        adata.uns['indices'] = idx
        adata.uns['weights'] = weights