import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union



class BEELINEDataset(Dataset):
    """
    PyTorch Dataset for BEELINE benchmark datasets.

    Supports loading expression data and GRN information for machine learning tasks
    such as GRN inference, cell type classification, and trajectory analysis.

    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing BEELINE data
    category : str
        One of 'Curated', 'scRNA-Seq', or 'Synthetic'
    dataset_name : str
        Base dataset name (e.g., 'GSD', 'dyn-BF', 'hESC')
    variant : str
        Variant identifier (e.g., '2000-1', '500-3')
    transform : callable, optional
        Optional transform to apply to expression data
    """

    def __init__(
        self,
        base_dir: str = '/projects/heads/data',
        category: str = 'Curated',
        dataset_name: str = 'GSD',
        variant: str = '2000-1',
        transform: Optional[callable] = None
    ):
        super().__init__()

        self.base_dir = Path(base_dir)
        self.category = category
        self.dataset_name = dataset_name
        self.variant = variant
        self.transform = transform
        self.adjacency = None

        # Load data
        self._load_data()

        # Prepare adjacency matrix
        self._prepare_adjacency_matrix()

    def _load_data(self):
        """Load expression data and GRN from files"""
        # Construct path (variant should be just the suffix, e.g., '2000-1')
        dataset_path = self.base_dir / 'inputs' / self.category / self.dataset_name / f"{self.dataset_name}-{self.variant}"

        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        # Load expression data
        expr_file = dataset_path / 'ExpressionData.csv'
        if expr_file.exists():
            self.expression = pd.read_csv(expr_file, index_col=0)
            self.gene_names = self.expression.index.tolist()
            self.cell_names = self.expression.columns.tolist()
        else:
            raise FileNotFoundError(f"Expression data not found: {expr_file}")

        # Load GRN
        grn_file = dataset_path / 'refNetwork.csv'
        if grn_file.exists():
            self.grn = pd.read_csv(grn_file)
        else:
            raise FileNotFoundError(f"GRN file not found: {grn_file}")

        # Load pseudotime
        pseudo_file = dataset_path / 'PseudoTime.csv'
        if pseudo_file.exists():
            pseudotime_df = pd.read_csv(pseudo_file)
            
            # Create lineage dataframe (cell name and lineage index)
            pseudo_cols = [col for col in pseudotime_df.columns if 'PseudoTime' in col]
            cell_col = pseudotime_df.columns[0]  # First column is cell names
            
            if pseudo_cols:
                # Determine which lineage each cell belongs to
                lineage_indices = []
                for idx, row in pseudotime_df.iterrows():
                    # Find which lineage column has non-NaN value
                    for i, col in enumerate(pseudo_cols, start=1):
                        if pd.notna(row[col]):
                            lineage_indices.append(i)
                            break
                    else:
                        lineage_indices.append(0)  # No lineage if all NaN
                
                # Create lineage dataframe
                self.lineage = pd.DataFrame({
                    'cell_name': pseudotime_df[cell_col],
                    'lineage': lineage_indices
                })
                
                # Create pseudotime dataframe with cell names and pseudotime values
                # Combine all lineage columns into single pseudotime column
                pseudotime_values = pseudotime_df[pseudo_cols].max(axis=1).fillna(0.0)
                self.pseudotime = pd.DataFrame({
                    'cell_name': pseudotime_df[cell_col],
                    'pseudotime': pseudotime_values
                })
            else:
                self.pseudotime = None
                self.lineage = None
        else:
            self.pseudotime = None
            self.lineage = None

        # Convert to numpy for faster access
        self.expression_array = self.expression.values.astype(np.float32)

        # Transpose to cells x genes
        self.expression_array = self.expression_array.T

    def _prepare_adjacency_matrix(self):
        """Create adjacency matrix from GRN"""
        # Create gene to index mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}

        # Initialize adjacency matrix
        n_genes = len(self.gene_names)
        adjacency = np.zeros((n_genes, n_genes), dtype=np.float32)

        # Determine column names
        if 'Gene1' in self.grn.columns:
            col1, col2 = 'Gene1', 'Gene2'
            weight_col = 'Type' if 'Type' in self.grn.columns else None
        else:
            col1 = self.grn.columns[0]
            col2 = self.grn.columns[1]
            weight_col = self.grn.columns[2] if len(self.grn.columns) > 2 else None

        # Fill adjacency matrix
        for _, row in self.grn.iterrows():
            gene1, gene2 = row[col1], row[col2]
            if gene1 in gene_to_idx and gene2 in gene_to_idx:
                i, j = gene_to_idx[gene1], gene_to_idx[gene2]
                
                # Handle edge weights/types
                if weight_col:
                    edge_type = row[weight_col]
                    # Convert edge type to numeric weight
                    if isinstance(edge_type, str):
                        if edge_type == '+' or edge_type.lower() == 'activation':
                            weight = 1.0
                        elif edge_type == '-' or edge_type.lower() == 'repression':
                            weight = -1.0
                        else:
                            try:
                                weight = float(edge_type)
                            except (ValueError, TypeError):
                                weight = 1.0  # Default to activation
                    else:
                        weight = float(edge_type)
                else:
                    weight = 1.0
                    
                adjacency[i, j] = weight

        # Convert to binary: keep only -1, 0, 1 (remove any other values)
        adjacency = np.where(adjacency > 0, 1.0, adjacency)
        adjacency = np.where(adjacency < 0, -1.0, adjacency)
        
        self.adjacency = adjacency

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.cell_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Returns:
        --------
        tuple: (expression_vector, adjacency_matrix)
        """
        # Get expression vector
        expr = torch.from_numpy(self.expression_array[idx]).float()

        # Return expression and adjacency matrix
        adj = torch.from_numpy(self.adjacency).float()
        return expr, adj

    def get_full_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data as PyTorch tensors

        Returns:
        --------
        dict: Dictionary with 'expression', 'adjacency', 'pseudotime', etc.
        """
        data = {
            'expression': torch.from_numpy(self.expression_array).float(),
            'adjacency': torch.from_numpy(self.adjacency).float(),
            'gene_names': self.gene_names,
            'cell_names': self.cell_names
        }

        if self.pseudotime is not None:
            data['pseudotime'] = self.pseudotime
        
        if self.lineage is not None:
            data['lineage'] = self.lineage

        return data

    def get_metadata(self) -> Dict:
        """Get dataset metadata"""
        # Calculate sparsity (fraction of zero values in expression data)
        sparsity = (self.expression_array == 0).sum() / self.expression_array.size
        
        return {
            'category': self.category,
            'dataset': self.dataset_name,
            'variant': self.variant,
            'n_genes': len(self.gene_names),
            'n_cells': len(self.cell_names),
            'n_edges': len(self.grn),
            'sparsity': float(sparsity)
        }


