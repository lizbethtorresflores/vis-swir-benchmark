import torch
import numpy as np

class SequenceMatchesContainer:
    """
    Conteneur pour stocker les matches entre paires d'images VIS/SWIR.
    OPTIMISÉ: Ne stocke QUE les keypoints (pas les images)
    """
    
    def __init__(self) -> None:
        self.pairs: list[dict] = []
    
    def add_pair(
        self,
        matches_idx: np.ndarray,
        pts_vis: np.ndarray,
        pts_swir: np.ndarray
    ) -> None:
        """
        Ajoute une paire avec uniquement les matches.
        
        OPTIMISÉ: Ne stocke plus rgb_image ni swir_image
        
        Parameters
        ----------
        pts_vis : np.ndarray
            Points keypoints VIS, shape (K, 2).
        pts_swir : np.ndarray
            Points keypoints SWIR, shape (K, 2).
        matches_idx : np.ndarray
            Indices des matches, shape (K, 2).
        """
        pair_data = {
            'pts_vis': pts_vis if isinstance(pts_vis, np.ndarray) else pts_vis.cpu().numpy(),
            'pts_swir': pts_swir if isinstance(pts_swir, np.ndarray) else pts_swir.cpu().numpy(),
            'matches_idx': matches_idx,
            'nb_matches': len(matches_idx)
        }
        
        self.pairs.append(pair_data)
    


    def get_all_matches(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Concatène tous les points de toutes les paires.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (pts_vis_all, pts_swir_all) avec shape (N_total_global, 2).
        """
        if not self.pairs:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        # Collecter les arrays valides (non vides)
        valid_vis = [p['pts_vis'] for p in self.pairs if len(p['pts_vis']) > 0]
        valid_swir = [p['pts_swir'] for p in self.pairs if len(p['pts_swir']) > 0]

        # DEBUG
        print(f"DEBUG get_all_matches: {len(valid_vis)} paires valides")
        print(f"Shapes: {[v.shape for v in valid_vis[:3]]}")

        pts_vis_all = np.vstack(valid_vis)
        pts_swir_all = np.vstack(valid_swir)

        return pts_vis_all, pts_swir_all

    def stats(self) -> tuple[int, int]:
        """
        Statistiques sur les matches.

        Returns
        -------
        tuple[int, int]
            (total_matches, nb_pairs)
        """
        total = sum(p['nb_matches'] for p in self.pairs)
        return total, len(self.pairs)

    def __len__(self) -> int:
        """Retourne le nombre de paires stockées."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Accès à une paire par index."""
        return self.pairs[idx]
