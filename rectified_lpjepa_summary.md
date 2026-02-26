# Rectified LpJEPA — Résumé de compréhension

**Auteurs :** Yilun Kuang, Yash Dagade, Tim Rudner, Randall Balestriero, Yann LeCun (NYU, 2026)  
**Paper :** arXiv:2602.01456 | **GitHub :** github.com/YilunKuang/rectified-lp-jepa

---

## 1. Le problème de base : Self-Supervised Learning et Feature Collapse

En apprentissage auto-supervisé (SSL), on veut apprendre une représentation `z = f_θ(x)` sans labels. L'idée : deux vues `x` et `x'` du même input (crop, rotation, etc.) doivent avoir des représentations similaires → on minimise la distance ℓ₂ entre `z` et `z'`.

**Problème : feature collapse.** Le réseau peut tricher en mappant tout au même vecteur → loss minimisée mais représentation inutile. Solution : ajouter une régularisation qui force les features à être bien distribués dans l'espace.

---

## 2. Distribution-Matching comme Régularisation

On ajoute un terme de régularisation qui pousse la distribution des features `P_z` vers une distribution cible `Q` :

```
min_θ  E[||z - z'||₂]  +  L(P_z || Q)  +  L(P_z' || Q)
```

**Problème en haute dimension :** matcher directement deux distributions en dim d est intractable (curse of dimensionality).

**Solution : Théorème de Cramér-Wold.** Deux distributions sont égales ssi toutes leurs projections 1D sont égales. On décompose donc le matching haute dimension en une expectation de matchings 1D :

```
E_c [ L( P_{c⊤z} || P_{c⊤y} ) ]
```

avec `c` tiré uniformément sur la sphère ℓ₂ unitaire. En pratique, un nombre fini de projections suffit.

---

## 3. La progression des distributions cibles Q

### LeJEPA original — Gaussienne isotrope N(0, I_d)
- Distribution max-entropie sous contrainte ℓ₂ fixée
- Features denses, bien répartis sur la sphère ℓ₂
- Régularisation : **SIGReg** (test d'Epps-Pulley sur les fonctions caractéristiques)
- **One-sample test** : la Gaussienne est stable sous projection → la cible a une forme analytique fermée

### LpJEPA — Generalized Gaussian ∏GN_p(0, σ)
- Distribution max-entropie sous contrainte ℓ_p fixée
- p=2 → Gaussienne (LeJEPA), p=1 → Laplace (sparsité via ℓ₁)
- Pour 0 < p < 1 : quasi-norm ℓ_p, encore plus proche de ℓ₀ → représentations encore plus sparse
- La géométrie de la sphère ℓ_p (plus "étoilée" quand p → 0) explique géométriquement pourquoi ça induit la sparsité

### Rectified LpJEPA — RGG ∏RGN_p(μ, σ) ← **L'INNOVATION CLEF**
- On applique une **ReLU coordonnée par coordonnée** sur la Generalized Gaussian → Rectified Generalized Gaussian
- La RGG est un mélange : **masse de Dirac en 0** (coordonnées exactement nulles) + **Truncated GG sur (0,∞)**
- Elle encode directement la **ℓ₀-norm** : E[||x||₀] = d · Φ(μ/σ) → contrôlable analytiquement via {μ, σ, p}
- En haute dimension, presque toute la masse se concentre sur les axes (bord du cône orthant positif)
- Elle préserve la propriété max-entropie (via l'entropie dimensionnelle de Rényi)

---

## 4. Pourquoi la RGG nécessite un two-sample test (RDMReg)

La Gaussienne est **stable sous projections linéaires** (une projection d'une Gaussienne reste Gaussienne → forme analytique disponible → one-sample test possible).

La RGG **n'est PAS stable** sous projections : projeter une RGG donne une distribution qui ne fait plus partie de la famille RGG. Impossible d'avoir une forme analytique → il faut un **two-sample test nonparamétrique**.

**Solution : Rectified Distribution Matching Regularization (RDMReg)** = **Sliced 2-Wasserstein distance** :

```python
def rdmreg_loss(z, target_samples, num_projections):
    # 1. Random projections sur la sphère ℓ₂
    projections = torch.randn(num_projections, D, device=device)
    projections = projections / projections.norm(dim=1, keepdim=True)
    
    # 2. Projeter les features z ET les samples RGG
    proj_z      = torch.matmul(z, projections.T)
    proj_target = torch.matmul(target_samples, projections.T)
    
    # 3. Trier les deux distributions projetées
    proj_z_sorted,      _ = torch.sort(proj_z,      dim=0)
    proj_target_sorted, _ = torch.sort(proj_target, dim=0)
    
    # 4. MSE entre les distributions triées = SW₂²
    return torch.mean((proj_z_sorted - proj_target_sorted)**2)
```

---

## 5. Architecture Rectified LpJEPA

**Standard SSL :**
```
z = g_θ₂( g_θ₁(x) )       # backbone + MLP projector
```

**Rectified LpJEPA :**
```
z = ReLU( g_θ₂( g_θ₁(x) ) )   # ReLU FINAL obligatoire !
```

Ce ReLU final est un **choix architectural délibéré et essentiel** : il garantit que les features sont dans l'orthant positif, ce qui est nécessaire pour que la distribution des features corresponde à la distribution cible RGG.

**Objective complète :**
```
min_θ  E[||z - z'||₂]  +  E_c[RDMReg(P_{c⊤z} || P_{c⊤y})]  +  E_c[RDMReg(P_{c⊤z'} || P_{c⊤y})]

avec y ~ ∏ RGN_p(μ, σ)
```

---

## 6. Paramètres et tradeoff Sparsité/Performance

| Paramètre | Rôle |
|-----------|------|
| `μ` | Contrôle la sparsité : μ négatif → plus de zéros → plus sparse |
| `p` | Forme de la distribution : p petit → sphère ℓ_p plus étoilée → plus sparse |
| `σ` | Fixé à Γ(1/p)^(1/2) / (p^(1/p) · Γ(3/p)^(1/2)) |

Sur CIFAR-100 : zone optimale autour de **20-40% de sparsité** où la performance reste ~0.61. Au-delà de ~80%, la performance chute brutalement.

---

## 7. Récap des différences LeJEPA vs Rectified LpJEPA

| | LeJEPA (original) | Rectified LpJEPA |
|--|--|--|
| **Cible Q** | Gaussienne N(0, I) | RGG ∏RGN_p(μ, σ) |
| **Sparsité** | Dense (ℓ₂) | Sparse (ℓ₀ explicite) |
| **Test** | One-sample (Epps-Pulley) | Two-sample (Sliced Wasserstein) |
| **Régularisation** | SIGReg | RDMReg |
| **Architecture** | z = MLP(backbone(x)) | z = ReLU(MLP(backbone(x))) |
| **Sampling cible** | Analytique (pas de samples) | Samples RGG au runtime |

---

*Paper : arXiv:2602.01456 — Kuang et al., 2026*
