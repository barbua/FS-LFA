import numpy as np
import torch

def LFA(data, q, max_iter=30):
    """
    Args:
        X (torch.Tensor): Data matrix (d x n).
        S_x (torch.Tensor): Sample covariance matrix (d x d).
        r (int): Number of principal components.
        n (int): Number of samples.
        max_iter (int, optional): Maximum number of EM iterations. Defaults to 20.

    Returns:
        W (torch.Tensor): Loading matrix (d x r).
        Psi (torch.Tensor): Initial noise variance (d).
    
    """                                                   
    n,d = data.shape
    mu = torch.mean(data, dim=0)
    X = (data - mu).t()
    S = torch.cov(X) + 0.0001 * torch.eye(d)
    u, s, _ = torch.linalg.svd(X)
    s=s**2/n
    sig_ML = (1 / (d - q)) * torch.sum(s[q:])
    W = u[:, :q] @ ((torch.diag(s[:q]) - sig_ML * torch.eye(q))**(0.5))
    Psi = sig_ML * torch.ones(d)
    #W_ML, Psi_ML = EMv2(data_centered_t, S, W_st, Psi_st, q, n)
    for t in range(max_iter):
        C=torch.diag(Psi) + W @ W.t()
        beta=torch.linalg.solve(C,W.t(),left=False)
        E_gam = beta @ X
        M2 = E_gam @ E_gam.t()
        E_ggam = torch.eye(q) - beta @ W
        E_ggam = n*E_ggam + M2

        #M3 = torch.linalg.inv(E_ggam)
        W_new = X @ E_gam.t()
        #W_new = W_new @ M3
        W_new =torch.linalg.solve(E_ggam,W_new,left=False)
        M4 = beta @ S
        Psi_new = torch.diag(S - W_new @ M4)
        Psi_new = torch.clamp(Psi_new, min=1e-11) # Ensure Psi is positive

        #print(t,torch.max(torch.abs(W_new-W)),torch.max(torch.abs(Psi_new-Psi)))
        W = W_new
        Psi = Psi_new
    return W, Psi, beta

def PPCA(data, q=20):
    """
    Performs Feature Selection using Probabilistic Principal Component Analysis (PPCA)
    for each class and returns a list of feature indices sorted by their relevance.

    Args:
        trFeatures (torch.Tensor): Training features (N x d), where N is the number of samples
                                     and d is the number of features.
        trY (torch.Tensor): Training labels (N).
        nc (int): Number of classes.
        d (int): Number of features.
        q (int, optional): Number of principal components to retain. Defaults to 20.

    Returns:
        torch.Tensor: A tensor of shape (nc x d) containing feature indices sorted in
                      descending order of their relevance for each class.
    """
    n,d=data.shape
    mu = torch.mean(data, dim=0)
    data_centered = (data - mu).t()  # Center the data

    # Perform Singular Value Decomposition (SVD)
    U, S, V = torch.linalg.svd(data_centered)
    S_squared = S**2/n

    # Estimate the noise variance (sig_ML)
    sig_ML = (1 / (d - q)) * torch.sum(S_squared[q:])

    # Construct the loading matrix A_ML
    eigenvalues_adjusted = torch.diag(S_squared[:q]) - sig_ML * torch.eye(q)
    # Ensure the diagonal elements are non-negative before taking the square root
#        eigenvalues_adjusted = torch.relu(eigenvalues_adjusted)
    A_ML = U[:, :q] @ (eigenvalues_adjusted**(0.5))
    return A_ML
