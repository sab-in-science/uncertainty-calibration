# ============================================================
# refactored_UC_functions.jl
# Implementation for the paper uncertainty calibration companion repo
# Supports datasets: GermanyHyper, newShiftData (LMA only), Corn
# Implements LV-localized split-conformal-inspired uncertainty calibration
# Duma, Z.-S., Lamminpää, O., Susiluoto, J., Haario, H., Zheng, T.,
# Sihvonen, T., Braverman, A., Townsend, P. A., Reinikainen, S.-P.
# (2025).\
# *Uncertainty calibration for latent-variable regression models.*\
# arXiv:2512.23444v1.
# ============================================================

module LVLocalizedUQ

using Random
using Statistics
using LinearAlgebra
using MAT
using XLSX
using DataFrames

# -----------------------------
# Utilities
# -----------------------------

"""
Column-wise z-score standardization.
Returns standardized data, and (mu, sigma).
"""
function zscore_cols(X::AbstractMatrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .== 0.0] .= 1.0
    return (X .- μ) ./ σ, μ, σ
end

normalize_cols(X, μ, σ) = (X .- μ) ./ σ

"""
Simple causal moving average for each vector (used optionally for smoothing spectra).
"""
function moving_average(x::AbstractVector, window_size::Int)
    @assert window_size ≥ 1
    return [mean(x[max(1, i-window_size+1):i]) for i in 1:length(x)]
end

"""
Remove rows where X has NaN/Inf OR y has NaN/Inf.
"""
function mask_finite_rows(X::AbstractMatrix, y::AbstractVector)
    badX = findall(r -> any(z -> isnan(z) || isinf(z), r), eachrow(X))
    bady = findall(z -> isnan(z) || isinf(z), y)
    mask = trues(size(X, 1))
    mask[badX] .= false
    mask[bady] .= false
    return mask
end

# -----------------------------
# Dataset loading (only 3 kept)
# -----------------------------

"""
load_raw_data(dataset, model)

Returns (X, y) in Float64.
- GermanyHyper: data/soilMoisture.mat with variables x, y
- newShiftData: data/shiftNewData.mat with spectraMat, traitsMat; ONLY LMA allowed
- Corn: data/corn_spectra.xlsx and data/corn_propvals.xlsx
"""
function load_raw_data(model::Dict{Symbol,Any})
    dataset = model[:dataset]

    if dataset == "GermanyHyper"
        mat = matread("soilMoisture.mat")
        @assert haskey(mat, "x") "soilMoisture.mat must contain variable 'x'"
        @assert haskey(mat, "y") "soilMoisture.mat must contain variable 'y'"
        X = Array{Float64}(mat["x"])
        y = Array{Float64}(vec(mat["y"]))
        return X, y

    elseif dataset == "newShiftData"
        @assert get(model, :dataSubset, "LMA") == "LMA" "This repo supports only newShiftData with dataSubset=\"LMA\"."
        mat = matread("shiftNewData.mat")
        @assert haskey(mat, "spectraMat") "shiftNewData.mat must contain 'spectraMat'"
        @assert haskey(mat, "traitsMat")  "shiftNewData.mat must contain 'traitsMat'"

        X0 = Array{Float64}(mat["spectraMat"])
        Y0 = mat["traitsMat"]

        # Your original mapping: LMA = column 20
        y0 = Array{Float64}(Y0[:, 20])

        # Optional band selection
        if haskey(model, :selectedBands) && !(model[:selectedBands] isa Integer) && !isempty(model[:selectedBands])
            X0 = X0[:, model[:selectedBands]]
        end

        mask = mask_finite_rows(X0, y0)
        return X0[mask, :], y0[mask]

    elseif dataset == "Corn"
        spectra_file  = "corn_spectra.xlsx"
        propvals_file = "corn_propvals.xlsx"

        spectra_df  = DataFrame(XLSX.readdata(spectra_file, "corn_spectra", "A1:ZX80"), :auto)
        propvals_df = DataFrame(XLSX.readdata(propvals_file, "Sheet1", "A1:D80"), :auto)

        X = Array{Float64}(Matrix(spectra_df))
        Y = Array{Float64}(Matrix(propvals_df))

        target_col = get(model, :cornTargetCol, 1)  # 1=moisture if that's what you used; set explicitly in scripts
        y = Array{Float64}(Y[:, target_col])

        mask = mask_finite_rows(X, y)
        return X[mask, :], y[mask]

    else
        error("Unsupported dataset $(dataset). Only GermanyHyper, newShiftData (LMA), Corn are included in paper repo.")
    end
end

# -----------------------------
# Preprocess / split / standardize
# -----------------------------

"""
preprocess(X, model)

Currently supports optional smoothing.
"""
function preprocess(X::AbstractMatrix, model::Dict{Symbol,Any})
    if get(model, :filtering, false)
        w = get(model, :smoothWindow, 5)
        Xs = hcat([moving_average(view(X, i, :), w) for i in 1:size(X, 1)]...)'
        return Array{Float64}(Xs)
    end
    return Array{Float64}(X)
end

"""
split_train_test(X, y, testSize, rng)

Returns (Xtr, ytr, Xte, yte).
"""
function split_train_test(X::AbstractMatrix, y::AbstractVector; testSize::Float64=0.2, rng=Random.default_rng())
    n = size(X, 1)
    n_test = round(Int, n * testSize)
    perm = randperm(rng, n)
    idx_test = perm[1:n_test]
    idx_train = perm[n_test+1:end]
    return X[idx_train, :], y[idx_train], X[idx_test, :], y[idx_test]
end

"""
standardize!(model)

Standardizes Y always.
Standardizes X unless you explicitly set model[:standardizeX]=false.
Stores μ/σ for X and Y.
"""
function standardize!(model::Dict{Symbol,Any})
    # y as column matrix for consistency
    model[:Y] = reshape(model[:Y], :, 1)
    model[:Ytest] = reshape(model[:Ytest], :, 1)

    model[:Y], model[:muY], model[:stdY] = zscore_cols(model[:Y])
    model[:Ytest] = normalize_cols(model[:Ytest], model[:muY], model[:stdY])

    doX = get(model, :standardizeX, true)
    if doX
        model[:X], model[:muX], model[:stdX] = zscore_cols(model[:X])
        model[:Xtest] = normalize_cols(model[:Xtest], model[:muX], model[:stdX])
    end

    return model
end

# -----------------------------
# PCA / PLS core
# -----------------------------

"""
compute_pca(X)

Returns:
T: scores
P: loadings
s2: eigenvalues (variance explained per component, proportional to singular values^2)
"""
function compute_pca(X::AbstractMatrix)
    X = Array{Float64}(X)
    @assert !any(isnan, X) && !any(isinf, X) "X contains NaN/Inf."
    U, s, Vt = svd(X; full=false)
    T = U * Diagonal(s)
    P = Vt'
    s2 = s .^ 2
    return T, P, s2
end

"""
compute_pls_nipals(X, y, H)

Minimal NIPALS PLS for univariate y.
Returns:
T, P, W, q (vector), and weights w_explained (LV weights based on explained X-variance proxy)
"""
function compute_pls_nipals(X::AbstractMatrix, y::AbstractVector, H::Int)
    Xh = copy(Array{Float64}(X))
    yh = copy(Array{Float64}(y))

    n, p = size(Xh)
    T = zeros(n, H)
    P = zeros(p, H)
    W = zeros(p, H)
    q = zeros(H)

    # crude explained-variance weights (optional; you can replace with something else)
    # We'll compute w_explained after loop using reconstruction RSS.
    for h in 1:H
        # w ∝ X' y
        w = Xh' * yh
        w ./= norm(w) + eps()
        t = Xh * w

        # regression of y on t
        qh = (t' * yh) / (t' * t + eps())
        pvec = (Xh' * t) / (t' * t + eps())

        # deflation
        Xh .-= t * pvec'
        yh .-= t * qh

        T[:, h] = t
        P[:, h] = pvec
        W[:, h] = w
        q[h] = qh
    end

    # weights based on X-variance explained per LV (simple proxy)
    # compute how much each LV reduces RSS on X (using original X and rank-1 reconstructions)
    X0 = Array{Float64}(X)
    TSS = sum((X0 .- mean(X0, dims=1)).^2)
    expl = zeros(H)
    Xrec = zeros(size(X0))
    for h in 1:H
        Xrec_h = T[:, h] * P[:, h]'
        RSS = sum((X0 .- Xrec_h).^2)
        expl[h] = 1 - RSS / (TSS + eps())
    end
    # normalize positive weights
    expl = max.(expl, 0.0)
    w_explained = expl / (sum(expl) + eps())

    return T, P, W, q, w_explained
end

# -----------------------------
# Kernel functions (cleaned)
# -----------------------------


function normDiff(X::AbstractMatrix, Y::AbstractMatrix)
    X = Array{Float64}(X)
    Y = Array{Float64}(Y)
    nsx = vec(sum(X.^2, dims=2))          # n
    nsy = vec(sum(Y.^2, dims=2))          # m
    return -2.0 .* (X * Y') .+ nsx .+ nsy'
end

# keep name sqdist if you like, but make it identical
sqdist(X::AbstractMatrix, Y::AbstractMatrix) = normDiff(X, Y)

"""
kernel_matrix(X, Y, params, model; training::Bool)

Supports kernelType: gaussian, matern1/2, matern3/2, matern5/2, cauchy
Supports kernelVersion: combined (single lengthscale), individual (per-feature), family, individualScale
Adds ridge term params[end] when model[:PLS] != "PCR" AND training=true.
Centers kernel when training=true.
"""

"Raw (uncentered) kernel, no ridge, works for train or test"
function kernel_raw(X::AbstractMatrix, Y::AbstractMatrix, params::AbstractVector, model::Dict{Symbol,Any})
    kv = get(model, :kernelVersion, "combined")
    kt = get(model, :kernelType, "gaussian")

    X = Array{Float64}(X); Y = Array{Float64}(Y)
    nX, nY = size(X,1), size(Y,1)
    K = zeros(Float64, nX, nY)

    if kv == "combined"
        D = sqdist(X, Y)
        K .= kernel_from_sqdist(D, params[1], kt)

    elseif kv == "individual"
        p = size(X, 2)
        @assert length(params) ≥ p
        for j in 1:p
            D = normDiff(X[:, j:j], Y[:, j:j])
            K .+= kernel_from_sqdist(D, params[j], kt)
        end

    elseif kv == "individualScale"
        p = size(X, 2)
        @assert length(params) ≥ 2p
        for j in 1:p
            D = normDiff(X[:, j:j], Y[:, j:j])
            K .+= params[j+p] .* kernel_from_sqdist(D, params[j], kt)
        end

    elseif kv == "family"
        D = sqdist(X, Y)
        d = sqrt.(max.(D, 0.0) .+ 1e-12)
        # gaussian
        K .+= params[2] .* exp.(-D ./ (2 * params[1]^2 + eps()))
        # matern 1/2
        K .+= params[3] .* exp.(-d ./ (params[4] + eps()))
        # matern 3/2
        t32 = sqrt(3.0) .* d ./ (params[6] + eps())
        K .+= params[5] .* (1 .+ t32) .* exp.(-t32)
        # matern 5/2
        t52 = sqrt(5.0) .* d ./ (params[8] + eps())
        K .+= params[7] .* (1 .+ t52 .+ (5 .* D) ./ (3 .* (params[8]^2 + eps()))) .* exp.(-t52)
        # cauchy
        K .+= params[10] .* (1.0 ./ (1 .+ D ./ (params[9]^2 + eps())))
    else
        error("Unsupported kernelVersion $(kv)")
    end

    return K
end

"Center a square training kernel"
center_train(K::AbstractMatrix) = begin
    n = size(K,1)
    one = ones(n,n) / n
    K .- one*K .- K*one .+ one*K*one
end

"Center test kernel using the *raw* training kernel moments"
center_test(Ktest::AbstractMatrix, Ktrain_raw::AbstractMatrix) = begin
    n = size(Ktrain_raw,1)
    nT = size(Ktest,1)
    oneN = ones(n,n) / n
    oneT = ones(nT,n) / n
    Ktest .- oneT*Ktrain_raw .- Ktest*oneN .+ oneT*Ktrain_raw*oneN
end


function kernel_matrix(X::AbstractMatrix, Y::AbstractMatrix, params::AbstractVector, model::Dict{Symbol,Any}; training::Bool)
    kv = get(model, :kernelVersion, "combined")
    kt = get(model, :kernelType, "gaussian")
    pls = get(model, :PLS, "PCR")

    X = Array{Float64}(X)
    Y = Array{Float64}(Y)

    nX = size(X, 1)
    nY = size(Y, 1)

    # build uncentered kernel
    K = zeros(Float64, nX, nY)

    if kv == "combined"
        D = sqdist(X, Y)
        K .= kernel_from_sqdist(D, params[1], kt)

    elseif kv == "individual"
        # params: per-feature lengthscales + maybe ridge at end
        p = size(X, 2)
        @assert length(params) ≥ p "individual kernel needs at least p params"
        for j in 1:p
            D = normDiff(X[:, j:j], Y[:, j:j])
            K .+= kernel_from_sqdist(D, params[j], kt)
        end

    elseif kv == "individualScale"
        p = size(X, 2)
        @assert length(params) ≥ 2p "individualScale needs 2p params (lengthscales + scales)"
        for j in 1:p
            D = normDiff(X[:, j:j], Y[:, j:j])
            K .+= params[j + p] * kernel_from_sqdist(D, params[j], kt)
        end

    elseif kv == "family"
        # params interpreted similarly to your old family version
        # [σ_g, a_g, a_m12, ℓ_m12, a_m32, ℓ_m32, a_m52, ℓ_m52, ℓ_cauchy, a_cauchy, (optional ridge)]
        D = sqdist(X, Y)
        d = sqrt.(max.(D, 0.0) .+ 1e-12)

        # gaussian
        K .+= params[2] .* exp.(-D ./ (2 * params[1]^2 + eps()))

        # matern 1/2: exp(-d/ℓ)
        K .+= params[3] .* exp.(-d ./ (params[4] + eps()))

        # matern 3/2: (1 + √3 d/ℓ) exp(-√3 d/ℓ)
        t32 = sqrt(3.0) .* d ./ (params[6] + eps())
        K .+= params[5] .* (1 .+ t32) .* exp.(-t32)

        # matern 5/2: (1 + √5 d/ℓ + 5 d^2 / (3ℓ^2)) exp(-√5 d/ℓ)
        t52 = sqrt(5.0) .* d ./ (params[8] + eps())
        K .+= params[7] .* (1 .+ t52 .+ (5 .* D) ./ (3 .* (params[8]^2 + eps()))) .* exp.(-t52)

        # cauchy
        K .+= params[10] .* (1.0 ./ (1 .+ D ./ (params[9]^2 + eps())))

    else
        error("Unsupported kernelVersion $(kv)")
    end

    # For training kernels: center and maybe add ridge
    if training
        # Center kernel in feature space using double-centering
        # Kc = K - 1_n K - K 1_m + 1_n K 1_m
        oneX = ones(nX, nX) / nX
        oneY = ones(nY, nY) / nY
        K = K .- oneX * K .- K * oneY .+ oneX * K * oneY

        if pls != "PCR"
            # ridge term is last element (consistent with your old code)
            ridge = params[end]
            @assert nX == nY "training kernel must be square to add ridge"
            K .+= ridge .* I(nX)
        end
    end

    return K
end

function kernel_from_sqdist(D::AbstractMatrix, ℓ::Real, kernelType::String)
    ℓ = float(ℓ) + eps()
    if kernelType == "gaussian"
        return exp.(-D ./ (2 * ℓ^2))
    elseif kernelType == "matern1/2"
        d = sqrt.(max.(D, 0.0) .+ 1e-12)
        return exp.(-d ./ ℓ)
    elseif kernelType == "matern3/2"
        d = sqrt.(max.(D, 0.0) .+ 1e-12)
        t = sqrt(3.0) .* d ./ ℓ
        return (1 .+ t) .* exp.(-t)
    elseif kernelType == "matern5/2"
        d = sqrt.(max.(D, 0.0) .+ 1e-12)
        t = sqrt(5.0) .* d ./ ℓ
        return (1 .+ t .+ (5 .* D) ./ (3 .* ℓ^2)) .* exp.(-t)
    elseif kernelType == "cauchy"
        return 1.0 ./ (1 .+ D ./ (ℓ^2))
    else
        error("Unsupported kernelType $(kernelType)")
    end
end

# -----------------------------
# LV model fitting / prediction
# -----------------------------

"""
fit_model!(model)

Supported model[:method]:
- "PCR"   (linear PCR)
- "PLS"   (NIPALS PLS, univariate y)
- "KPCR"  (kernel PCR via KPCA scores)
- "KPLS"  (simple kernel PLS via NIPALS in kernel space; uses same PLS implementation on K)
"""
function fit_model!(model::Dict{Symbol,Any})
    method = model[:method]
    H = model[:dim]

    X = model[:X]
    y = vec(model[:Y])  # standardized y

    if method == "PCR"
        T, P, s2 = compute_pca(X)
        T = T[:, 1:H]
        P = P[:, 1:H]
        # regression in score space
        b = T \ y
        model[:P] = P
        model[:b] = b
        model[:T_train] = T

        # explained-variance weights
        w = s2[1:H] ./ (sum(s2[1:H]) + eps())
        model[:lv_weights] = vec(w)

    elseif method == "PLS"
        T, P, W, q, w_lv = compute_pls_nipals(X, y, H)
        model[:T_train] = T
        model[:P] = P
        model[:W] = W
        model[:q] = q
        model[:lv_weights] = vec(w_lv)

        R = inv(P' * W)          # H×H
        model[:b] = W * R * q    # p×1  (vector)

    elseif method == "KPCR"
        params = vec(model[:kernelParams])
        K = kernel_matrix(X, X, params, merge(model, Dict(:PLS=>"PCR")); training=true)
        # KPCA on centered K
        T, Pk, s2 = compute_pca(K)
        T = T[:, 1:H]
        b = T \ y
        model[:K_train] = K
        model[:Pk] = Pk[:, 1:H]
        model[:b] = b
        model[:T_train] = T
        w = s2[1:H] ./ (sum(s2[1:H]) + eps())
        model[:lv_weights] = vec(w)

    elseif method == "KPLS"
        params = vec(model[:kernelParams])

        Kraw = kernel_raw(X, X, params, model)
        Kc   = center_train(Kraw)

        # ridge ONLY here (training only, square only)
        ridge = params[end]
        Kc_reg = Kc .+ ridge .* I(size(Kc,1))

        T, P, W, q, w_lv = compute_pls_nipals(Kc_reg, y, H)

        model[:K_train_raw] = Kraw      # <-- needed for correct test centering
        model[:K_train]     = Kc_reg    # (optional, for debugging)
        model[:T_train] = T
        model[:P] = P
        model[:W] = W
        model[:q] = q
        model[:lv_weights] = vec(w_lv)

        R = inv(P' * W)
        model[:bK] = W * R * q

    else
        error("Unsupported method $(method). Use PCR, PLS, KPCR, KPLS.")
    end

    return model
end

"""
predict_scores(model, Xnew)

Returns latent scores for new samples for localization.
"""
function predict_scores(model::Dict{Symbol,Any}, Xnew::AbstractMatrix)
    method = model[:method]
    H = model[:dim]

    if method == "PCR"
        return Xnew * model[:P]  # P is p×H

    elseif method == "KPCR"
        params = vec(model[:kernelParams])
        Ktest = kernel_matrix(Xnew, model[:X], params, merge(model, Dict(:PLS=>"PCR")); training=false)
        # center test kernel using training kernel moments
        # approximate centering: Kc = Ktest - 1_test*Ktrain - Ktest*1_train + 1_test*Ktrain*1_train
        Ktrain = model[:K_train]
        n = size(Ktrain, 1)
        oneN = ones(n, n) / n
        nT = size(Ktest, 1)
        oneT = ones(nT, n) / n

        Kc = Ktest .- oneT * Ktrain .- Ktest * oneN .+ oneT * Ktrain * oneN
        return Kc * model[:Pk]  # scores

    elseif method == "PLS"
        W = model[:W]; P = model[:P]
        R = inv(P' * W)
        return Xnew * (W * R)   # scores Tnew

    elseif method == "KPLS"
        params = vec(model[:kernelParams])
        Ktest_raw = kernel_raw(Xnew, model[:X], params, model)
        Kc = center_test(Ktest_raw, model[:K_train_raw])

        W = model[:W]; P = model[:P]
        R = inv(P' * W)
        return Kc * (W * R)
    else
        error("Unsupported method $(method)")
    end
end

"""
predict_y(model, Xnew)

Returns predicted y in standardized space.
"""
function predict_y(model::Dict{Symbol,Any}, Xnew::AbstractMatrix)
    method = model[:method]

    if method == "PCR"
        Tnew = Xnew * model[:P]
        return Tnew * model[:b]

    elseif method == "KPCR"
        Tnew = predict_scores(model, Xnew)
        return Tnew * model[:b]

    elseif method == "PLS"
        return Xnew * model[:b]

    elseif method == "KPLS"
        params = vec(model[:kernelParams])

        Ktest_raw = kernel_raw(Xnew, model[:X], params, model)
        Kc = center_test(Ktest_raw, model[:K_train_raw])

        return Kc * model[:bK]
    else
        error("Unsupported method $(method)")
    end
end

# -----------------------------
# LV-localized conformal-inspired uncertainty calibration
# -----------------------------

"""
Create interval breaks from calibration LV scores.
"""
function lv_breaks(tcal::AbstractVector, k::Int)
    a = minimum(tcal)
    b = maximum(tcal)
    return range(a, stop=b, length=k+1)
end

"""
Assign each value to an interval index in 1..k.
Open-ended outside range -> clamp to nearest edge.
"""
function assign_intervals(vals::AbstractVector, breaks)
    k = length(breaks) - 1
    idx = similar(vals, Int)
    for i in eachindex(vals)
        v = vals[i]
        if v ≤ breaks[1]
            idx[i] = 1
        elseif v ≥ breaks[end]
            idx[i] = k
        else
            # find right edge
            j = searchsortedlast(breaks, v)  # returns position in breaks
            idx[i] = clamp(j, 1, k)
        end
    end
    return idx
end

"""
conformal_uq!(model)

Implements:
- split internal X into XTrain/XCal for calibration (default 50/50)
- fit predictive model on XTrain
- compute calibration residuals on XCal
- discretize LV scores into k intervals per LV
- compute per-interval residual quantiles
- aggregate per-LV conditional quantiles with lv_weights
Returns:
- model with qCal, qTest, yPredCal, yPredTest
"""
function conformal_uq!(model::Dict{Symbol,Any})
    α = get(model, :alpha, 0.05)     # nominal 95% => alpha=0.05
    k = get(model, :ni, 7)           # number of LV intervals
    H = model[:dim]
    rng = get(model, :rng, Random.default_rng())

    # internal split of training into (train/cal) for conformal calibration
    Xfull = model[:X]
    yfull = vec(model[:Y])

    n = size(Xfull, 1)
    ntrain = round(Int, 0.5n)
    perm = randperm(rng, n)
    idx_train = perm[1:ntrain]
    idx_cal = perm[ntrain+1:end]

    XTrain = Xfull[idx_train, :]
    yTrain = yfull[idx_train]
    XCal   = Xfull[idx_cal, :]
    yCal   = yfull[idx_cal]

    # Fit model on XTrain only
    m2 = copy(model)
    m2[:X] = XTrain
    m2[:Y] = reshape(yTrain, :, 1)
    fit_model!(m2)

    # Predict on cal and test
    yPredCal  = vec(predict_y(m2, XCal))
    yPredTest = vec(predict_y(m2, model[:Xtest]))

    # Scores for localization
    TCal  = predict_scores(m2, XCal)
    TTest = predict_scores(m2, model[:Xtest])

    # residuals on calibration set
    rCal = abs.(yPredCal .- yCal)

    # allocate
    qCal  = zeros(length(yCal))
    qTest = zeros(size(model[:Xtest], 1))

    # For diagnostics / plotting
    qHistory = fill(NaN, k, H)

    # LV weights (explained variance)
    w = vec(m2[:lv_weights])
    @assert length(w) == H

    # per LV: discretize and fill quantiles
    for h in 1:H
        tcal = TCal[:, h]
        ttest = TTest[:, h]

        breaks = lv_breaks(tcal, k)
        idx_c = assign_intervals(tcal, breaks)
        idx_t = assign_intervals(ttest, breaks)

        for i in 1:k
            Ical = findall(==(i), idx_c)
            Itest = findall(==(i), idx_t)

            if !isempty(Ical)
                q_i = quantile(rCal[Ical], 1 - α)
                qHistory[i, h] = q_i

                qCal[Ical] .+= w[h] * q_i
                if !isempty(Itest)
                    qTest[Itest] .+= w[h] * q_i
                end
            else
                # If an interval is empty in calibration, fall back to nearest available quantile in this LV
                # (conservative fallback: nearest non-NaN if any)
                if !isempty(Itest)
                    # nearest non-NaN in same LV
                    available = findall(x -> !isnan(x), qHistory[:, h])
                    if isempty(available)
                        # ultimate fallback: global quantile from whole calibration set
                        q_global = quantile(rCal, 1 - α)
                        qTest[Itest] .+= w[h] * q_global
                    else
                        # use closest available interval
                        nearest = available[argmin(abs.(available .- i))]
                        qTest[Itest] .+= w[h] * qHistory[nearest, h]
                    end
                end
            end
        end
    end

    # attach results back to original model
    model[:XTrain_uq] = XTrain
    model[:XCal_uq] = XCal
    model[:YTrain_uq] = reshape(yTrain, :, 1)
    model[:YCal_uq] = reshape(yCal, :, 1)

    model[:yPredCal] = reshape(yPredCal, :, 1)
    model[:yPredTest] = reshape(yPredTest, :, 1)

    model[:qCal] = reshape(qCal, :, 1)
    model[:qTest] = reshape(qTest, :, 1)

    model[:qHistory] = qHistory
    model[:lv_weights] = w

    return model
end

# -----------------------------
# High-level one-call runner
# -----------------------------

"""
run_pipeline(model)

Expected minimum keys:
:dataset  ("GermanyHyper" | "newShiftData" | "Corn")
:method   ("PCR" | "PLS" | "KPCR" | "KPLS")
:dim      (Int)
:ni       (Int) number of LV intervals
:testSize (Float64)
Optional:
:filtering::Bool, :smoothWindow::Int
:selectedBands::Vector{Int} (shift)
:cornTargetCol::Int (corn)
:kernelParams::Vector{Float64} (for kernel methods)
:kernelType, :kernelVersion
:alpha
:standardizeX
"""
function run_pipeline(model::Dict{Symbol,Any})
    rng = get(model, :rng, Random.default_rng())

    X, y = load_raw_data(model)
    X = preprocess(X, model)

    Xtr, ytr, Xte, yte = split_train_test(X, y; testSize=get(model, :testSize, 0.2), rng=rng)

    model[:X] = Xtr
    model[:Y] = reshape(ytr, :, 1)
    model[:Xtest] = Xte
    model[:Ytest] = reshape(yte, :, 1)

    standardize!(model)

    # run conformal-UQ (includes internal split train/cal and model fitting)
    conformal_uq!(model)

    return model
end

 # module

# ============================================================
# Embedded KernelFlow (KF) block — keep KF math unchanged
# ============================================================
module KernelFlow

using Random
using Statistics
using LinearAlgebra
using StatsBase
using Zygote
using CategoricalArrays

# Optional: only if your public datasets need them
using XLSX, DataFrames, MAT

# Keep a module-global model because your KF code uses it implicitly in places.
model = Dict{Symbol,Any}()

# Functions moved from the KF module

# kernelRBF2

function kernelRBF2(X::Matrix{Float64}, params, model::Dict{Symbol,Any}, state)
    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row, col))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-08.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row, col))
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (params[i + noVars] .* 1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-08.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(X, X)
        row, col = size(ND)
        d   = sqrt.(ND + 1e-08.*ones(row,col))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ params[10] * (1. ./ (ones(row, col) + (ND ./ params[9]^2)))

    else
        ND = normDiff(X, X)
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-08.*ones(row,col))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end
    
    if state == "training"
        nl   = size(K, 1)
        oneN = ones(nl, nl) / nl
        #if size(model[:initialParams]) == size(zeros(1))
        if model[:PLS] == "PCR"
            K    = K .- oneN * K .- K * oneN .+ oneN * K * oneN 
        else
            K    = K .- oneN * K .- K * oneN .+ oneN * K * oneN .+ params[end] * I(nl)
        end
    else 
        nl   = size(K, 1)
        if model[:PLS] == "PCR"
        K = K
        else
        K   =  K .+ params[end] * I(nl)
        end
    end
    return K
end

# kernelRBFTest

function kernelRBFTest(X::Matrix{Float64}, XTest::Matrix{Float64}, params, model::Dict{Symbol,Any})

    KCal = kernelRBF2(X, params, model, "test")

    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row, col))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-09.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end

    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row, col))
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + params[i + noVars] .* (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-09.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(XTest, X)
        row, col = size(ND)
        d   = sqrt.(ND + 1e-08.*ones(row,col))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ 1. ./ (ones(row, col) + (ND ./ params[9]^2))

    else
        ND = normDiff(XTest, X)
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-09.*ones(row,col))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end

    n       = size(KCal, 1)
    oneN    = ones(n, n)/n
    nTest   = size(XTest, 1)
    oneNTest = ones(nTest, n)/n
    KTest       = K .- oneNTest * KCal .- K * oneN .+ oneNTest * KCal * oneN 
    return KTest
end

# rho_average

function rho_average(params::Vector{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, model::Dict{Symbol,Any})
    #println(size(Y))
    #println(size(X))

    XBatch  = X[model[:idxBatch], :]
    YBatch  = Y[model[:idxBatch], :]

    loss    = zeros(1,1)

    if model[:loss] == 1
    KBatch  = kernelRBF2(XBatch, params, model, "training")
    weightsBatch = compute_kpls(KBatch, (YBatch .- mean(YBatch)), model[:dim])
 
    fullNorm    = weightsBatch' * KBatch * weightsBatch

    for i = 1:model[:nsamp]

        YSam    = YBatch[Int.(model[:idxSamp][:,i]), :]
        XSam    = XBatch[Int.(model[:idxSamp][:,i]), :]

        KSam    = kernelRBF2(XSam, params, model, "training")
        KCross  = KBatch[:, Int.(model[:idxSamp][:,i])]

        weightsSam = compute_kpls(KSam, (YSam .- mean(YSam)), model[:dim])

        #println(any(isnan.(KSam)))
        #println(any(isnan.(weightsSam)))
        sampleNorm  = weightsSam' * KSam * weightsSam
        
        crossNorm   = weightsBatch' * KCross * weightsSam
        #isnan(KCross)
        lossIter    = 1. + Float64.((sampleNorm[1] - 2.0 * crossNorm[1]) / fullNorm[1])
        
        #println(any(isnan.(lossIter)))
        loss        = hcat(loss,lossIter)

    end

    lossA = sum(loss)/model[:nsamp]
    
    elseif model[:loss] == 2
        rowB, colB = size(XBatch)
        for i = 1:6:rowB
            XIterTest = reshape(XBatch[i,:], 1, :)
            YIterTest = YBatch[i,:]

            if i == 1
                XIterCal  = XBatch[2:end, :]
                YIterCal  = YBatch[2:end, :]
            elseif i == 2
                XIterCal = vcat(XBatch[1:1, :], XBatch[3:end, :])
                YIterCal = vcat(YBatch[1:1, :], YBatch[3:end, :])
            else
                XIterCal  = vcat(XBatch[1:i-1, :], XBatch[i+1:end, :])
                YIterCal  = vcat(YBatch[1:i-1, :], YBatch[i+1:end, :])
            end


            KCal      = kernelRBF2(XIterCal, params, model, "training")
            KTest     = kernelRBFTest(XIterCal, XIterTest, params, model)

            B         = compute_kpls(KCal, YIterCal, model[:dim])
    
            YIterPred = reshape(KTest, 1, :) * B
            YIterTest = reshape(YIterTest, size(YIterPred))
            loss = loss + (YIterPred - YIterTest)*(YIterPred - YIterTest)'
            #println(loss)
        end
        lossA = loss[1,1]
        #println(lossA)
    end

    return lossA
end

#grad_rho

function grad_rho(model::Dict{Symbol, Any})
    params  = model[:initialParams]
    beta    = 0.9
    beta2   = 0.5
    params  = Float64.(model[:initialParams])
    threshold = 1

    model[:np] = size(params,1)
    
    parameterHistory= zeros(Float64, model[:iter], model[:np])
    gradHistory     = zeros(Float64, model[:iter], model[:np])
    lossHistory     = zeros(Float64, model[:iter], 1)

    for i = 1:model[:iter]
        
        rowData = size(model[:X], 1)
        model[:idxBatch] = sort(sample(rowData, model[:sp]))
        model[:idxSamp] = zeros(ceil(Int, length(model[:idxBatch])*model[:sp]), 0)
            
        if i == 1 || mode(i%10) == 0
            println(i)
            
        end

        if model[:loss] == 1
            for i = 1: model[:nsamp]
                model[:idxSamp] = hcat(model[:idxSamp], sort(sample(length(model[:idxBatch]), model[:sp])))
            end
        end

        for j = 1 : model[:np] 
           parameterHistory[i,j] = params[j]
           #println(params[j])
           
        end

        loss, grad = withgradient(param -> rho_average(exp.(param), model[:X], model[:Y], model), params)
        #println(grad)
        grad = 0.6 .* grad./norm(grad)

        lossHistory[i] = loss
        
        for j = 1:model[:np]
            if i > 1
                gradHistory[i,j]      = grad[1][j] 
            else
                gradHistory[i,j]      = 0.0
            end
            if i == 1
                params[j] = params[j] - model[:learnRate] * grad[1][j]
                
            else
                params[j] = params[j] - model[:learnRate] * (grad[1][j] + beta * (params[j] - parameterHistory[i-1,j]))
            end
        end

    end

    return model, parameterHistory, lossHistory, gradHistory
end

# optimize_parameters

function optimize_parameters(model::Dict{Symbol, Any})
    
    model, parameterHistory, lossHistory, gradHistory = grad_rho(model)

    model[:runningLoss] = movmean(lossHistory, 10)
    _, model[:bestLoss] = findmin(model[:runningLoss])
    model[:bestParam] = zeros(model[:np], 1)

    for i = 1:model[:np]
        model[:bestParam][i] = parameterHistory[model[:bestLoss], i]
    end
    return model, parameterHistory, lossHistory, gradHistory
end

# compute_kpls 1 si 2

function compute_kpls(K::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    if model[:PLS] == "NIPALS" || model[:PLS] == "SIMPLS"
        T, U    = compute_pls2(K, Y, dim)
        B       = U * ((T' * K * U))^(-1)*T'*Y;
    elseif model[:PLS] == "RBF"
        T, P, Q, W, U = compute_pls(K, Y, dim)
        B = W * inv(P' * W)* Q'
    elseif model[:PLS] == "PCR"
        ε = 1e-8  # Small regularization constant
        K_reg = K + ε * I
        T, P = compute_pca(K)               # 1. Compute KPCA
        Bsmall = T[:,1:dim]\Y       # 2. Weights in KPC space
        B = P[:,1:dim] * Bsmall     # 3. Weights in K space
    end    

    return B # W is only utilized in conformal inference
end 

function compute_pls2(X::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    m, nx = size(X)
    m, ny = size(Y)
    T = zeros(m, 0) 
    U = zeros(m, 0)
    K = X

    if model[:PLS] == "SIMPLS"

    for i in 1:dim
        t0 = K * Y
        
        t = t0./sqrt(sum(t0 .* t0))
        u = Y * (Y' * t)

        T = hcat(T, t)
        U = hcat(U, u)

        K       = K  - t * (t' * K)
        Y       = Y  - t * (t' * Y)
    end

    elseif model[:PLS] == "NIPALS"
    tol = sqrt(eps())
    maxit = 100

    n, zp = size(X)
    q = size(Y, 2)

    for a = 1:dim
        if q == 1
            t0 = K * Y
            t = t0./sqrt(sum(t0 .* t0))
            c = Y' * t
            u = Y * c
            u2 = u./sqrt(sum(u .* u))
        else
            u = Y[:, 1]
            ztol = 1.0
            iter = 1

            while ztol > tol && iter <= maxit
                t = K * u
                t ./= sqrt(sum(t .* t))
                c = Y' * t
                zu = Y * c
                zu ./= sqrt(sum(zu .* zu))
                ztol = norm(u - zu)
                u = zu
                iter += 1
            end
        end

        z = I - t * t'
        K = z * K * z'
        Y -= t * c'

        T = hcat(T,t)
        #C[:, a] = c
        U = hcat(U,u2)
    end

    
    end
    return T, U
end

function compute_kpls2(K::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    T, U = compute_pls2(K, Y, dim)
    row, col = size(T)

    R = zeros(col,1)
    w = zeros(col,1)
    TSS = sum((K .- mean(K, dims=1)).^2)
    for i = 1:col
        ti = T[:,i]
        #println(ti)
        ip = transpose(ti)*ti
        iv = inv(ip)
        ipp = iv*transpose(ti)
        s5 = ipp * K
        ti = reshape(ti, row, 1)
        s5 = reshape(s5, row, 1)
        Kr =  ti * s5'
        RSS = sum((Kr - K).^2)
        R[i] = 1 - RSS/TSS
        println(R[i])
    end

    for i = 1:col
        w[i] = R[i]/sum(R)
    end

    B = U * inv(T' * K * U) * T' * Y
    return B, T, U, w # W is only utilized in conformal inference
end 

function compute_pca(x::Matrix{Float64})
    m, n = size(x)
    # Check for NaN or infinite values
    if any(isnan, x) || any(isinf, x)
        error("Matrix contains NaN or infinite values")
    end

    if m < n
        p_adj, s, u = svd(x')
    else
        u, s, p_adj = svd(x)
    end
    t = u * Diagonal(s)
    eigenvalues = s .^ 2 
    return t, Matrix(p_adj)
end

function sample(n::Int, sp::Float64)
    ns    = ceil(Int, n*sp)
    perm  = Random.shuffle(1:n)
    idxS = perm[1:ns]
    return idxS
end

function movmean(x, m) 
    # this comes from github.com/chmendoza
    """Compute off-line moving average
    
    Parameters
    ----------
    x (Array):
        One-dimensional array with time-series. size(x)=(N,)
    m (int):
        Length of the window where the mean is computed
    
    Returns
    -------
    y (Array):
        Array with moving mean values. size(y)=(N-m+1)    
    """

    N = length(x)
    y = fill(0., N-m+1)

    y[1] = sum(x[1:m])/m
    for i in 2:N-m+1
        @views y[i] = y[i-1] + (x[i+m-1]-x[i-1])/m
    end
    return y
end


function normDiff(x, y)
    sqx = x .^ 2
    sqy = y .^ 2
    nsx = sum(sqx, dims=2)
    nsy = sum(sqy, dims=2)
    nsx = vec(nsx)
    nsy = vec(nsy)
    innerMat = x * y'
    nD = -2 .* innerMat .+ nsx .+ nsy'
    return nD
end

function zscore(data)
    mu      = mean(data, dims=1)
    sigma   = std(data, dims=1)
    normalized_data = (data .- mu) ./ sigma
    return normalized_data, mu, sigma
end


function normalize(data, mu, sigma)
    return (data .- mu) ./ sigma
end

function moving_average(x, window_size)
    return [mean(x[max(1, i-window_size+1):i]) for i in 1:length(x)]
end


end # module KernelFlow

# ------------------------------------------------------------
# Bridge: run KF optimization and set kernelParams for LVLocalizedUQ
# ------------------------------------------------------------
"""
learn_kernel_params!(model)

Runs embedded KernelFlow.optimize_parameters with the SAME KF operations.
Stores:
- model[:kf_bestParam_log]   (log-space)
- model[:kf_bestParam]       (exp)
- model[:kf_lossHistory]     (raw lossHistory)
Also copies kernel params into model[:kernelParams] for KPCR/KPLS in LVLocalizedUQ.
"""
function learn_kernel_params!(model::Dict{Symbol,Any}; kf=nothing)
    KernelFlow.model = model

    if kf !== nothing
        configure_kf!(model, kf)
    elseif !haskey(model, :initialParams)
        model[:initialParams] = ones(2)
    end

    model, paramHist, lossHist, gradHist = KernelFlow.optimize_parameters(model)

    model[:kf_lossHistory]   = lossHist
    model[:kf_bestParam_log] = model[:bestParam]
    model[:kf_bestParam]     = exp.(vec(model[:bestParam]))

    # handoff into refactored kernel models (KPCR/KPLS)
    model[:kernelParams] = Float64.(model[:kf_bestParam])

    return model
end

"""
configure_kf!(model, kf)

Sets all KF-related model keys BEFORE running KernelFlow.optimize_parameters(model).
Does not modify KF internals.
"""
function configure_kf!(model::Dict{Symbol,Any}, kf)

    model[:iter]      = get(kf, :iter, 500)
    model[:learnRate] = get(kf, :learnRate, 0.2)
    model[:sp]        = get(kf, :sp, 0.3)
    model[:nsamp]     = get(kf, :nsamp, 5)
    model[:loss]      = get(kf, :loss, 2)
    model[:PLS]       = get(kf, :PLS, "SIMPLS")

    # kernel controls (these are exactly what kernelRBF2/kernelRBFTest read)
    model[:kernelType]    = get(kf, :kernelType, get(model, :kernelType, "cauchy"))
    model[:kernelVersion] = get(kf, :kernelVersion, get(model, :kernelVersion, "combined"))

    # determine number of kernel parameters expected by your KF implementation
    p = size(model[:X], 2)
    kv = model[:kernelVersion]

    np =
        kv == "combined"        ? 2 :             # [ℓ, ridge]
        kv == "individual"      ? (p + 1) :       # [ℓ₁..ℓ_p, ridge]
        kv == "individualScale" ? (2p + 1) :      # [ℓ₁..ℓ_p, a₁..a_p, ridge]
        kv == "family"          ? 11 :            # as in your code (includes ridge at end)
        error("Unsupported kernelVersion = $kv")

    # initialization
    init = get(kf, :init, :ones)
    s    = get(kf, :init_scale, 1.0)

    if init == :from_vector
        v = get(kf, :init_vec, nothing)
        @assert v !== nothing "init=:from_vector requires kf.init_vec"
        @assert length(v) == np "init_vec length must be $np for kernelVersion=$kv"
        model[:initialParams] = Float64.(v)

    elseif init == :random
        rng = get(model, :rng, Random.default_rng())
        model[:initialParams] = s .* randn(rng, np)

    else # :ones default
        model[:initialParams] = s .* ones(np)
    end

    return model
end

using Plots
using Statistics

function make_figs(m::Dict{Symbol,Any}; do_kf::Bool=false, title_prefix::String="")
    figs = Dict{Symbol,Any}()

    α = m[:alpha]
    target = 1 - α

    # standardized quantities
    yhat = vec(m[:yPredTest])
    y    = vec(m[:Ytest])
    q    = vec(m[:qTest])

    lo = yhat .- q
    hi = yhat .+ q
    abs_resid = abs.(yhat .- y)

    covered = (y .>= lo) .& (y .<= hi)
    emp_cov = mean(covered)

    # (a) KF running loss (only if present)
    if do_kf && haskey(m, :kf_lossHistory)
        lossHist = vec(m[:kf_lossHistory])
        figs[:kf_loss] = plot(lossHist, lw=2,
                              xlabel="Iteration", ylabel="Loss",
                              title="$(title_prefix)KF running loss",
                              legend=false)
    end

    # (b) qHistory heatmap (intervals x LVs)
    if haskey(m, :qHistory)
        qH = m[:qHistory]
        k, H = size(qH)
        figs[:qHistory] = heatmap(1:H, 1:k, qH,
                                  xlabel="LV (h)", ylabel="Interval (i)",
                                  title="$(title_prefix)qHistory: intervals × LVs",
                                  yflip=false)
    end

    # (c) Predictions with PI bands (sorted by pred)
    ord = sortperm(yhat)
    p_pi = plot(1:length(ord), yhat[ord], label="pred",
                xlabel="Test sample (sorted)", ylabel="y (standardized)",
                title="$(title_prefix)Predictions with $(round(Int,100*target))% PI")
    plot!(p_pi, 1:length(ord), lo[ord], label="PI lower")
    plot!(p_pi, 1:length(ord), hi[ord], label="PI upper")
    scatter!(p_pi, 1:length(ord), y[ord], label="true", ms=3)
    figs[:pi] = p_pi

    # (d) residual vs q with y=x reference
    p_rvq = scatter(q, abs_resid, ms=4,
                    xlabel="Estimated q (PI half-width)",
                    ylabel="|ŷ − y|",
                    title="$(title_prefix)|residual| vs q",
                    legend=false)
    xmin = min(minimum(q), minimum(abs_resid))
    xmax = max(maximum(q), maximum(abs_resid))
    plot!(p_rvq, [xmin, xmax], [xmin, xmax], lw=2, label=false)
    figs[:resid_vs_q] = p_rvq

    # (e) Coverage bar
    figs[:coverage] = bar(["covered","missed"], [sum(covered), sum(.!covered)],
                          ylabel="Count",
                          title="$(title_prefix)Coverage: emp=$(round(emp_cov,digits=3)) target=$(round(target,digits=3))",
                          legend=false)

        # (c2) ytrue vs ypred with PI whiskers (±q around prediction)
    p_sc = scatter(y, yhat;
                   yerror=q,
                   ms=4,
                   xlabel="y true (standardized)",
                   ylabel="y pred (standardized)",
                   title="$(title_prefix)True vs Pred with $(round(Int,100*target))% PI whiskers",
                   label=false)

    # 1:1 reference line
    lo_all = min(minimum(y), minimum(yhat .- q))
    hi_all = max(maximum(y), maximum(yhat .+ q))
    plot!(p_sc, [lo_all, hi_all], [lo_all, hi_all], lw=2, label=false)

    figs[:ytrue_ypred_whiskers] = p_sc
    @show keys(figs)
    return figs
end

end