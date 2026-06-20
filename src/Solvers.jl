#  Copyright 2022 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Philip G. Brodrick, philip.g.brodrick@jpl.nasa.gov

using LinearAlgebra

# ==============================================================================
# ZERO-ALLOCATION TASK LOCAL STORAGE WORKSPACE
# ==============================================================================

"""
    UnmixingWorkspace

A Task Local Storage (TLS) workspace containing pre-allocated arrays and buffers 
for zero-allocation execution of optimization algorithms. 

This struct prevents garbage collection overhead during massively parallel 
hyperspectral unmixing runs by providing dynamically sized mathematical workspaces 
unique to each worker thread/process.
"""
struct UnmixingWorkspace
    AtA::Matrix{Float64}
    Atb::Vector{Float64}
    Ax::Vector{Float64}
    residual::Vector{Float64}
    AtAx::Vector{Float64}
    lb_bounds::Vector{Float64}     
    ub_bounds::Vector{Float64}     
    d_buffer::Matrix{Float64}      
    
    # Mathematical buffers for Custom Solvers (BVLS, LM)
    g::Vector{Float64}
    H_work::Matrix{Float64}
    delta::Vector{Float64}
    x_new::Vector{Float64}
    r_new::Vector{Float64}
    Ax_new::Vector{Float64}

    # Active-Set specific buffers (BVLS)
    on_bound::Vector{Int}
    active_set::Vector{Bool}
    free_set::Vector{Int}
    x_free::Vector{Float64}
    x_free_old::Vector{Float64}
    z::Vector{Float64}
    b_free::Vector{Float64}
end

"""
    get_workspace(n_bands::Int, max_endmembers::Int, img_size::Tuple)

Retrieves the worker-local `UnmixingWorkspace` from Task Local Storage. 

If the workspace does not exist on the current worker thread/process, it allocates 
it exactly once and reuses the memory dynamically for all subsequent calls.

# Arguments
- `n_bands::Int`: The number of wavelength bands in the hyperspectral data.
- `max_endmembers::Int`: The maximum number of endmembers that will be evaluated in a single combination.
- `img_size::Tuple`: The dimensions of the image being processed.

# Returns
- `UnmixingWorkspace`: The zero-allocation workspace struct for the current worker.
"""
function get_workspace(n_bands::Int, max_endmembers::Int, img_size::Tuple)::UnmixingWorkspace
    if !haskey(task_local_storage(), :unmix_ws)
        ws = UnmixingWorkspace(
            zeros(Float64, max_endmembers, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, n_bands),
            zeros(Float64, n_bands),
            zeros(Float64, max_endmembers),
            zeros(Float64, max_endmembers),
            ones(Float64, max_endmembers),
            zeros(Float64, img_size),
            
            # Custom Math Buffers
            zeros(Float64, max_endmembers),
            zeros(Float64, max_endmembers, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, n_bands),
            zeros(Float64, n_bands),
            
            # BVLS Buffers
            zeros(Int, max_endmembers),
            zeros(Bool, max_endmembers),
            zeros(Int, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, max_endmembers),
            zeros(Float64, n_bands)
        )
        task_local_storage(:unmix_ws, ws)
    end
    
    return task_local_storage(:unmix_ws)::UnmixingWorkspace
end

# ==============================================================================
# LINEAR ALGEBRA CORE
# ==============================================================================

"""
    dolsq(A, b; method::String="default")

Solve an unconstrained linear least squares problem, finding the vector `x` that 
minimizes the residual ||Ax - b||².

# Arguments
- `A`: Coefficient matrix of size (m, n).
- `b`: Target vector of size (m,).
- `method::String`: Solver method to use. Options are `"default"` (backslash operator), 
  `"pinv"` (pseudoinverse), or `"qr"` (QR decomposition).

# Returns
- `x`: Vector of size (n,) that minimizes the least squares residual.
"""
function dolsq(A, b; method::String="default")
    if method == "default"
        x = A \ b
    elseif method == "pinv"
        x = pinv(A) * b
    elseif method == "qr"
        qrA = qr(A)
        x = qrA \ b
    end
    return x
end

"""
    dolsq_fast!(z_view::AbstractVector, A_view::AbstractMatrix, b_view::AbstractVector, 
                ws::UnmixingWorkspace, lambda::Float64=0.0)

Zero-allocation linear least squares solver designed specifically for BVLS active-set 
subproblems. Uses normal equations and an in-place Cholesky factorization.

# Arguments
- `z_view::AbstractVector`: Pre-allocated output view to store the solution vector.
- `A_view::AbstractMatrix`: View of the active coefficient matrix (A_free).
- `b_view::AbstractVector`: View of the active target vector (b_free).
- `ws::UnmixingWorkspace`: The worker-local pre-allocated memory workspace.
- `lambda::Float64`: Tikhonov regularization (Ridge) penalty parameter (default: 0.0).

# Returns
- Modifies `z_view` in-place. Returns nothing.
"""
function dolsq_fast!(z_view::AbstractVector, A_view::AbstractMatrix, b_view::AbstractVector, ws::UnmixingWorkspace, lambda::Float64=0.0)
    k = size(A_view, 2)
    if k == 0
        return
    end

    H_view = @view ws.H_work[1:k, 1:k]
    g_view = @view ws.Atb[1:k] 
    
    # (A'A)z = A'b
    mul!(H_view, A_view', A_view)
    mul!(g_view, A_view', b_view)
    
    if lambda > 0.0
        for i in 1:k
            H_view[i, i] += lambda
        end
    end
    
    try
        # Fast, zero-allocation path
        H_fact = cholesky!(Symmetric(H_view))
        ldiv!(z_view, H_fact, g_view)
    catch
        # Fallback if matrix is numerically singular
        z_view .= A_view \ b_view
    end
end

"""
    compute_kkt_optimality_fast!(g_kkt::AbstractVector, g::AbstractVector, on_bound::AbstractVector)

Computes the Karush-Kuhn-Tucker (KKT) optimality condition value for a given gradient 
vector and boundary status without allocating memory.

# Arguments
- `g_kkt::AbstractVector`: Pre-allocated view to store the resulting KKT values.
- `g::AbstractVector`: The gradient of the objective function.
- `on_bound::AbstractVector`: Indicator vector for variable status (-1 for lower bound, 
  1 for upper bound, 0 for free).

# Returns
- `max_kkt::Float64`: The maximum KKT condition value across all variables.
"""
function compute_kkt_optimality_fast!(g_kkt::AbstractVector, g::AbstractVector, on_bound::AbstractVector)
    n = length(g)
    max_kkt = 0.0
    for i in 1:n
        val = on_bound[i] == 0 ? abs(g[i]) : g[i] * on_bound[i]
        g_kkt[i] = val
        if val > max_kkt
            max_kkt = val
        end
    end
    return max_kkt
end

# ==============================================================================
# OPTIMIZERS
# ==============================================================================

"""
    levenberg_marquardt(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, 
                        x0::AbstractVector{Float64}, lb::AbstractVector{Float64}, 
                        ub::AbstractVector{Float64}, ws::UnmixingWorkspace;
                        lambda0::Float64=1e-3, lambda_up::Float64=10.0, 
                        lambda_down::Float64=0.1, max_iter::Int=1000, tol::Float64=1e-3)

Bounded Levenberg-Marquardt damped least squares solver for ||Ax - b||². 
Utilizes the zero-allocation workspace for rapid execution.

# Arguments
- `A`: Coefficient matrix (m × n).
- `b`: Target vector (m,).
- `x0`: Initial guess (n,), which will be clamped to bounds.
- `lb`, `ub`: Lower and upper bounds (n,).
- `ws::UnmixingWorkspace`: Pre-allocated Task Local Storage workspace.
- `lambda0::Float64`: Initial damping parameter (default: 1e-3).
- `lambda_up::Float64`: Factor to increase λ on rejection (default: 10.0).
- `lambda_down::Float64`: Factor to decrease λ on acceptance (default: 0.1).
- `max_iter::Int`: Maximum number of iterations allowed (default: 1000).
- `tol::Float64`: Convergence tolerance on the gradient norm (default: 1e-3).

# Returns
- `x`: Optimized solution vector.
- `cost`: Final cost ||Ax - b||² (including damping penalty).
"""
function levenberg_marquardt(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x0::AbstractVector{Float64},
                             lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, ws::UnmixingWorkspace;
                             lambda0::Float64=1e-3, lambda_up::Float64=10.0, lambda_down::Float64=0.1, max_iter::Int=1000, tol::Float64=1e-3)

    m, n = size(A)
    x = clamp.(x0, lb, ub)
    lambda = lambda0

    # Views
    r = @view ws.residual[1:m]
    Ax = @view ws.Ax[1:m]
    g = @view ws.g[1:n]
    delta = @view ws.delta[1:n]
    x_new = @view ws.x_new[1:n]
    Ax_new = @view ws.Ax_new[1:m]
    r_new = @view ws.r_new[1:m]
    H = @view ws.H_work[1:n, 1:n]
    JtJ = @view ws.AtA[1:n, 1:n]

    mul!(Ax, A, x)
    r .= Ax .- b
    cost = 0.5 * dot(r, r)
    mul!(JtJ, A', A)

    for iter in 1:max_iter
        mul!(g, A', r)
        if norm(g) < tol
            break
        end

        H .= JtJ
        for i in 1:n
            H[i,i] += lambda
        end
        
        try
            delta .= -(cholesky!(Symmetric(H)) \ g)
        catch
            delta .= -(H \ g)
        end
        
        x_new .= clamp.(x .+ delta, lb, ub)

        mul!(Ax_new, A, x_new)
        r_new .= Ax_new .- b
        cost_new = 0.5 * dot(r_new, r_new)

        if cost_new < cost
            x .= x_new
            Ax .= Ax_new
            r .= r_new
            cost = cost_new
            lambda *= lambda_down
            if norm(delta) < tol
                break
            end
        else
            lambda *= lambda_up
            if lambda > 1e10
                break
            end
        end
    end

    return x, 2.0 * cost 
end

"""
    bvls(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x_lsq::AbstractVector{Float64}, 
         lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, tol::Float64, 
         max_iter::Int64, verbose::Int64, inverse_method::String, ws::UnmixingWorkspace;
         lambda::Float64=0.0)

Solve a bounded variable least squares problem, finding the vector `x` subject to lower (`lb`)
and upper (`ub`) bounds that minimizes the residual ||Ax - b||². 

See https://www.stat.berkeley.edu/~stark/Preprints/bvls.pdf for details.

This implementation uses an active-set (Lawson-Hanson style) method. It evaluates true KKT 
optimality conditions to definitively pin variables to bounds. This version is completely 
allocation-free, utilizing the `UnmixingWorkspace` and in-place subproblem solvers.

# Arguments
- `A`: Coefficient matrix of size (m, n).
- `b`: Target vector of size (m,).
- `x_lsq`: Initial guess vector of size (n,) for the optimization variables.
- `lb`: Vector of size (n,) specifying the lower bounds.
- `ub`: Vector of size (n,) specifying the upper bounds.
- `tol::Float64`: Tolerance for convergence based on the maximum KKT condition.
- `max_iter::Int64`: Maximum number of iterations. If -1, defaults to 5n.
- `ws::UnmixingWorkspace`: Pre-allocated Task Local Storage workspace.
- `lambda::Float64`: Tikhonov regularization (Ridge) parameter added to the normal 
  equations to stabilize collinear endmembers (default: 0.0).

# Returns
- `x`: Vector of size (n,) that minimizes the least squares residual.
- `cost`: The final cost function value of the minimized sum of squared residuals.
"""
function bvls(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x_lsq::AbstractVector{Float64}, 
              lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, 
              tol::Float64, max_iter::Int64, ws::UnmixingWorkspace;
              lambda::Float64=0.0)

    m, n = size(A)
    x = copy(x_lsq)

    # Views into workspace
    on_bound = @view ws.on_bound[1:n]
    active_set = @view ws.active_set[1:n]
    free_set = @view ws.free_set[1:n]
    r = @view ws.residual[1:m]
    g = @view ws.g[1:n]
    Ax = @view ws.Ax[1:m]
    b_free = @view ws.b_free[1:m]
    g_kkt = @view ws.AtAx[1:n] 

    on_bound .= 0
    for i in 1:n
        if x[i] <= lb[i]
            x[i] = lb[i]
            on_bound[i] = -1
        elseif x[i] >= ub[i]
            x[i] = ub[i]
            on_bound[i] = 1
        end
    end

    mul!(Ax, A, x)
    r .= Ax .- b
    cost = 0.5 * dot(r, r) + 0.5 * lambda * dot(x, x)
    mul!(g, A', r)
    if lambda > 0.0
        for i in 1:n
            g[i] += lambda * x[i]
        end
    end

    max_iter = max_iter == -1 ? n * 5 : max_iter + n
    termination_status = 0

    for iter in 1:max_iter
        optimality = compute_kkt_optimality_fast!(g_kkt, g, on_bound)
        if optimality < tol
            termination_status = 1
            break
        end

        # Find best constrained variable to release
        max_val = -Inf
        move_to_free = 1
        for i in 1:n
            val = g[i] * on_bound[i]
            if val > max_val
                max_val = val
                move_to_free = i
            end
        end
        on_bound[move_to_free] = 0

        while true
            n_free = 0
            for i in 1:n
                if on_bound[i] == 0
                    n_free += 1
                    free_set[n_free] = i
                    active_set[i] = false
                else
                    active_set[i] = true
                end
            end

            if n_free == 0
                break
            end

            free_idx = @view free_set[1:n_free]
            x_free = @view ws.x_free[1:n_free]
            for i in 1:n_free
                x_free[i] = x[free_idx[i]]
            end

            A_free = @view A[:, free_idx]
            b_free .= b
            for i in 1:n
                if active_set[i]
                    for j in 1:m
                        b_free[j] -= A[j, i] * x[i]
                    end
                end
            end

            z_free = @view ws.z[1:n_free]
            dolsq_fast!(z_free, A_free, b_free, ws, lambda)

            min_alpha = Inf
            i_free_limit = -1
            bound_type = 0

            for i in 1:n_free
                fi = free_idx[i]
                if z_free[i] < lb[fi]
                    alpha = (lb[fi] - x_free[i]) / (z_free[i] - x_free[i])
                    if alpha < min_alpha
                        min_alpha = alpha
                        i_free_limit = fi
                        bound_type = -1
                    end
                elseif z_free[i] > ub[fi]
                    alpha = (ub[fi] - x_free[i]) / (z_free[i] - x_free[i])
                    if alpha < min_alpha
                        min_alpha = alpha
                        i_free_limit = fi
                        bound_type = 1
                    end
                end
            end

            if i_free_limit != -1
                for i in 1:n_free
                    x[free_idx[i]] = x_free[i] + min_alpha * (z_free[i] - x_free[i])
                end
                on_bound[i_free_limit] = bound_type
            else
                for i in 1:n_free
                    x[free_idx[i]] = z_free[i]
                end
                break
            end
        end

        mul!(Ax, A, x)
        r .= Ax .- b
        cost_new = 0.5 * dot(r, r) + 0.5 * lambda * dot(x, x)
        
        if (cost - cost_new) < tol * cost
            termination_status = 2
        end
        cost = cost_new
        mul!(g, A', r)
        if lambda > 0.0
            for i in 1:n
                g[i] += lambda * x[i]
            end
        end
    end

    x[x .< 1e-5] .= 0
    return x, cost
end