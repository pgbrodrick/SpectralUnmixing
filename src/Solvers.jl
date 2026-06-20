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

using JuMP
using NLopt
using LinearAlgebra

import NLopt: Opt

# ==============================================================================
# ZERO-ALLOCATION TASK LOCAL STORAGE WORKSPACE
# ==============================================================================

struct UnmixingWorkspace
    opts::Dict{Int, NLopt.Opt}     
    AtA::Matrix{Float64}
    Atb::Vector{Float64}
    Ax::Vector{Float64}
    residual::Vector{Float64}
    AtAx::Vector{Float64}
    lb_bounds::Vector{Float64}     
    ub_bounds::Vector{Float64}     
    d_buffer::Matrix{Float64}      
    
    # Mathematical buffers for Custom Solvers (BVLS, LM, Trust-Region)
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
    lbv::Vector{Bool}
    ubv::Vector{Bool}
    v_mask::Vector{Bool}
    alphas::Vector{Float64}
end

function get_workspace(n_bands::Int, max_endmembers::Int, img_size::Tuple)::UnmixingWorkspace
    if !haskey(task_local_storage(), :unmix_ws)
        opts = Dict{Int, NLopt.Opt}()
        for i in 1:max_endmembers
            opt = NLopt.Opt(:LD_LBFGS, i)
            opt.maxeval = 1000
            opt.ftol_rel = 1e-3
            opts[i] = opt
        end
        
        ws = UnmixingWorkspace(
            opts,
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
            zeros(Float64, n_bands),
            zeros(Bool, max_endmembers),
            zeros(Bool, max_endmembers),
            zeros(Bool, max_endmembers),
            zeros(Float64, max_endmembers)
        )
        task_local_storage(:unmix_ws, ws)
    end
    
    return task_local_storage(:unmix_ws)::UnmixingWorkspace
end

# ==============================================================================
# LINEAR ALGEBRA CORE
# ==============================================================================

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
Zero-allocation least squares solver for BVLS subproblems. Uses normal equations 
and in-place Cholesky factorization.
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
    
    # Apply Tikhonov Regularization (Ridge) to the diagonal
    if lambda > 0.0
        for i in 1:k
            H_view[i, i] += lambda
        end
    end
    
    try
        H_fact = cholesky!(Symmetric(H_view))
        ldiv!(z_view, H_fact, g_view)
    catch
        z_view .= A_view \ b_view
    end
end

# ==============================================================================
# OPTIMIZERS
# ==============================================================================

function nlopt_solve_fast(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x0::AbstractVector{Float64},
                          lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, ws::UnmixingWorkspace;
                          maxeval::Int=1000, ftol_rel::Float64=1e-3)

    n_vars = length(x0)
    m_bands = length(b)

    opt = ws.opts[n_vars]
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.maxeval = maxeval
    opt.ftol_rel = ftol_rel

    AtA_view  = @view ws.AtA[1:n_vars, 1:n_vars]
    Atb_view  = @view ws.Atb[1:n_vars]
    AtAx_view = @view ws.AtAx[1:n_vars]
    Ax_view   = @view ws.Ax[1:m_bands]
    res_view  = @view ws.residual[1:m_bands]

    mul!(AtA_view, A', A)
    mul!(Atb_view, A', b)

    function objective_with_grad!(x::Vector{Float64}, grad::Vector{Float64})
        if length(grad) > 0
            mul!(AtAx_view, AtA_view, x)
            grad .= AtAx_view .- Atb_view
        end
        mul!(Ax_view, A, x)
        res_view .= Ax_view .- b
        return 0.5 * dot(res_view, res_view)
    end

    opt.min_objective = objective_with_grad!
    x_init = clamp.(x0, lb, ub) 
    (minf, minx, ret) = NLopt.optimize(opt, x_init)

    return minx, 2.0 * minf 
end

function nlopt_lbfgs(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x0::AbstractVector{Float64},
                     lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, ws::UnmixingWorkspace)
    return nlopt_solve_fast(A, b, x0, lb, ub, ws, maxeval=1000, ftol_rel=1e-3)
end

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

        # In-place damping and factorization
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

function trust_region_newton(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x0::AbstractVector{Float64},
                             lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, ws::UnmixingWorkspace;
                             Delta0::Float64=1.0, max_iter::Int=1000, tol::Float64=1e-3,
                             eta1::Float64=0.1, eta2::Float64=0.75, gamma1::Float64=0.5, gamma2::Float64=2.0)

    m, n = size(A)
    x = clamp.(x0, lb, ub)
    Delta = Delta0
    
    H = @view ws.AtA[1:n, 1:n]
    mul!(H, A', A) 
    
    # Pre-factorize once!
    H_fact = try cholesky(Symmetric(H)) catch; nothing end

    r = @view ws.residual[1:m]
    Ax = @view ws.Ax[1:m]
    g = @view ws.g[1:n]
    s = @view ws.delta[1:n]
    x_new = @view ws.x_new[1:n]
    Ax_new = @view ws.Ax_new[1:m]
    r_new = @view ws.r_new[1:m]

    mul!(Ax, A, x)
    r .= Ax .- b
    f = 0.5 * dot(r, r)
    mul!(g, A', r)

    for iter in 1:max_iter
        if norm(g) < tol
            break
        end

        s_newton = !isnothing(H_fact) ? -(H_fact \ g) : -(H \ g)
        norm_newton = norm(s_newton)

        if norm_newton <= Delta
            s .= s_newton
        else
            Hg = H * g
            alpha = dot(g, g) / dot(g, Hg)
            s_cauchy = -alpha * g
            norm_cauchy = norm(s_cauchy)

            if norm_cauchy >= Delta
                s .= (Delta / norm_cauchy) * s_cauchy
            else
                diff = s_newton - s_cauchy
                a = dot(diff, diff)
                b_coef = 2 * dot(s_cauchy, diff)
                c = dot(s_cauchy, s_cauchy) - Delta^2

                tau = (-b_coef + sqrt(b_coef^2 - 4*a*c)) / (2*a)
                tau = clamp(tau, 0.0, 1.0)
                s .= s_cauchy .+ tau * diff
            end
        end

        x_new .= clamp.(x .+ s, lb, ub)
        mul!(Ax_new, A, x_new)
        r_new .= Ax_new .- b
        f_new = 0.5 * dot(r_new, r_new)

        actual_red = f - f_new
        Hs = H * s
        predicted_red = -(dot(g, s) + 0.5 * dot(s, Hs))

        rho = abs(predicted_red) < 1e-12 ? 1.0 : actual_red / predicted_red

        if rho < eta1
            Delta *= gamma1
        elseif rho > eta2 && norm(s) ≈ Delta
            Delta *= gamma2
        end

        if rho > eta1
            x .= x_new
            Ax .= Ax_new
            r .= r_new
            f = f_new
            mul!(g, A', r)
        end

        Delta = clamp(Delta, 1e-8, 1e3)
    end

    return x, 2.0 * f
end

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

function bvls(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, x_lsq::AbstractVector{Float64}, 
              lb::AbstractVector{Float64}, ub::AbstractVector{Float64}, 
              tol::Float64, max_iter::Int64, verbose::Int64, inverse_method::String, ws::UnmixingWorkspace;
              lambda::Float64=0.0) # Added lambda keyword argument

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
    
    # Regularized Cost & Gradient
    cost = 0.5 * dot(r, r) + 0.5 * lambda * dot(x, x)
    mul!(g, A', r)
    if lambda > 0.0
        for i in 1:n
            g[i] += lambda * x[i]
        end
    end

    iteration = 0
    
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

        iteration += 1
        free_idx = @view free_set[1:n_free]
        
        x_free_old = @view ws.x_free_old[1:n_free]
        for i in 1:n_free
            x_free_old[i] = x[free_idx[i]]
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
        
        # Pass lambda to the linear solver
        #dolsq_fast!(z_free, A_free, b_free, ws, lambda)
        dolsq(A_free, b_free)

        bound_hit = false
        for i in 1:n_free
            fi = free_idx[i]
            if z_free[i] <= lb[fi]
                x[fi] = lb[fi]
                on_bound[fi] = -1
                bound_hit = true
            elseif z_free[i] >= ub[fi]
                x[fi] = ub[fi]
                on_bound[fi] = 1
                bound_hit = true
            else
                x[fi] = z_free[i]
            end
        end

        mul!(Ax, A, x)
        r .= Ax .- b
        
        # Regularized Cost & Gradient
        cost = 0.5 * dot(r, r) + 0.5 * lambda * dot(x, x)
        mul!(g, A', r)
        if lambda > 0.0
            for i in 1:n
                g[i] += lambda * x[i]
            end
        end

        if !bound_hit
            break
        end
    end

    max_iter = max_iter == -1 ? n : max_iter + iteration
    termination_status = 0

    for iter in iteration:max_iter
        optimality = compute_kkt_optimality_fast!(g_kkt, g, on_bound)
        if optimality < tol
            termination_status = 1
            break
        end

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
            
            # Pass lambda to the linear solver
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
        
        # Regularized Cost & Gradient Update
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