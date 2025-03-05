"""
Ok we're doing this this time
"""

using LinearAlgebra
using Plots

# Define a forward euler function
function forward_euler!(f, c, h, result, time_mesh, params)
    # Forward Euler function to find an approximate solution for an ODE
    # INPUTS
        # f is a function fo t and y, f(t, y), for the ODE, y is a vector
        # c is an initial condition vector
            # -> We'll assume its of dim 2(N+1) since we have to keep track of U_1 and U_2
        # h is a step size in time
        # result is our resulting matrix of dim (N+1xdim(y))
        # time_mesh is a 1D array for our time scale with N+1 nodes
        # params is anything needed in f
    # Output: a resultant vector with the solution

    # Start with initial condition
    result[:, 1] = c

    # Move through time with FE steps
    for n = 2:length(time_mesh) - 1
        result[:, n] = result[:, n-1] + h*f(time_mesh[n-1], result[:, n-1], params)
    end
    
    return nothing
end

# Stencil maker for M
function M_stencil!(Mu, M) 
    m,n = size(Mu)
        # Still open # Questions! What about edges??
        # TO DO, ask Brit about that
        # For now just set to 1 for N+1 and 1 rows
    M[1, 1] = Mu[1,1]
    M[m, m] = Mu[m, m]
    
    # Go through and stencil the remaining rows on the diagonal
    for i in 2:m-1
        M[i, i - 1] = -0.5*(Mu[i - 1, i - 1] + Mu[i, i])
        M[i, i] = 0.5*(Mu[i - 1, i - 1] + Mu[i+1, i+1]) + Mu[i, i]
        M[i, i + 1] = -0.5*(Mu[i, i] + Mu[i+1, i+1])
    end

    return nothing

end

# Boundary condition
function g(t, x, L)
    c = pi / L 
    return  c^2 * sin(c*x - t)
end

function source_term(t, x_mesh, L)
    c = pi / L 
    return [c^2* sin(c*x - t) for x in x_mesh]
end

function u_0!(t, u_vec, x_mesh, L)
    # Function to set intial u
    c = pi / L 
    for i in eachindex(u_vec)
        u_vec[i] = sin(c*x_mesh[i])
    end
    return nothing
end

function v_0!(t, v_vec, x_mesh, L)
    # Function to set intial v
    for i in eachindex(v_vec)
        v_vec[i] = 0
    end
    return nothing
end

function p_l(t, H_inv, e_0, Mu, B, S, u_vec, x_mesh)
    # Bad for now but have it make and return a vector
    n = length(u_vec)
    res = zeros(n)
    alpha_0 = -1 # constants from E&D paper
    res[:] = (alpha_0 * H_inv * e_0) .* ( (Mu * B * S * u_vec)[1] - g(t, 0, x_mesh[n]))
    return res
end

function p_r(t, dx, H_inv, e_n, Mu, B, S, u_vec, x_mesh)
    # Bad for now but have it make and return a vector
    n = length(u_vec)
    res = zeros(n)
    alpha_1 = -13/dx # constants from E&D paper
    beta = 1
    res[:] = (alpha_1 * Mu[n, n] * H_inv * e_n) * (u_vec[n] - g(t, x_mesh[n], x_mesh[n]) ) +  beta .* (H_inv * transpose(Mu*B*S) * e_n * (u_vec[n] - g(t, x_mesh[n], x_mesh[n])))
    return res
end

function sbp_sat(t, dx, H_inv, e_0, e_n, Mu, B, S, D, u_vec, x_mesh)
    # Full SBP SAT term expression for E&D paper, 
        # TODO source term is still a bit wonk, but not affecting final answer much rn.... sus
    n = length(u_vec)
    res = zeros(n)
    sbp = D*u_vec
    sat_r = p_r(t, dx, H_inv, e_n, Mu, B, S, u_vec, x_mesh)
    sat_l = p_l(t, H_inv, e_0, Mu, B, S, u_vec, x_mesh)
    res[:] = sbp + sat_r + sat_l + source_term(t, x_mesh, x_mesh[n])
    return res
end

function f(t, y_vec, params)
    # Finally definining the total right hand side. Assume that y_vec is u and v stacked -> vcat(u, v)
    
    dx, H_inv, e_0, e_n, Mu, B, S, D, N_x, x_mesh = params # unpack all the params
    
    # separate the vectors
    u = y_vec[1:N_x+1] 
    v = y_vec[N_x+2: 2*N_x+2]

    y_res = zeros(2*N_x+2)

    # start with basic return for u' = v
    y_res[1:N_x+1] = v

    # now send in for V
    y_res[N_x+2:2*N_x+2] = sbp_sat(t, dx, H_inv, e_0, e_n, Mu, B, S, D, u, x_mesh)
    return y_res
end

function main()
    
    # Define our 1D Mesh
    dx = 0.1 # Step size between grid points
    L = 2 # Length of grid starting at 0
    x_mesh = 0:dx:L
    N_x = length(x_mesh) - 1 # Again follow convention N = num grid points - 1

    # Define our time scales: We'll go with how many steps and go from there

    a = 0 # start
    b = 10 # end
    N_t = 1000000 # How many time steps (N + 1 total nodes)
    dt = (b - a) / N_t
    time_mesh = a:dt:b
    # print(N_t+1,"\n", length(time_mesh))

    # setup out initial U and V results
    u_vec = zeros(N_x + 1)
    u_0!(0, u_vec, x_mesh, L)
    
    v_vec = zeros(N_x+ 1)
    v_0!(0, v_vec, x_mesh, L)

    ################################################
    # Now we go through and setup our matrices and SBP terms
    # H Matrix
    H = Matrix{Float64}(I, N_x+1, N_x+1)
    H[1, 1] = .5
    H[N_x+1, N_x+1] = .5
    H = dx .* H # set H proportional to dx
    
    # Inverse H Matrix
    H_in = H\Matrix{Float64}(I, N_x+1, N_x+1)

    # B Matrix
    B = zeros(N_x+1, N_x+1)
    B[1,1] = -1.0
    B[N_x+1, N_x+1] = 1.0

    # S Matrix
    S = Matrix{Float64}(I, N_x+1, N_x+1)
    S[1,1] = -1.5
    S[1,2] = 2.0
    S[1,3] = -0.5
    S[N_x+1,N_x-1] = 0.5
    S[N_x+1,N_x] = -2.0
    S[N_x+1,N_x+1] = 1.5
    S = (1/dx) .* S # Make the BS term prop to 1/dx

    Mu = Matrix{Float64}(I, N_x+1, N_x+1) # start with all Mu entries being 1, can add heterogeneity later
    
    M = zeros(N_x+1, N_x+1)
    M_stencil!(Mu, M) 
    M = (1/dx) .* M # Make BS term prop to 1/dx

    # start and end filters
    e_n = zeros(N_x+1, 1)
    e_n[N_x+1,1] = 1

    e_0 = zeros(N_x+1, 1)
    e_0[1,1] = 1    

    # Actual 2nd Derivative operator
    D = H_in * ((-1 .* M) + (Mu*B*S))
    # print(D)
    ################################################
    # OK get ready AHHHHH

    # Setting initial conditions and results
    c = vcat(u_vec, v_vec)
    result = zeros(2*N_x + 2, N_t + 1) 
    params = [dx, H_in, e_0, e_n, Mu, B, S, D, N_x, x_mesh]

    forward_euler!(f, c, dt, result, time_mesh, params)


    plot(time_mesh, result[1, :], label="U")
    plot!(time_mesh, result[1+N_x+1, :], label="V")
    xlabel!("Time")
    ylabel!("Displacement and Velocity Overlay")
    title!("Overlay of U and V vs Time at X = 0")
    png("T4, X0")

    plot(time_mesh, result[11, :], label="U")
    plot!(time_mesh, result[11+N_x+1, :], label="V")
    xlabel!("Time")
    ylabel!("Displacement and Velocity Overlay")
    title!("Overlay of U and V vs Time at X = 10")
    png("T4, X10")

    plot(time_mesh, result[21, :], label="U")
    plot!(time_mesh, result[21+N_x+1, :], label="V")
    xlabel!("Time")
    ylabel!("Displacement and Velocity Overlay")
    title!("Overlay of U and V vs Time at X = 20")
    png("T4, X20")

    plot(x_mesh, result[1:N_x+1, 1], label="U at t=$(time_mesh[1])")
    for i in eachindex(time_mesh)
        if i % 100000 == 0
            plot!(x_mesh, result[1:N_x+1, i+1], label="U at t=$(time_mesh[i+1])")
        end
    end
    png("U Overlay")
    return nothing
end

main()
