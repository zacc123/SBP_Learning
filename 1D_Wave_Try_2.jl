"""
Attempting to program up the 1D wave equation using the SBP-SAT method's in 

Erickson, B. A. and Dunham, E. M. (2014), An efficient numerical method for earthquake 
cycles in heterogeneous media: Alternating sub-basin and surface-rupturing events on faults 
crossing a sedimentary basin,  Journal of  Geophysical Research, doi:10.1002/2013JB010614.

Refer to notebooks for math and stuff : )
"""

# Imports
using Plots
using LinearAlgebra

# Start with FE
# Same code as used in all the homeworks so 85% it works XD

# Define a forward euler function
function forward_euler!(f, c, h, result, time_mesh)
    # Forward Euler function to find an approximate solution for an ODE
    # INPUTS
        # f is a function fo t and y, f(t, y), for the ODE, y is a vector
        # c is an initial condition vector
            # -> We'll assume its of dim 2(N+1) since we have to keep track of U_1 and U_2
        # h is a step size in time
        # result is our resulting matrix of dim (N+1xdim(y))
        # time_mesh is a 1D array for our time scale with N+1 nodes
    # Output: a resultant vector with the solution

    # Start with initial condition
    result[:, 1] = c
    
    for n = 2:length(time_mesh) - 1
        result[:, n] = result[:, n-1] + h*f(time_mesh[n-1], result[:, n-1])
    end
    
    return result
end


function forward_euler_decoupled!(f_u, f_v, c_u, c_v, h, u_result, v_result, time_mesh, x_mesh=[0])
    # Forward Euler function to find an approximate solution for an ODE
    # Same logic as above but now decouple U and V so we do both separately for my sanity
    # INPUTS
        # f is a function fo t and y, f(t, y), for the ODE, y is a vector
        # c1 is an initial condition vector for u
        # c2 is an initial condition vector for v
            # -> We'll assume its of dim 2(N+1) since we have to keep track of U_1 and U_2
        # h is a step size in time
        # result is our resulting matrix of dim (N+1xdim(y))
        # time_mesh is a 1D array for our time scale with N+1 nodes
    # Output: a resultant vector with the solution

    # Start with initial condition
    u_result[:, 1] = c_u
    v_result[:, 1] = c_v
    
    for n = 2:length(time_mesh) - 1
        u_result[:, n] = u_result[:, n-1] + h*f_u(time_mesh[n-1], u_result[:, n-1], v_result[:, n-1], x_mesh)
        v_result[:, n] = v_result[:, n-1] + h*f_v(time_mesh[n-1], u_result[:, n-1], v_result[:, n-1], x_mesh)
    end
    
    return "All Done :)"
end

# Now setup some initial conditions

# Function to setup the initial U grid
function u_0!(x)
    
    # A few params to play around with
    l = 10
    t = 0
    for i in eachindex(x)
        x[i] = (pi / l)^2 * sin((pi * x[i]) / l + t)
    end

    return x
end

function v_0!(x)
    # Set the initial velocities of the mesh
    # for now set all to 0
    for i in eachindex(x)
        x[i] = 0
    end
end

# Stencil matrix definition
function M_stencil!(Mu, M) 
    m,n = size(Mu)
    # Wtf is this line doing??
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

end

function _f(t, x_mesh)
    # Wtf was I high??? >.<
    # Maybe fixed
    L=10
    # Uses x, not U 
    return [(pi / L)^2 * sin(pi / L * x - t) for x in x_mesh]
end

function u_prime(t, u_vec, v_vec, x_mesh)
    res = zeros(length(v_vec))
    res[:] = v_vec
    return res
end

function main()

    # Time init conditions
    h = 0.0001
    a = 0
    b = 100
    t_mesh = a:h:b

    # Set up some initial conditions for our mesh
    L = 0.1
    x_step = 0.01
    dx = x_step # adding in for debug : )

    x_mesh = 0:x_step:L
    N_u = length(x_mesh) - 1

    # Setup matrices

    # H Matrix
    H = Matrix{Float64}(I, N_u+1, N_u+1) 
    H[1, 1] = .5
    H[N_u+1, N_u+1] = .5
    H = dx .* H # extra dx here

    # Inverse H Matrix
    H_in = H\Matrix{Float64}(I, N_u+1, N_u+1)

    # B Matrix
    B = zeros(N_u+1, N_u+1)
    B[1,1] = -1.0
    B[N_u+1, N_u+1] = 1.0

    # S Matrix
    S = Matrix{Float64}(I, N_u+1, N_u+1)
    S[1,1] = -1.5
    S[1,2] = 2.0
    S[1,3] = -0.5
    S[N_u+1,N_u-1] = 0.5
    S[N_u+1,N_u] = -2.0
    S[N_u+1,N_u+1] = 1.5
    S = S ./ dx # extra dx here


    

    # μ matrix, can add in a function to set non-homogeneous case
    μ = Matrix{Float64}(I, N_u+1, N_u+1)

    # M stencil
    M = zeros(N_u+1, N_u+1)
    M_stencil!(μ, M)
    M = dx^2 .* M

    # start and end filters
    e_n = zeros(N_u+1, 1)
    e_n[N_u+1,1] = 1

    e_0 = zeros(N_u+1, 1)
    e_0[1,1] = 1    

    # Actual operator
    D = H_in * (-1* M + (μ)*B*S)

    # Constants for SAT terms
    α0 = -1
    α1 = -13/dx
    β = 1


    # Ok now we pray
    # Time init conditions
    N_t = length(t_mesh) - 1

    # Set up some initial conditions for our mesh
    c_u = zeros(N_u+1)
    c_u[:] = 0:x_step:L
    u_0!(c_u)

    c_v = zeros(N_u+1)
    v_0!(c_v) #redundant but ok

    u_res = zeros(N_u+1, N_t+1)
    v_res = zeros(N_u+1, N_t+1)



    # SAT Terms:
        # Using c_u as u(0)
    
    p_l =  (α0 * H_in * e_0) * (μ*B*S*c_u)[1] # - g(l)
    p_r = α1 * μ[N_u+1] * H_in * e_n + β*H_in*transpose(μ*B*S)*e_n *c_u[N_u+1]

    function sbp_function!(t, u_vec, v_vec, x_mesh)
    
        # Idea: Pass in only the velocities
            # Decouple the U and V updates bc im losing my mind

        # Setup the returns
        N = length(v_vec)
        res = zeros(N)
        # return only the vector update
        res[:] =  D*v_vec + p_r + p_l+  _f(t, x_mesh) 
    end

    #forward_euler!(sbp_function!, c, h, res_mat, t_mesh)
    # forward_euler_decoupled!(f_u, f_v, c_u, c_v, h, u_result, v_result, time_mesh, x_mesh=[0])
    # forward_euler_decoupled!(f_u=u_prime, f_v=sbp_function!, c_u=c_u, c_v=c_v, h=h, u_result=u_res, v_result=v_res, time_mesh=t_mesh, x_mesh=x_mesh)
    forward_euler_decoupled!(u_prime, sbp_function!, c_u, c_v, h, u_res, v_res, t_mesh, x_mesh)
    print("Made it to plot")
    # println("U res size: $(size(u_res)), $(size(t_mesh))")
    plot(t_mesh, u_res[1, :])
    xlabel!("TIME")
    ylabel!("X_0")
    png("U_0_Plot")
    plot(t_mesh, v_res[1, :])
    xlabel!("TIME")
    ylabel!("X_0")
    png("V_0_Plot")

    plot(t_mesh, u_res[5, :])
    xlabel!("TIME")
    ylabel!("X_10")
    png("U_5_Plot")
    plot(t_mesh, v_res[5, :])
    xlabel!("TIME")
    ylabel!("X_10")
    png("V_5_Plot")

    plot(t_mesh, u_res[11, :])
    xlabel!("TIME")
    ylabel!("X_10")
    png("U_11_Plot")
    plot(t_mesh, v_res[11, :])
    xlabel!("TIME")
    ylabel!("X_10")
    png("V_11_Plot")

end

main()
