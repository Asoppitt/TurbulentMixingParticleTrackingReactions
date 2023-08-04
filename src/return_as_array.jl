include("PSP_Particletracking_module.jl")
include("stepwise_funcs.jl")

export PSP_model_no_save_return_moments!

function PSP_model_no_save_return_moments!(turb_k_e::T, nt::Integer, dt::T, np::Integer, initial_condition::Union{String,Tuple{String,Vararg}}, m_params::MotionParams{T}, p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false; saving_rate_moments=1) where T<:AbstractFloat
    nt-=1
    precomp_P = min.(bc_params.bc_k.*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    

    x_pos = zeros(T, np)
    y_pos = zeros(T, np)
    x_pos = space_cells.length_domain.*rand(T, np)
    y_pos = space_cells.height_domain.*rand(T, np)

    if typeof(m_params.u_mean)==T
        ux = randn(T, np).*sqrt.(T(2/3) .*turb_k_e).+m_params.u_mean
        uy = randn(T, np).*sqrt.(T(2/3) .*turb_k_e)
    else
        ux = randn(T, np).*sqrt.(T(2/3) .*turb_k_e).+m_params.u_mean[1].(x_pos,y_pos)
        uy = randn(T, np).*sqrt.(T(2/3) .*turb_k_e).+m_params.u_mean[2].(x_pos,y_pos)
    end

    phip = zeros(T, (2, np,1)) #scalar concentration at these points
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle

    set_phi_as_ic!(phip,initial_condition,x_pos,y_pos,space_cells,1)

    omega0_dist =  make_omega_dist(p_params)#this should now match long term distribution of omega
    omegap = Omega(omega0_dist,np,p_params)
    
    celli= Array{Array{Int,1},2}(undef,space_cells.y_res,space_cells.x_res)
    #assign boundary particles and count cell_particles
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip, cell_particles, cell_particles);
        celli[i,j] = cell_particles;
    return nothing) , x_pos, y_pos, space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed
    t_decorr_p = T(1) ./(p_params.c_t.*omegap[phi_pm[1,:]]).*rand(T,np)
    t_decorr_m = T(1) ./(p_params.c_t.*omegap[phi_pm[2,:]]).*rand(T,np)

    means=zeros(T,2,ceil(Int,nt/saving_rate_moments))
    vars=zeros(T,2,ceil(Int,nt/saving_rate_moments))

    for i in 1:nt
        bc_interact=particle_motion_model_step!(x_pos,y_pos, ux,uy,omegap, turb_k_e, m_params, dt, space_cells, np)
        PSP_model_step!(x_pos,y_pos,phip,celli,omegap, t_decorr_m, t_decorr_p, phi_pm, bc_interact, dt, p_params,space_cells, bc_params,np,precomp_P)
        if i%saving_rate_moments==0
            (means[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:]))#this is higher precision for some reason
            (means[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:]))
            (vars[1,ceil(Int,i/saving_rate_moments)] = var(phip[1,:]))
            (vars[2,ceil(Int,i/saving_rate_moments)] = var(phip[2,:]))
        end
        verbose && print(i,' ')
    end

    return means, vars
end