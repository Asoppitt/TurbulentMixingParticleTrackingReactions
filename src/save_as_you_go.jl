# include("PSP_Particletracking_module.jl")
# include("stepwise_funcs.jl")
# module ContinuousSaving
export no_psp_motion_model!, PSP_model!



function PSP_model!(foldername::String,turb_k_e::T, nt::Integer, dt::T, np::Integer, initial_condition::Union{String,Tuple{String,Vararg}}, m_params::MotionParams{T}, p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false, chunk_length::Integer=50; record_moments=false, saving_rate=1, saving_rate_moments=saving_rate) where T<:AbstractFloat
    nt-=1
    if chunk_length%saving_rate!=0 || chunk_length%saving_rate!=0
        local rate_lcm = lcm(saving_rate,saving_rate_moments)
        @warn "setting chunk_length to multiple of saving_rates:" chunk_length = rate_lcm*ceil(Int,chunk_length)
    end
    n_chunks=floor(Int, nt/chunk_length)
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

    f_phi=zeros(T,psi_mesh.psi_partions_num_1, psi_mesh.psi_partions_num_2, space_cells.y_res, space_cells.x_res, ceil(Int,chunk_length/saving_rate))

    set_phi_as_ic!(phip,initial_condition,x_pos,y_pos,space_cells,1)
    assign_f_phi!(f_phi,phip, x_pos, y_pos, psi_mesh, space_cells,1)

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

    if record_moments
        means=zeros(T,2,ceil(Int,chunk_length/saving_rate_moments))
        mom_2=zeros(T,2,ceil(Int,chunk_length/saving_rate_moments))
    end

    for chunk=0:n_chunks-1
        for (i,t) in enumerate((chunk*chunk_length+1):((chunk+1)*chunk_length))
            bc_interact=particle_motion_model_step!(x_pos,y_pos, ux,uy,omegap, turb_k_e, m_params, dt, space_cells, np)
            PSP_model_step!(x_pos,y_pos,phip,celli,omegap, t_decorr_m, t_decorr_p, phi_pm, bc_interact, dt, p_params,space_cells, bc_params,np,precomp_P)
            if t%saving_rate==0
                for (ind, cell_parts) in pairs(celli)#pariticle-cell pairs are already defined, so use them for f_phi
                    assign_f_phi_cell!(f_phi,phip[:,cell_parts], psi_mesh, ind[1],ind[2],ceil(Int,i/saving_rate))
                end
            end
            if record_moments && t%saving_rate_moments==0
                (means[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:]))#this is higher precision for some reason
                (means[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:]))
                (mom_2[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:].^2))
                (mom_2[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:].^2))
            end
            verbose && print(t,' ')
        end
        write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"data",f_phi)
        write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"array_shape",[i for i in size(f_phi)])
        if record_moments
            write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"mean",means)
            write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"2nd_moment",mom_2)
        end
        write(foldername*'/'*"total_shape", [chunk+1,chunk_length,floor(Int,((chunk+1)*chunk_length)),false,record_moments,saving_rate] )
        verbose && println('\n',"saved steps: "*string(chunk*chunk_length+1)*" to "*string((chunk+1)*chunk_length))
    end
    if (n_chunks)*chunk_length < nt
        f_phi=zeros(T,psi_mesh.psi_partions_num_1, psi_mesh.psi_partions_num_2, space_cells.y_res, space_cells.x_res, ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate) )
        if record_moments
            means=zeros(T,2,ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate_moments))
            mom_2=zeros(T,2,ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate_moments))
        end
        for (i,t) in enumerate(((n_chunks)*chunk_length+1):nt)
            bc_interact=particle_motion_model_step!(x_pos,y_pos, ux,uy, omegap, turb_k_e, m_params, dt, space_cells, np)
            PSP_model_step!(x_pos,y_pos,phip,celli,omegap,t_decorr_p, t_decorr_m, phi_pm, bc_interact, dt, p_params,space_cells, bc_params,np,precomp_P)
            if t%saving_rate==0
                for (ind, cell_parts) in pairs(celli)#pariticle-cell pairs are already defined, so use them for f_phi
                    assign_f_phi_cell!(f_phi,phip[:,cell_parts], psi_mesh, ind[1],ind[2],ceil(Int,i/saving_rate))
                end
            end
            if record_moments && t%saving_rate_moments==0
                (means[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:]))
                (means[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:]))
                (mom_2[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:].^2))
                (mom_2[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:].^2))
            end
            verbose && print(t,' ')
        end
        write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"data",f_phi)
        write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"array_shape",[i for i in size(f_phi)])
        if record_moments
            write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"mean",means)
            write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"2nd_moment",mom_2)
        end
        write(foldername*'/'*"total_shape", [(n_chunks+1),chunk_length,floor(Int,nt),true, record_moments,saving_rate] )
        verbose && println('\n',"saved steps: "*string(chunk_length*(n_chunks+1)+1)*" to "*string(nt))
    end
    verbose && println("end")
    return nothing
end

function no_psp_motion_model!(foldername::String,turb_k_e::T, nt::Integer, dt::T, np::Integer, initial_condition::Union{String,Tuple{String,Vararg}}, m_params::MotionParams{T}, p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false, chunk_length::Integer=50; record_moments=false, saving_rate=1, saving_rate_moments=saving_rate) where T<:AbstractFloat
    nt-=1
    n_chunks=floor(Int, nt/chunk_length)
 
    ux = randn(T, np).*sqrt.(T(2/3) .*turb_k_e).+m_params.u_mean
    uy = randn(T, np).*sqrt.(T(2/3) .*turb_k_e)
    x_pos = zeros(T, np)
    y_pos = zeros(T, np)
    x_pos = space_cells.length_domain.*rand(T, np)
    y_pos = space_cells.height_domain.*rand(T, np)

    precomp_P = min.(bc_params.bc_k.*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    
    phip = zeros(T, (2, np,1)) #scalar concentration at these points

    f_phi=zeros(T,psi_mesh.psi_partions_num_1, psi_mesh.psi_partions_num_2, space_cells.y_res, space_cells.x_res, ceil(Int,chunk_length/saving_rate))

    omega0_dist =  make_omega_dist(p_params)#this should now match long term distribution of omega
    omegap = Omega(omega0_dist,np,p_params)

    set_phi_as_ic!(phip,initial_condition,x_pos,y_pos,space_cells,1)
    assign_f_phi!(f_phi,phip, x_pos, y_pos, psi_mesh, space_cells,1)

    if record_moments
        means=zeros(T,2,ceil(Int,chunk_length/saving_rate_moments))
        mom_2=zeros(T,2,ceil(Int,chunk_length/saving_rate_moments))
    end

    for chunk=0:n_chunks-1
        for (i,t) in enumerate((chunk*chunk_length+1):((chunk+1)*chunk_length))
            omega_step!(omegap,dt)
            bc_interact=particle_motion_model_step!(x_pos,y_pos, ux,uy,omegap, turb_k_e, m_params, dt, space_cells, np)
            bc_absorbtion!(phip,any(bc_interact[:,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1, precomp_P) #reactive bc is chosen by bc_params.reacting_boundaries
            t%saving_rate==0&&(eval_by_cell!((ind_1,ind_2,cell_particles)-> (
                assign_f_phi_cell!(f_phi,phip[:,cell_particles], psi_mesh, ind_1,ind_2,ceil(Int,i/saving_rate));
            return nothing) , x_pos, y_pos, space_cells))
            if record_moments && t%saving_rate_moments==0 
                (means[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:]))
                (means[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:]))
                (mom_2[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:].^2))
                (mom_2[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:].^2))
            end
            verbose && print(t,' ')
        end
        write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"data",f_phi)
        write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"array_shape",[i for i in size(f_phi)])
        if record_moments
            write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"mean",means)
            write(foldername*'/'*string(chunk*chunk_length+1)*'_'*string((chunk+1)*chunk_length)*"2nd_moment",mom_2)
        end
        write(foldername*'/'*"total_shape", [chunk+1,chunk_length,((chunk+1)*chunk_length),false,record_moments,saving_rate] )
        verbose && println('\n',"saved steps: "*string(chunk*chunk_length+1)*" to "*string((chunk+1)*chunk_length))
    end
    if (n_chunks)*chunk_length < nt
        f_phi=zeros(T,psi_mesh.psi_partions_num_1, psi_mesh.psi_partions_num_2, space_cells.y_res, space_cells.x_res, ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate ))
        if record_moments
            means=zeros(T,2,ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate_moments))
            mom_2=zeros(T,2,ceil(Int,(nt-(n_chunks)*chunk_length)/saving_rate_moments))
        end
        for (i,t) in enumerate(((n_chunks)*chunk_length+1):nt)
            omega_step!(omegap,dt)
            bc_interact=particle_motion_model_step!(x_pos,y_pos, ux,uy,omegap, turb_k_e, m_params, dt, space_cells, np)
            bc_absorbtion!(phip,any(bc_interact[:,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1, precomp_P) #reactive bc is chosen by bc_params.reacting_boundaries
            t%saving_rate==0 && eval_by_cell!((ind_1,ind_2,cell_particles)-> (
                assign_f_phi_cell!(f_phi,phip[:,cell_particles], psi_mesh, ind_1,ind_2,ceil(Int,i/saving_rate));
            return nothing) , x_pos, y_pos, space_cells)
            if record_moments && t%saving_rate_moments==0 
                (means[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:]))
                (means[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:]))
                (mom_2[1,ceil(Int,i/saving_rate_moments)] = mean(phip[1,:].^2))
                (mom_2[2,ceil(Int,i/saving_rate_moments)] = mean(phip[2,:].^2))
            end
            verbose && print(t,' ')
        end
        write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"data",f_phi)
        write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"array_shape",[i for i in size(f_phi)])
        if record_moments
            write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"mean",means)
            write(foldername*'/'*string(chunk_length*(n_chunks)+1)*'_'*string(nt)*"2nd_moment",mom_2)
        end
        write(foldername*'/'*"total_shape", [(n_chunks+1),chunk_length,nt,true, record_moments,saving_rate] )
        verbose && println('\n',"saved steps: "*string(chunk_length*(n_chunks+1)+1)*" to "*string(nt))
    end
    verbose && println("end")
    return nothing
end

# end