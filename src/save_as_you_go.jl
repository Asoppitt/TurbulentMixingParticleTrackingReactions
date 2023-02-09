include("PSP_Particletracking_module.jl")
# module ContinuousSaving
export no_psp_motion_model!, PSP_model!

function particle_motion_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1}, ux::AbstractArray{T,1},uy::AbstractArray{T,1}, omegap::Omega{T}, turb_k_e::T, m_params::MotionParams{T,T}, dt::T, space_cells::CellGrid{T},np::Integer) where T<:AbstractFloat
    "takes x_pos, y_pos, ux, uy and computes the correct velocity/position, stored in index
    Also records the boundary interaction array, as output"
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    bc_interact = falses(np, 4)#index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy
    x_pos[:]= x_pos + ux*dt # random walk in x-direction
    y_pos[:]= y_pos + uy*dt # random walk in y-direction
    ux_f=ux.-m_params.u_mean #find fluctuating velocity
    ux[:]= ux_f+(-T(0.5)*B.*omegap.omega.*ux_f).*dt.+randn(T, np).*sqrt.(C_0.*turb_k_e.*omegap.omega.*dt); 
    uy[:]= uy+(-T(0.5).*B.*omegap.omega.*uy)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omegap.omega.*dt); 
    ux[:].+= m_params.u_mean

    # Reflection particles at boundaries

    # Reflection at upper boundary y>height_domain
    # doing closed on top open on bottom, as cell detection is open on top,
    # closed on bottom
    mag = findall(y_pos.>=space_cells.height_domain) # index of particle with yp>height_domain
    dim_mag = size(mag) # dimension of array "mag"

    y_mag_succ = y_pos[mag] # yp at time t+1 corresponding to the index "mag"

    V1 = space_cells.height_domain.*ones(T, dim_mag) 

    ypr_mag = V1*2 .- y_mag_succ  # yp at time t+1 of the reflected particle

    y_pos[mag]= ypr_mag #replacement of yp>1 with yp of reflected particle
    uy[mag] = -uy[mag] #reflecting velocity
    bc_interact[mag,1] .= true

    # Reflection at lower boundary y<0
    mag = findall(y_pos.<=0) # index of particle with yp>height_domain
    dim_mag = size(mag) # dimension of array "mag"

    y_mag_succ = y_pos[mag] # yp at time t+1 corresponding to the index "mag"

    ypr_mag = - y_mag_succ  # yp at time t+1 of the reflected particle

    y_pos[mag]= ypr_mag #replacement of yp<0 with yp of reflected particle
    uy[mag] = -uy[mag] #reflecting velocity
    bc_interact[mag,2] .= true

    #bc at end (y=length_domain) of domain
    end_indicies = x_pos.>=space_cells.length_domain #index of particle with xp>length

    end_x = x_pos[end_indicies,]
    xpr_end = end_x .- space_cells.length_domain #shifting particles back to begining
    x_pos[end_indicies] = xpr_end #replacing x coords

    bc_interact[end_indicies,3] .= true

    #bc at start (x=0) of domain
    start_indicies = x_pos.<=0 #index of particle with xp>length

    xpr_start = space_cells.length_domain .+ x_pos[start_indicies] 
    x_pos[start_indicies] = xpr_start #replacing x coords
    bc_interact[start_indicies,4] .= true
    return bc_interact
end

function particle_motion_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1}, ux::AbstractArray{T,1},uy::AbstractArray{T,1}, omegap::Omega{T}, turb_k_e::T, m_params::MotionParams{T,Tuple{F,G}}, dt::T, space_cells::CellGrid{T},np::Integer) where T<:AbstractFloat where F<:Function where G<:Function
    "takes x_pos, y_pos, ux, uy and computes the correct velocity/position, stored in index
    Also records the boundary interaction array, as output"
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    ux_mean=m_params.u_mean[1]
    uy_mean=m_params.u_mean[2]
    bc_interact = falses(np, 4)#index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy

    x_pos[:]= x_pos + ux*dt # random walk in x-direction
    y_pos[:]= y_pos + uy*dt # random walk in y-direction
    ux[:]= ux+T(0.5)*B.*omegap.omega.*(ux_mean.(x_pos,y_pos)-ux)*dt.+randn(T, np).*sqrt.(C_0.*turb_k_e.*omegap.omega.*dt); 
    uy[:]= uy+T(0.5)*B.*omegap.omega.*(uy_mean.(x_pos,y_pos)-uy)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omegap.omega.*dt);

    # Reflection particles at boundaries

    # Reflection at upper boundary y>height_domain
    # doing closed on top open on bottom, as cell detection is open on top,
    # closed on bottom
    mag = findall(y_pos.>=space_cells.height_domain) # index of particle with yp>height_domain
    dim_mag = size(mag) # dimension of array "mag"

    y_mag_succ = y_pos[mag] # yp at time t+1 corresponding to the index "mag"

    V1 = space_cells.height_domain.*ones(T, dim_mag) 

    ypr_mag = V1*2 .- y_mag_succ  # yp at time t+1 of the reflected particle

    y_pos[mag]= ypr_mag #replacement of yp>1 with yp of reflected particle
    uy[mag] = -uy[mag] #reflecting velocity
    bc_interact[mag,1] .= true

    # Reflection at lower boundary y<0
    mag = findall(y_pos.<=0) # index of particle with yp>height_domain
    dim_mag = size(mag) # dimension of array "mag"

    y_mag_succ = y_pos[mag] # yp at time t+1 corresponding to the index "mag"

    ypr_mag = - y_mag_succ  # yp at time t+1 of the reflected particle

    y_pos[mag]= ypr_mag #replacement of yp<0 with yp of reflected particle
    uy[mag] = -uy[mag] #reflecting velocity
    bc_interact[mag,2] .= true

    #bc at end (y=length_domain) of domain
    end_indicies = x_pos.>=space_cells.length_domain #index of particle with xp>length

    end_x = x_pos[end_indicies,]
    xpr_end = end_x .- space_cells.length_domain #shifting particles back to begining
    x_pos[end_indicies] = xpr_end #replacing x coords

    bc_interact[end_indicies,3] .= true

    #bc at start (x=0) of domain
    start_indicies = x_pos.<=0 #index of particle with xp>length

    xpr_start = space_cells.length_domain .+ x_pos[start_indicies] 
    x_pos[start_indicies] = xpr_start #replacing x coords
    bc_interact[start_indicies,4] .= true
    return bc_interact
end

function omega_step!(omegap::Omega{T,Gamma,:Min},dt::T) where T<:AbstractFloat
    #E-M solver for omega, in gamma dist
    dw = sqrt(dt).*randn(T, size(omegap,1)) #random draws
    omegap .-= -(omegap.-omegap.omega_bar).*omegap.inv_T_omega.*dt + sqrt.((omegap.-omegap.omega_min).*(2*omegap.omega_sigma_2*omegap.omega_bar*omegap.inv_T_omega)).*dw
    omegap = omegap.*(omegap.>=omegap.omega_min)+omegap.omega_min.*(omegap.<=omegap.omega_min) #enforcing positivity
    return nothing
end

function omega_step!(omegap::Omega{T,Gamma},dt::T) where T<:AbstractFloat
    #E-M solver for omega, in gamma dist
    dw = sqrt(dt).*randn(T, size(omegap,1)) #random draws
    omegap.log_omega .-= -(omegap.+(2*omegap.omega_sigma_2)*omegap.omega_bar).*omegap.inv_T_omega./omegap.*dt + sqrt.((2*omegap.omega_sigma_2*omegap.omega_bar*omegap.inv_T_omega)./omegap).*dw
    omegap = exp.(omegap.log_omega)
    return nothing
end

function omega_step!(omegap::Omega{T,LogNormal},dt::T) where T<:AbstractFloat
    #E-M solver for LogNormal
    dw = sqrt(dt).*randn(T,size(omegap,1)) #random draws
    omegap.log_omega[:] .+=  -(omegap.inv_T_omega).*(omegap.log_omega.+T(0.5)*omegap.log_sigma_2).*dt.+sqrt.(2*omegap.inv_T_omega*omegap.log_sigma_2).*dw

    omegap[:] = exp.(omegap.log_omega.+omegap.log_omega_bar)

    return nothing
end

function make_omega_dist(p_params::PSPParams{T}) where T<:AbstractFloat
    if p_params.omega_dist === :Gamma
        return eval(p_params.omega_dist){T}(T((p_params.omega_bar-p_params.omega_min)^2/(p_params.omega_sigma_2)),
        T((p_params.omega_sigma_2)/(p_params.omega_bar-p_params.omega_min))) #this should now match long term distribution of omega
    elseif p_params.omega_dist === :LogNormal
        return eval(p_params.omega_dist){T}(T((log(p_params.omega_bar^2/sqrt(p_params.omega_bar^2+p_params.omega_sigma_2)))),T(sqrt(log((p_params.omega_bar^2+p_params.omega_sigma_2)/p_params.omega_bar^2)))) #this should now match long term distribution of omega
    else
        throw(ArgumentError(p_params.omega_dist))
    end
end

function PSP_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1},phip::AbstractArray{T,3},
        celli::AbstractArray{Array{Int,1},2}, omegap::Omega{T}, t_decorr_m::AbstractArray{T,1},
        t_decorr_p::AbstractArray{T,1}, phi_pm::AbstractArray{index_type, 2}, bc_interact::BitArray{2}, 
        dt::T, p_params::PSPParams{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, np::Integer,
        precomp_P::T) where T<:AbstractFloat where index_type<:Integer 
    
    c_phi = p_params.c_phi
    c_t = p_params.c_t

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle]]-phip[:,particle]),(phip[:,phi_pm[2,particle]]-phip[:,particle]))
    end
    
    omega_step!(omegap,dt)

    #stepping the decorrelation times
    t_decorr_p = t_decorr_p.-dt;
    t_decorr_m = t_decorr_m.-dt;
    #finding particles needing new pairs
    t_p0 = t_decorr_p.<=0
    t_m0 = t_decorr_m.<=0
    t_pm0 = t_p0 .& t_m0
    t_pm_n0 = findall(.!(t_p0 .| t_m0)) # where t_p and t_m>0 
    t_pm0[t_pm_n0] .|= (test_dot.(t_pm_n0).>0)#add those that fail the boundary condition to be updated
    t_p0 = xor.(t_p0,t_pm0)
    t_m0 = xor.(t_m0,t_pm0)
    
    #update cell particle lists
    eval_by_cell!(function (i,j,cell_particles)
        (length(cell_particles)==0) && throw(BoundsError(cell_particles))
        #adjust mass to match new cell particle count. ref Elisa Baioni 2021
        celli[i,j]=cell_particles
        #reassigning particles that completed decorrelation time
        t_p0_cell = cell_particles[t_p0[cell_particles]]
        t_m0_cell = cell_particles[t_m0[cell_particles]]
        t_pm0_cell = cell_particles[t_pm0[cell_particles]]
        (length(t_p0_cell)>0)&&assign_pm_single!(phi_pm, phip,t_p0_cell, cell_particles, 1)
        (length(t_m0_cell)>0)&&assign_pm_single!(phi_pm, phip,t_m0_cell, cell_particles, 2)
        (length(t_pm0_cell)>0)&&assign_pm!(phi_pm, phip, t_pm0_cell, cell_particles)
        #update pairs to ensure all are within the same bounds
        pm_check_and_recal_for_cell_change!(phi_pm, phip, cell_particles)
        return nothing
    end, x_pos, y_pos, space_cells)

    #reset decorrelation time for particles it had run out on
    t_decorr_p[t_p0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[1,t_p0.&t_pm0]])
    t_decorr_m[t_m0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[2,t_m0.&t_pm0]])

    phi_c = T(0.5).*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
    diffusion = zeros(T, 2,np)
    diffusion[1,:] = (phip[1,:]-phi_c[1,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,1].*dt).-1.0)
    diffusion[2,:] = (phip[2,:]-phi_c[2,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,1].*dt).-1.0)
    dphi = diffusion 
    # ensuring mean 0 change
    # generating a random orthonormal basis
    # is 2-d so genrate a random unit vector from an angle and proceed based
    # on that
    angle = 2*pi*rand(T, 1)[1];
    e_1 = [cos(angle),sin(angle)]
    handedness = sb.sample([-1,1],1)[1] #randomly choose betwen left or right handed system
    e_2 = handedness*[e_1[2],-e_1[1]]
    T_mat=zeros(2,2)
    T_mat[:,1] = e_1  #coord transform matrix
    T_mat[:,2] = e_2
    dphi = T_mat\dphi  #transform to new coords
    #performing adjustment to mean 0
    corr_factor = zeros(T, 2,np)
    
    # eval_by_cell!(function (i,j,cell_particles) #if recomputaion of cell is needed
    for cell_particles in celli
        for phi_i=1:2
            phi_mean = mean(dphi[phi_i,cell_particles])
            if phi_mean != 0 #isn't true for empty cells
                cell_points_pos = dphi[phi_i,cell_particles].>0
                cell_points_neg = dphi[phi_i,cell_particles].<0
                phi_pos_mean = mean(cell_points_pos.*dphi[phi_i,cell_particles])
                phi_neg_mean = mean((cell_points_neg).*dphi[phi_i,cell_particles])
                if phi_mean>0
                    corr_factor[phi_i,cell_particles]=.-cell_points_pos*(phi_neg_mean./phi_pos_mean) + (1 .- cell_points_pos)
                else
                    corr_factor[phi_i,cell_particles]=.-(cell_points_neg)*(phi_pos_mean./phi_neg_mean) + (1 .- cell_points_neg)
                end
            end
        end
    end
        # return nothing
    # end, x_pos, y_pos, space_cells)

    dphi = corr_factor.*dphi
    dphi = T_mat*dphi #return to old coords

    #reaction has to be done after mean zero correction - or it has no effect
    reaction = zeros(T, 2,np) # bulk reaction
    reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:],phip[2,:]))#.*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
    reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:],phip[2,:]))
    dphi .+= reaction
    phip[:,:] .+= dphi
    phip[:,:,:] .*= (phip[:,:,:].>0) #forcing positive concentration
    bc_absorbtion!(phip,any(bc_interact[:,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1, precomp_P) #reactive bc is chosen by bc_params.reacting_boundaries
    return nothing
end

function PSP_model!(foldername::String,turb_k_e::T, nt::Integer, dt::T, np::Integer, initial_condition::Union{String,Tuple{String,Vararg}}, m_params::MotionParams{T}, p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false, chunk_length::Integer=50; record_moments=false, saving_rate=1, saving_rate_moments=saving_rate) where T<:AbstractFloat
    c_t = p_params.c_t
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
    t_decorr_p = T(1) ./(c_t.*omegap[phi_pm[1,:]]).*rand(T,np)
    t_decorr_m = T(1) ./(c_t.*omegap[phi_pm[2,:]]).*rand(T,np)

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