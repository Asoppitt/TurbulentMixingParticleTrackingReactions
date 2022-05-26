include("PSP_Particletracking_module.jl")

module ContinuousSaving

function particle_motion_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1}, ux::AbstractArray{T,1},uy::AbstractArray{T,1}, turb_k_e::T, m_params::MotionParams{T}, dt::T, space_cells::CellGrid{T},np::Integer) where T<:AbstractFloat
    "takes x_pos, y_pos, ux, uy and computes the correct velocity/position, stored in index
    Also records the boundary interaction array, as output"
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    u_mean=m_params.u_mean
    bc_interact = falses(np, 4)#index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy
    ux_flux=ux.-u_mean
    ux = ux_flux+(-0.5*B*omega_bar*ux_flux)*dt.+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 
    uy = uy+(-0.5*B*omega_bar*uy)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 
    ux .+= u_mean
    x_pos = x_pos + ux*dt # random walk in x-direction
    y_pos = y_pos + uy*dt # random walk in y-direction

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
    uy[mag] = -uyp[mag] #reflecting velocity
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

function PSP_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1},phip::AbstractArray{T,2},
        omegap::AbstractArray{T,1},t_decorr_p::AbstractArray{T,1}, phi_pm::AbstractArray{Integer, 2},
        t_decorr_m::AbstractArray{T,1}, bc_interact::BitArray{2}, dt::T, p_params::PSPParams{T},
        space_cells::CellGrid{T}, bc_params::BCParams{T},np::Integer,precomp_P::T) where T<:AbstractFloat
    
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    
    #E-M solver for omega 
    dw = sqrt(dt).*randn(T, np) #random draws
    omegap = omegap-(omegap.-omega_mean)./T_omega.*dt + sqrt.((omegap.-p_params.omega_min).*(2*omega_sigma_2*omega_mean/T_omega)).*dw
    omegap = omegap.*(omegap.>=p_params.omega_min)+p_params.omega_min.*(omegap.<=p_params.omega_min) #enforcing positivity

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

    #split into cells, compute centres/targets, run ODE step
    eval_by_cell!(function (i,j,cell_particles)
        (length(cell_particles)==0) && throw(BoundsError(cell_particles))
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

    phi_c = 0.5.*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
    diffusion = zeros(T, 2,np)
    diffusion[1,:] = (phip[1,:]-phi_c[1,:]).*(exp.(-c_phi.*0.5.*omegap[:,1].*dt).-1.0)
    diffusion[2,:] = (phip[2,:]-phi_c[2,:]).*(exp.(-c_phi.*0.5.*omegap[:,1].*dt).-1.0)
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
    
    eval_by_cell!(function (i,j,cell_particles)
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
        return nothing
    end, x_pos, y_pos, space_cells)

    dphi = corr_factor.*dphi
    dphi = T_mat*dphi #return to old coords

    #reaction has to be done after mean zero correction - or it has no effect
    reaction = zeros(T, 2,np) # bulk reaction
    reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:],phip[2,:]))#.*exp.(c_phi.*0.5.*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
    reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:],phip[2,:]))
    dphi .+= reaction
    phip .+= dphi
    phip .*= (phip[:,:].>0) #forcing positive concentration

    bc_absorbtion!(phip,any(bc_interact[:,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1, precomp_P) #currently only reacting on bottom bc
    return nothing
end

function PSP_model!(foldername::String,turb_k_e::T, dt::T, initial_condition::Union{String,Tuple{String,Vararg}}, m_params::MotionParams{T}, p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false, chunk_length::Integer=50) where T<:AbstractFloat
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    np, nt = size(x_pos)
    nt-=1
    n_chuncks=floor(Int, nt/chunk_length)
    precomp_P = min.(bc_params.bc_k.*sqrt.(bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)

    up = randn(T, np).*sqrt.(2/3 .*turb_k_e[:,1])
    up = randn(T, np).*sqrt.(2/3 .*turb_k_e[:,1])
    x_pos = zeros(T, np)
    y_pos = zeros(T, np)
    x_pos = length_domain.*rand(float_type, np)
    y_pos = height_domain.*rand(float_type, np)

    phip = zeros(T, (2, np)) #scalar concentration at these points
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle

    f_phi=zeros(T,psi_mesh.psi_partions_num, psi_mesh.psi_partions_num, space_cells.y_res, space_cells.x_res, chunk_length)

    set_phi_as_ic!(phip,initial_condition,x_pos,y_pos,space_cells,1)
    assign_f_phi!(f_phi,phip, x_pos, y_pos, psi_mesh, space_cells,1)

    omegap = zeros(T, np) #turbulence frequency
    omega0_dist = Gamma((omega_mean-p_params.omega_min)^2/(omega_sigma_2),(omega_sigma_2)/(omega_mean-p_params.omega_min)) #this should now match long term distribution of omega
    omegap = rand(omega0_dist, np).+p_params.omega_min
    
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip, cell_particles, cell_particles)
    ;assign_f_phi_cell!(f_phi,phip[:,cell_particles],psi_mesh,i,j,1);return nothing) , x_pos, y_pos, space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed
    t_decorr_p = 1 ./(c_t.*omegap[phi_pm[1,:]]).*rand(T,np)
    t_decorr_m = 1 ./(c_t.*omegap[phi_pm[2,:]]).*rand(T,np)

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle]]-phip[:,particle]),(phip[:,phi_pm[2,particle]]-phip[:,particle]))
    end

    for chunck=1:n_chuncks
    for t in (chunck+1):(chunck+chunck_length)
        particle_motion_model_step!(x_pos,y_pos, ux,uy, turb_k_e, m_params, dt, space_cells, np)
        PSP_model_step!(x_pos,y_pos,phip,omegap,t_decorr_p, phi_pm,t_decorr_m, bc_interact, dt, p_params,space_cells, bc_params,np,precomp_P)
        assign_f_phi!(f_phi,phip[:,:], x_pos, y_pos, psi_mesh, space_cells,t+1)
        verbose && print(t,' ')
    end
    write(foldername*'/'*string(chunck+1)*'_'*string(chunck+chunck_length),f_phi)
    write(foldername*'/'*string(chunck+1)*'_'*string(chunck+chunck_length)*"shape",size(f_phi))
    verbose && println('\n',"saved steps: "*string(chunck+1)*" to "*string(chunck+chunck_length))
    end
    for t in (chunck_length*n_chuncks+1):nt
        particle_motion_model_step!(x_pos,y_pos, ux,uy, turb_k_e, m_params, dt, space_cells, np)
        PSP_model_step!(x_pos,y_pos,phip,omegap,t_decorr_p, phi_pm,t_decorr_m, bc_interact, dt, p_params,space_cells, bc_params,np,precomp_P)
        assign_f_phi!(f_phi,phip[:,:], x_pos, y_pos, psi_mesh, space_cells,t+1)
        verbose && print(t,' ')
    end
    write(foldername*'/'*string(chunck_length*n_chuncks+1)*'_'*string(nt),f_phi)
    write(foldername*'/'*string(chunck+1)*'_'*string(chunck+chunck_length)*"shape",size(f_phi))
    verbose && println('\n',"saved steps: "*string(chunck+1)*" to "*string(chunck+chunck_length))
    verbose && println("end")
    return nothing
end

end