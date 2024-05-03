# include("PSP_Particletracking_module.jl")


function particle_motion_model_step!(x_pos::AbstractArray{T,1},y_pos::AbstractArray{T,1}, ux::AbstractArray{T,1},uy::AbstractArray{T,1}, omegap::Omega{T}, turb_k_e::T, m_params::MotionParams{T,T}, dt::T, space_cells::CellGrid{T},np::Integer) where T<:AbstractFloat
    "takes x_pos, y_pos, ux, uy and computes the correct velocity/position, stored in index
    Also records the boundary interaction array, as output"
    C_0=m_params.C_0
    B=m_params.B
    bc_interact = falses(np, 4)#index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy
    x_pos[:]= x_pos + ux*dt # random walk in x-direction
    y_pos[:]= y_pos + uy*dt # random walk in y-direction
    ux_f=ux.-m_params.u_mean #find fluctuating velocity
    #an exact solver of O-U, should be more stable for stiff cases/high omega
    ux[:]= ux_f.*exp(-T(0.5)*B.*omegap.omega_bar*dt).+
        sqrt.(C_0.*turb_k_e.*omegap.omega_bar)./sqrt(B.*omegap.omega_bar).*
        sqrt(-expm1(-B.*omegap.omega_bar.*dt)).*randn(T, np)
    uy[:]= uy.*exp(-T(0.5)*B.*omegap.omega_bar.*dt).+
        sqrt.(C_0.*turb_k_e.*omegap.omega_bar)./sqrt(B.*omegap.omega_bar).*
        sqrt(-expm1(-B.*omegap.omega_bar.*dt)).*randn(T, np)
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
    C_0=m_params.C_0
    B=m_params.B
    ux_mean=m_params.u_mean[1]
    uy_mean=m_params.u_mean[2]
    bc_interact = falses(np, 4)#index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy

    x_pos[:]= x_pos + ux*dt # random walk in x-direction
    y_pos[:]= y_pos + uy*dt # random walk in y-direction
    #an exact solver of O-U, should be more stable
    ux[:]= ux.*exp(-T(0.5)*B.*omegap.omega_bar*dt).+
        .-ux_mean.(x_pos,y_pos).*expm1(-T(0.5)*B.*omegap.omega_bar*dt).+
        sqrt.(C_0.*turb_k_e.*omegap.omega_bar)./sqrt(B.*omegap.omega_bar).*
        sqrt(-expm1(-B.*omegap.omega_bar.*dt)).*randn(T, np)
    uy[:]= uy.*exp(-T(0.5)*B.*omegap.omega_bar.*dt).+
        .-uy_mean.(x_pos,y_pos).*expm1(-T(0.5)*B.*omegap.omega_bar*dt).+
        sqrt.(C_0.*turb_k_e.*omegap.omega_bar)./sqrt(B.*omegap.omega_bar).*
        sqrt(-expm1(-B.*omegap.omega_bar.*dt)).*randn(T, np)

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

function omega_step!(omegap::Omega{T,Gamma,true},dt::T) where T<:AbstractFloat
    #E-M solver for omega, in gamma dist
    dw = sqrt(dt).*randn(T, size(omegap,1)) #random draws
    omegap .-= -(omegap.-omegap.omega_bar).*omegap.inv_T_omega.*dt + sqrt.((omegap.-omegap.omega_min).*(2*omegap.omega_sigma_2*omegap.omega_bar*omegap.inv_T_omega)).*dw
    omegap = omegap.*(omegap.>=omegap.omega_min)+omegap.omega_min.*(omegap.<=omegap.omega_min) #enforcing positivity
    return nothing
end

function omega_step!(omegap::Omega{T,Gamma,false},dt::T) where T<:AbstractFloat
    #E-M solver for omega, in gamma dist
    dw = sqrt(dt).*randn(T, size(omegap,1)) #random draws
    omegap.log_omega_normed .-= -(omegap.+(2*omegap.omega_sigma_2)*omegap.omega_bar).*omegap.inv_T_omega./omegap.*dt + sqrt.((2*omegap.omega_sigma_2*omegap.omega_bar*omegap.inv_T_omega)./omegap).*dw
    omegap = exp.(omegap.log_omega_normed)
    return nothing
end

function omega_step!(omegap::Omega{T,LogNormal},dt::T) where T<:AbstractFloat
    #E-M solver for LogNormal
    dw = sqrt(dt).*randn(T,size(omegap,1)) #random draws
    omegap.log_omega_normed[:] .+=  .-(omegap.inv_T_omega).*(omegap.log_omega_normed.+T(0.5).*omegap.log_sigma_2).*dt.+sqrt.(2*omegap.inv_T_omega*omegap.log_sigma_2).*dw

    omegap[:] = exp.(omegap.log_omega_normed)*(omegap.omega_bar)

    return nothing
end

function make_omega_dist(p_params::PSPParams{T}) where T<:AbstractFloat
    if p_params.omega_dist === :Gamma
        return eval(p_params.omega_dist){T}(T(1/(p_params.omega_sigma_2)),
            T(p_params.omega_sigma_2*(p_params.omega_bar-p_params.omega_min))) #this should now match long term distribution of omega
    elseif p_params.omega_dist === :LogNormal
        return eval(p_params.omega_dist){T}(T((log(p_params.omega_bar/sqrt(1+p_params.omega_sigma_2)))),
            T(sqrt(log(1+p_params.omega_sigma_2)))) #this should now match long term distribution of omega
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
