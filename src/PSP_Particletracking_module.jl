# module PSPParticleTrackingReactions

using Random, Distributions#, Plots
import StatsBase as sb
import LinearAlgebra as la
import Statistics as st
import Base.getindex, Base.setindex!, Base.size, Base.length, Base.iterate, Base.ndims, Base.copyto!

export cell_grid,BC_params,psi_grid,motion_params,PSP_motion_bc_params,
particle_motion_model, PSP_model!, particle_motion_model_ref_start, PSP_model_record_phi_local_diff!,
PSP_model_record_reacting_mass!

phi_eps=1e-12
struct CellGrid{T<:AbstractFloat}
    x_res ::Int
    y_res ::Int
    length_domain ::T
    height_domain ::T
    x_edges ::AbstractVector{T}
    y_edges ::AbstractVector{T}
end

struct PsiGrid{T<:AbstractFloat}
    psi_partions_num ::Int
    psi_partions_num_1 ::Int
    psi_partions_num_2 ::Int
    phi_domain ::AbstractVector{T}
    psi_1 ::AbstractVector{T}
    psi_2 ::AbstractVector{T}
end

struct PSPParams{T<:AbstractFloat}
    omega_bar::T
    omega_sigma_2::T
    omega_min::T
    T_omega::T
    omega_dist::Symbol
    c_phi::T
    c_t::T
    reaction_form :: Tuple{Function, Function}
end

#structures to hold the data needed for Omega
abstract type Omega{AbstractFloat,Distribution} end

struct OmegaG{T}<:Omega{T,Gamma}
    omega::Vector{T}
    dist::Gamma
    omega_bar::T
    omega_sigma_2::T
    omega_min::T
    T_omega::T
    inv_T_omega::T
end

struct OmegaL{T}<:Omega{T,LogNormal}
    omega::Vector{T}
    log_omega::Vector{T}
    dist::LogNormal
    omega_bar::T
    omega_sigma_2::T
    omega_min::T
    T_omega::T
    log_omega_bar::T
    log_sigma_2::T
    inv_T_omega::T
end

function copyto!(o::O,a...) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    copyto!(o.omega, a...)
    return nothing
end

function ndims(o::O) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return 1
end

function ndims(t::T2) where T2<:Type{O} where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return 1
end

function getindex(o::O, i...) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return o.omega[i...]
end

function getindex(o::O, i::I) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution where I<:AbstractArray
    return o.omega[i]
end

function setindex!(o::O, info::T2, i...) where T2<:Union{TA,T} where TA<:AbstractArray{T} where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    o.omega[i...]=info
    return nothing
end

function setindex!(o::O, info::T2, i::I ) where T2<:Union{TA,T} where TA<:AbstractArray{T} where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution where I<:AbstractArray
    o.omega[i]=info
    return nothing
end

function length(o::O) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return length(o.omega)
end

function size(o::O) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return size(o.omega)
end

function size(o::O, i::I) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution where I<:Integer
    return size(o.omega, i)
end

function iterate(o::O) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution
    return iterate(o.omega)
end

function iterate(o::O, i::I) where O<:Omega{T,Dist} where T<:AbstractFloat where Dist<:Distribution where I<:Integer
    return iterate(o.omega, i)
end
struct MotionParams{T<:AbstractFloat,T2<:Union{AbstractFloat,Tuple{Function,Function}}}
    omega_bar::T
    C_0::T
    B::T
    u_mean::T2
end

struct BCParams{T<:AbstractFloat, Tvp<:Real, T_corr<:Union{Nothing, Function, Tuple{Function,Function}}, binary}
    bc_k::T
    C_0::T
    B::T
    num_vp::Tvp
    corr_factor::T_corr
    reacting_boundaries::AbstractArray{Bool}#which boundary(ies) react
end

function Omega(dist::D,np::I,p_params::PSPParams{T}) where I<:Integer where D<:Gamma where T<:AbstractFloat
    omega=T.(rand(dist,np))
    return OmegaG(omega,dist,p_params.omega_bar,p_params.omega_sigma_2,p_params.omega_min,p_params.T_omega,1/p_params.T_omega)
end

function Omega(dist::D,np::I,p_params::PSPParams{T}) where I<:Integer where D<:LogNormal where T<:AbstractFloat
    omega=T.(rand(dist,np))
    return OmegaL(omega,log.(omega),dist,p_params.omega_bar,p_params.omega_sigma_2,p_params.omega_min,p_params.T_omega,log(p_params.omega_bar),1/p_params.T_omega,varlogx(dist))
end

function cell_grid(x_res ::Int,y_res ::Int,length_domain ::T,height_domain ::T) where T<:AbstractFloat #constructor for CellGrid
    return CellGrid(x_res,y_res,length_domain,height_domain,LinRange(0,length_domain,x_res+1),LinRange(0,height_domain,y_res+1))
end

function psi_grid(psi_partions_num ::Int, phi_domain ::AbstractVector{T}) where T<:AbstractFloat #constructor for PsiGrid
    return PsiGrid(psi_partions_num,psi_partions_num,psi_partions_num, phi_domain,LinRange(phi_domain[1], phi_domain[2], psi_partions_num+1),LinRange(phi_domain[1], phi_domain[2], psi_partions_num+1))
end
function psi_grid(psi_partions_num1 ::Int,psi_partions_num2 ::Int, phi_domain ::AbstractVector{T}) where T<:AbstractFloat #constructor for PsiGrid
    return PsiGrid(psi_partions_num1,psi_partions_num1,psi_partions_num2, phi_domain,LinRange(phi_domain[1], phi_domain[2], psi_partions_num1+1),LinRange(phi_domain[1], phi_domain[2], psi_partions_num2+1))
end

function PSP_params(omega_bar::T, omega_sigma_2::T,omega_min::T, c_phi::T, c_t::T, react_func=((x,y)->0,(x,y)->0), omega_dist=:Gamma::Symbol; omega_t=1/omega_bar) where T<:AbstractFloat
    (omega_bar <= omega_min) && throw(DomainError(omega_min,"omega_min must be less than omega_bar"))
    return PSPParams(omega_bar, omega_sigma_2, omega_min, omega_t, omega_dist, c_phi, c_t, react_func)
end

function motion_params(omega_bar::T,C_0::T, B_format::String, u_mean::T2) where T<:AbstractFloat where T2<:Union{AbstractFloat,Tuple{Function,Function}}
    if B_format == "Decay"
        return MotionParams(omega_bar, C_0, (1+T(1.5)*C_0), u_mean)
    elseif B_format == "Constant"
        return MotionParams(omega_bar, C_0, (T(1.5)*C_0), u_mean)
    end
end

function BC_params(bc_k::T, C_0::T, B_format::String, num_vp::Tvp; corr_func::T_corr=nothing, reacting_boundaries::AbstractArray{String}=["lower"]) where T<:AbstractFloat where Tvp<:Int where T_corr<:Union{Nothing, Function, Tuple{Function,Function}}
    if num_vp==1
        binary=true
    else
        binary=false
    end
    boundary_names=["upper", "lower", "right", "left"]
    reacting_boundaries=lowercase.(reacting_boundaries)
    any(.!in.(reacting_boundaries,[boundary_names])) && @warn "invalid boundary names:" *join(reacting_boundaries[.!in.(reacting_boundaries,[boundary_names])] , ", ")*"\ncontinuing using valid boundaries"
    reacting_boundaries_ind = in.(boundary_names,[reacting_boundaries])
    if B_format == "Decay"
        return BCParams{T,Tvp,T_corr,binary}(bc_k, C_0, (1+T(1.5)*C_0),num_vp,corr_func,reacting_boundaries_ind)
    elseif B_format == "Constant"
        return BCParams{T,Tvp,T_corr,binary}(bc_k, C_0, (T(1.5)*C_0),num_vp,corr_func,reacting_boundaries_ind)
    end
end

function BC_params(bc_k::T, C_0::T, B_format::String, num_vp::Tvp; corr_func::T_corr=nothing, reacting_boundaries::AbstractArray{String}=["lower"]) where T<:AbstractFloat where Tvp<:AbstractFloat where T_corr<:Union{Nothing, Function}
    boundary_names=["upper", "lower", "right", "left"]
    reacting_boundaries=lowercase.(reacting_boundaries)
    any(.!in.(reacting_boundaries,[boundary_names])) && @warn "invalid boundary names:" *join(reacting_boundaries[.!in.(reacting_boundaries,[boundary_names])] , ", ")*"\ncontinuing using valid boundaries"
    reacting_boundaries_ind = in.(boundary_names,[reacting_boundaries])
    num_vp == Inf || throw(DomainError("nvpart_per_part must be Int or Inf"))
    if B_format == "Decay"
        return BCParams{T,Tvp,T_corr,false}(bc_k, C_0, (1+T(1.5)*C_0),num_vp,corr_func,reacting_boundaries_ind)
    elseif B_format == "Constant"
        return BCParams{T,Tvp,T_corr,false}(bc_k, C_0, (T(1.5)*C_0),num_vp,corr_func,reacting_boundaries_ind)
    end
end

function PSP_motion_bc_params(omega_bar::T, omega_sigma_2::T,omega_min::T, C_0::T, B_format::String, c_phi::T, c_t::T,u_mean::T2,bc_k::T,num_vp::Real; corr_func::T_corr=nothing, bulk_reaction=((x,y)->0,(x,y)->0), reacting_boundaries::AbstractArray{String}=["lower"], omega_dist=:Gamma::Symbol, omega_t=1/omega_bar) where T<:AbstractFloat where T_corr<:Union{Nothing, Function}  where T2<:Union{AbstractFloat,Tuple{Function,Function}}
    return PSP_params(omega_bar, omega_sigma_2,omega_min, c_phi, c_t, bulk_reaction, omega_dist,omega_t=omega_t), motion_params(omega_bar,C_0, B_format, u_mean), BC_params(bc_k, C_0, B_format, num_vp, corr_func=corr_func,reacting_boundaries=reacting_boundaries)
end

function assign_pm!(phi_pm::Matrix{Int}, phi_array_t::Array{T}, particles::Vector{Int}, cell_points::Vector{Int}) where T<:AbstractFloat
    # might be worth a strategy that removes tested pairs, can't see how to make it not require a biiiig temp variable though 
    for particle in particles
        while true
            try_pm = sb.sample(cell_points,2,replace=false, ordered=true) #ordering added to reduce sample space as order for pm pairs is irrlevent
            test_dot = la.dot((phi_array_t[:,try_pm[1]]-phi_array_t[:,particle]),(phi_array_t[:,try_pm[2]]-phi_array_t[:,particle]))
            if test_dot<= 0
                phi_pm[:,particle] = try_pm
                break
            end
        end
    end
    return nothing
end

function assign_pm_single!(phi_pm::Matrix{Int}, phi_array_t::Array{T}, particles::Vector{Int}, cell_points::Vector{Int}, index_tbc::Int) where T<:AbstractFloat
    known_index = 1-(index_tbc-1)+1 #flips 1 to 2 and 2 to 1
    cell_points_set = Set(cell_points)
    for particle in particles
        !in(particle, cell_points) && throw(ArgumentError("Particles must be in cell"))
        cell_points_set_ = copy(cell_points_set)
        for attempt_count=1:length(cell_points)
            try_pm = rand(cell_points_set_) 
            test_dot = la.dot((phi_array_t[:,try_pm[1]]-phi_array_t[:,particle]),(phi_array_t[:,phi_pm[known_index,particle]]-phi_array_t[:,particle]))
            if  test_dot<= 0
                phi_pm[index_tbc,particle] = try_pm[1]
                break
            end
            setdiff!(cell_points_set_,try_pm)
            attempt_count == length(cell_points) && throw(DomainError(cell_points))
        end
    end
    return nothing
end

function pm_check_and_recal_for_cell_change!(phi_pm::Matrix{Int}, phi_array_t::Array{T}, cell_particles::Vector{Int}) where T<:AbstractFloat
    p_nin = .!(in.(phi_pm[1,cell_particles],Ref(cell_particles)))
    m_nin = .!(in.(phi_pm[2,cell_particles],Ref(cell_particles)))
    if (any(m_nin)||any(p_nin))
        #split into partitions in dim 2, use a sort based approach within each bin
        n_partition = floor(Int,sqrt(length(cell_particles))) 
        cell_particles_sort_perm_1 = sortperm(phi_array_t[1,cell_particles])
        cell_particles_sorts = [cell_particles[cell_particles_sort_perm_1[
            floor(Int, i/n_partition*length(cell_particles))+1:floor(Int, (i+1)/n_partition*length(cell_particles))]]
            for i=0:(n_partition-1)]
        #use to define the edge of the bins
        cell_particles_mins = [cell_particles_sorts[i][1] for i in 1:n_partition]
        for (index,list) in enumerate(cell_particles_sorts)
            cell_particles_sorts[index] = list[sortperm(phi_array_t[2,list])]
        end
        for (p_or_m, nin) in enumerate([p_nin,m_nin])
            for part in cell_particles[nin]
                index_1_float=0.0
                #find which bin
                for i in 1:ceil(Int,log2(n_partition))
                    if phi_array_t[1,cell_particles_mins[floor(Int,index_1_float+n_partition/(2^i))+1]] <= phi_array_t[1,phi_pm[p_or_m,part]]
                        index_1_float+=n_partition/(2^i)
                    end
                end
                index_1=floor(Int,index_1_float)+1
                index_2=0#defining index
                succ = false
                low_high_normal_1=0
                for j in 1:n_partition
                    #find within bin
                    index_2_float=0.0
                    for split_num = 1:ceil(Int,log2(length(cell_particles_sorts[index_1])))
                        #search by splitting
                        if phi_array_t[2,cell_particles_sorts[index_1][floor(Int,index_2_float+length(cell_particles_sorts[index_1])/(2^split_num))+1]] <= phi_array_t[2,phi_pm[p_or_m,part]]
                            index_2_float+=length(cell_particles_sorts[index_1])/(2^split_num)
                        end
                    end
                    index_2 = floor(Int,index_2_float)+1
                    #test condition is held
                    low_high_normal=0#a code to see what type of search to use
                    succ=false
                    for i=1:length(cell_particles_sorts[index_1])
                        test_dot = la.dot((phi_array_t[:,cell_particles_sorts[index_1][index_2]]-phi_array_t[:,part]),(phi_array_t[:,phi_pm[1-(p_or_m-1)+1,part]]-phi_array_t[:,part]))
                        if test_dot<=0
                            succ=true
                            break
                        else
                            if low_high_normal==0
                                new_index=index_2+(-1)^i*i#propose a new index in case of failure
                                if new_index<=0
                                    low_high_normal = 1
                                    new_index=index_2+1
                                elseif new_index>length(cell_particles_sorts[index_1])
                                    low_high_normal = 2
                                    new_index=index_2-1
                                end
                            elseif low_high_normal==1
                                new_index=index_2+1
                            else
                                new_index=index_2-1
                            end
                            index_2=new_index
                        end
                    end
                    if succ
                        break
                    else
                        if low_high_normal_1==0
                            new_index=index_1+(-1)^j*j#propose a new index in case of failure
                            if new_index<=0
                                low_high_normal_1 = 1
                                new_index=index_1+1
                            elseif new_index>n_partition
                                low_high_normal_1 = 2
                                new_index=index_1-1
                            end
                        elseif low_high_normal_1==1
                            new_index=index_1+1
                        else
                            new_index=index_1-1
                        end
                        index_1=new_index
                    end
                end
                if succ
                    phi_pm[p_or_m,part]=cell_particles_sorts[index_1][index_2]
                else
                    throw(ErrorException("couldn't find pair"))
                end
            end
        end
    else
        # print("nothing to reassign") # probably should hide this in a verbosity setting
    end
    return nothing
end

function set_phi_as_ic_up1!(phi_array::Array{TF,3}, t_index::Int, val=TF(1)::TF) where TF<:AbstractFloat
    #Initial_condition == "Uniform phi1"
    nparticles = size(phi_array)[2]
    phi_array[2,:,t_index] = abs.(phi_eps*randn(TF, nparticles)) #pdf can't find zeros
    phi_array[1,:,t_index] .= val
    return nothing
end
function set_phi_as_ic_empty!(phi_array::Array{TF,3}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "zero"
    nparticles = size(phi_array)[2]
    phi_array[:,:,t_index] = abs.(phi_eps*randn(TF,2, nparticles)) #pdf can't find zeros
    return nothing
end
#using subset of particles, mainly for inflow
function set_phi_as_ic_up1!(phi_array::Array{TF,3}, t_index::Int, subset_indicies::IA) where TF<:AbstractFloat where IA<:AbstractArray{I} where I<:Integer
    #Initial_condition == "Uniform phi1"
    nparticles = size(phi_array[:,subset_indicies,:])[2]
    phi_array[2,subset_indicies,t_index] = abs.(phi_eps*randn(TF, nparticles)) #pdf can't find zeros
    phi_array[1,subset_indicies,t_index] .= 1 
    return nothing
end
function set_phi_as_ic_td!(phi_array::Array{TF,3}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "triple_delta"
    nparticles = size(phi_array)[2]
    local delta_selector = rand(1:3, nparticles)
    local noise_term = randn(TF, nparticles)

    phi_array[1,delta_selector.=1,t_index] = -sqrt(3)/2 .+phi_eps .*noise_term[delta_selector.=1]
    phi_array[2,delta_selector.=1,t_index] .= -TF(0.5)

    phi_array[1,delta_selector.=2,t_index] = sqrt(3)/2 .+phi_eps .*noise_term[delta_selector.=2]
    phi_array[2,delta_selector.=2,t_index] .= -TF(0.5)

    phi_array[1,delta_selector.=3,t_index] .= phi_eps
    phi_array[2,delta_selector.=3,t_index] = 1.0 .+phi_eps .*noise_term[delta_selector.=3]
    return nothing
end
function set_phi_as_ic_2l!(phi_array::Array{TF,3},yp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "2 layers"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    phi_array[2,yp.>TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )
    phi_array[1,yp.>TF(0.5)*space_cells.height_domain,t_index] .= 1

    phi_array[1,yp.<=TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
    phi_array[2,yp.<=TF(0.5)*space_cells.height_domain,t_index] .= 1 #.+ uniform_noise[yp[particles,1].<=TF(0.5)*height_domain].*0.05
    return nothing
end
function set_phi_as_ic_2l_one_empty!(phi_array::Array{TF,3},empty_layer::Integer,yp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "1 layer scalar, 1 layer empty"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    if empty_layer==0
        phi_array[2,yp.>TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )
        phi_array[1,yp.>TF(0.5)*space_cells.height_domain,t_index] .= 1

        phi_array[1,yp.<=TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
        phi_array[2,yp.<=TF(0.5)*space_cells.height_domain,t_index] .= abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
    elseif empty_layer==1
        phi_array[2,yp.<=TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
        phi_array[1,yp.<=TF(0.5)*space_cells.height_domain,t_index] .= 1

        phi_array[1,yp.>TF(0.5)*space_cells.height_domain,t_index] = abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )
        phi_array[2,yp.>TF(0.5)*space_cells.height_domain,t_index] .= abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )
    end
    return nothing
end
function set_phi_as_ic_2l_one_empty_x!(phi_array::Array{TF,3},empty_layer::Integer,xp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "1 layer scalar, 1 layer empty"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    if empty_layer==0
        phi_array[2,xp.>TF(0.5)*space_cells.length_domain,t_index] = abs.(phi_eps*noise_term[xp.>TF(0.5)*space_cells.length_domain] )
        phi_array[1,xp.>TF(0.5)*space_cells.length_domain,t_index] .= 1

        phi_array[1,xp.<=TF(0.5)*space_cells.length_domain,t_index] = abs.(phi_eps*noise_term[xp.<=TF(0.5)*space_cells.length_domain] )
        phi_array[2,xp.<=TF(0.5)*space_cells.length_domain,t_index] .= abs.(phi_eps*noise_term[xp.<=TF(0.5)*space_cells.length_domain] )
    elseif empty_layer==1
        phi_array[2,xp.<=TF(0.5)*space_cells.length_domain,t_index] = abs.(phi_eps*noise_term[xp.<=TF(0.5)*space_cells.length_domain] )
        phi_array[1,xp.<=TF(0.5)*space_cells.length_domain,t_index] .= 1

        phi_array[1,xp.>TF(0.5)*space_cells.length_domain,t_index] = abs.(phi_eps*noise_term[xp.>TF(0.5)*space_cells.length_domain] )
        phi_array[2,xp.>TF(0.5)*space_cells.length_domain,t_index] .= abs.(phi_eps*noise_term[xp.>TF(0.5)*space_cells.length_domain] )
    end
    return nothing
end

function set_phi_as_ic_vert_strip!(phi_array::Array{TF,3},left_edge::TF,right_edge::TF,xp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "1 layer scalar, 1 layer empty"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)
    in_strip=(xp.>left_edge) .& (xp.<right_edge)

    phi_array[2,in_strip,t_index] = abs.(phi_eps*noise_term[in_strip] )
    phi_array[1,in_strip,t_index] .= 1

    phi_array[1,.!in_strip,t_index] = abs.(phi_eps*noise_term[.!in_strip] )
    phi_array[2,.!in_strip,t_index] .= abs.(phi_eps*noise_term[.!in_strip] )
    return nothing
end
function set_phi_as_ic_vert_strip!(phi_array::Array{TF,3},left_edge::TF,right_edge::TF,max_phi::TF,xp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "1 layer scalar, 1 layer empty"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)
    in_strip=(xp.>left_edge) .& (xp.<right_edge)

    phi_array[2,in_strip,t_index] = abs.(phi_eps*noise_term[in_strip] )
    phi_array[1,in_strip,t_index] .= max_phi

    phi_array[1,.!in_strip,t_index] = abs.(phi_eps*noise_term[.!in_strip] )
    phi_array[2,.!in_strip,t_index] .= abs.(phi_eps*noise_term[.!in_strip] )
    return nothing
end

function set_phi_as_ic_vert_strip_diff!(phi_array::Array{TF,3},left_edge::TF,right_edge::TF,K::TF,xp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "2 layers difference"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)
    in_strip=(xp.>left_edge) .& (xp.<right_edge)

    phi_array[2,in_strip,t_index] .= K
    phi_array[1,in_strip,t_index] .= 1

    phi_array[1,.!in_strip,t_index] .= sqrt(K)
    phi_array[2,.!in_strip,t_index] .= sqrt(K)
    return nothing
end

function set_phi_as_ic_dd!(phi_array::Array{TF,3},t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "double delta"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    local delta_selector = rand([true,false], nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    phi_array[2,delta_selector,t_index] = abs.(phi_eps*noise_term[delta_selector] )
    phi_array[1,delta_selector,t_index] .= 1 #.+ uniform_noise[delta_selector.==1].*0.05

    phi_array[1,.!delta_selector,t_index] = abs.(phi_eps*noise_term[.!delta_selector] )
    phi_array[2,.!delta_selector,t_index] .= 1 #.+ uniform_noise[delta_selector.==2].*0.05
    return nothing
end
function set_phi_as_ic_norm1!(phi_array::Array{TF,3}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "Uniform phi1"
    nparticles = size(phi_array)[2]
    phi_array[2,:,t_index] = abs.(phi_eps*randn(TF, nparticles)) #pdf can't find zeros
    phi_array[1,:,t_index] = randn(TF, nparticles)*TF(0.5)*(1/3).+TF(0.5) #high amount of mass already in system, truncation shouldn't cause problems
    reject=(phi_array[1,:,t_index].<=0) .| (phi_array[1,:,t_index].>1)
    while any(reject) #trucation via rejection sampling
        phi_array[1,reject,t_index] = randn(TF, sum(reject))*TF(0.5)*(1/3) .+TF(0.5)
        reject=(phi_array[1,:,t_index].<=0) .| (phi_array[1,:,t_index].>1)
    end
    return nothing
end

function set_phi_as_ic_norm1x!(phi_array::Array{TF,3},lengthscale::TF,xp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "Uniform phi1"
    nparticles = size(phi_array)[2]
    gaussian_space(x::TF) = exp(-TF(0.5)*((x-TF(0.5)*space_cells.length_domain)/lengthscale)^TF(2))
    phi_array[1,:,t_index] = gaussian_space.(xp)
    phi_array[2,:,t_index] = abs.(phi_eps*randn(TF, nparticles))
    return nothing
end

function set_phi_as_ic_normboth!(phi_array::Array{TF,3}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "Uniform phi1"
    nparticles = size(phi_array)[2]
    phi_array[2,:,t_index] = randn(TF, nparticles)*TF(0.5)*(1/3).+TF(0.5) #high amount of mass already in system, truncation shouldn't cause problems
    reject=(phi_array[2,:,t_index].<=0) .| (phi_array[2,:,t_index].>1)
    while any(reject) #trucation via rejection sampling
        phi_array[2,reject,t_index] = randn(TF, sum(reject))*TF(0.5)*(1/3) .+TF(0.5)
        reject=(phi_array[2,:,t_index].<=0) .| (phi_array[2,:,t_index].>1)
    end
    phi_array[1,:,t_index] = randn(TF, nparticles)*TF(0.5)*(1/3).+TF(0.5) #high amount of mass already in system, truncation shouldn't cause problems
    reject=(phi_array[1,:,t_index].<=0) .| (phi_array[1,:,t_index].>1)
    while any(reject) #trucation via rejection sampling
        phi_array[1,reject,t_index] = randn(TF, sum(reject))*TF(0.5)*(1/3) .+TF(0.5)
        reject=(phi_array[1,:,t_index].<=0) .| (phi_array[1,:,t_index].>1)
    end
    return nothing
end

function set_phi_as_ic_dd_diff!(phi_array::Array{TF,3}, K::TF, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "double delta difference" with constant K
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    local delta_selector = rand([true,false], nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    phi_array[2,delta_selector,t_index] .= sqrt(K)#.+(phi_eps*noise_term[delta_selector] )
    phi_array[1,delta_selector,t_index] .= sqrt(K)#.-(phi_eps*noise_term[delta_selector] )

    phi_array[1,.!delta_selector,t_index] .= 1 #.-(phi_eps*noise_term[.!delta_selector] )
    phi_array[2,.!delta_selector,t_index] .= K #.+(phi_eps*noise_term[.!delta_selector] )
    return nothing
end
function set_phi_as_ic_2l_diff!(phi_array::Array{TF,3},K::TF,yp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    #Initial_condition == "2 layers difference"
    nparticles = size(phi_array)[2]
    local noise_term = randn(TF, nparticles)
    # local uniform_noise = rand(nparticles).-TF(0.5)

    phi_array[2,yp.>TF(0.5)*space_cells.height_domain,t_index] .= sqrt(K) #.+abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )
    phi_array[1,yp.>TF(0.5)*space_cells.height_domain,t_index] .= sqrt(K)#.-abs.(phi_eps*noise_term[yp.>TF(0.5)*space_cells.height_domain] )

    phi_array[1,yp.<=TF(0.5)*space_cells.height_domain,t_index] .= 1 #.- abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
    phi_array[2,yp.<=TF(0.5)*space_cells.height_domain,t_index] .= K #.+ abs.(phi_eps*noise_term[yp.<=TF(0.5)*space_cells.height_domain] )
    return nothing
end

#a generic function to contain the switch case between ics
function set_phi_as_ic!(phi_array::Array{TF,3},IC_type::String,xp::Vector{TF},yp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    IC_type=lowercase(IC_type)
    if IC_type == "uniform phi_1"
        set_phi_as_ic_up1!(phi_array,t_index)
    elseif IC_type == "triple delta"
        set_phi_as_ic_td!(phi_array,t_index)
    elseif IC_type == "2 layers"
        set_phi_as_ic_2l!(phi_array,yp,space_cells,t_index)
    elseif IC_type == "double delta"
        set_phi_as_ic_dd!(phi_array,t_index)
    elseif IC_type == "centred normal"
        set_phi_as_ic_norm1!(phi_array,t_index)
    elseif IC_type == "centred 2 normal"
        set_phi_as_ic_normboth!(phi_array,t_index)
    elseif IC_type == "empty"
        set_phi_as_ic_empty!(phi_array,t_index)
    elseif IC_type in ["double delta difference","2 layers difference","1 layer transport, 1 layer empty","1 layer transport, 1 layer empty x"]
        throw(ArgumentError("Requires addtional parameters"))
    else
        throw(ArgumentError("Not a valid intitial condition"))
    end
    return nothing
end

function set_phi_as_ic!(phi_array::Array{TF,3},IC_type::Tuple{String,Vararg},xp::Vector{TF},yp::Vector{TF},space_cells::CellGrid{TF}, t_index::Int) where TF<:AbstractFloat
    IC_type_str=lowercase(IC_type[1])
    if length(IC_type)==1
        set_phi_as_ic!(phi_array,IC_type[1],xp,yp,space_cells,t_index)
    elseif IC_type_str == "uniform phi_1"
        set_phi_as_ic_up1!(phi_array,t_index,IC_type[2:end]...)
    elseif IC_type_str == "double delta difference"
        set_phi_as_ic_dd_diff!(phi_array,IC_type[2],t_index)
    elseif IC_type_str == "2 layers difference"
        set_phi_as_ic_2l_diff!(phi_array,IC_type[2],yp,space_cells,t_index)
    elseif IC_type_str == "1 layer transport, 1 layer empty"
        set_phi_as_ic_2l_one_empty!(phi_array,IC_type[2],yp,space_cells,t_index)
    elseif IC_type_str == "1 layer transport, 1 layer empty x"
        set_phi_as_ic_2l_one_empty_x!(phi_array,IC_type[2],xp,space_cells,t_index)
    elseif IC_type_str == "vertical strip"
        set_phi_as_ic_vert_strip!(phi_array,IC_type[2:end]...,xp,space_cells,t_index)
    elseif IC_type_str == "vertical strip difference"
        set_phi_as_ic_vert_strip_diff!(phi_array,IC_type[2:end]...,xp,space_cells,t_index)
    elseif IC_type_str == "x centred normal"
        set_phi_as_ic_norm1x!(phi_array,IC_type[2:end]...,xp,space_cells,t_index)
    else
        throw(ArgumentError("Not a valid intitial condition"))
    end
    return nothing
end


function assign_f_phi_cell!(f_phi_cell::AbstractArray{TF,5},phi_array::AbstractArray{TF,2}, psi_mesh::PsiGrid{TF}, cell_row::Int, cell_column::Int, t_index::Int) where TF <: AbstractFloat
    "use if cell_points has alreday been determined"
    phi_array_sort_ind_1 = sortperm(phi_array[1,:])
    phi1_i = 2 #upper bound index #assuming no values outside range
    prev_stop_1 = 1
    current_i = 0
    for i in phi_array_sort_ind_1
        current_i += 1
        while phi_array[1,i] > psi_mesh.psi_1[phi1_i]
            phi_array_sort_ind_2 = phi_array_sort_ind_1[sortperm(phi_array[2,phi_array_sort_ind_1[prev_stop_1:current_i-1]]).+prev_stop_1.-1]
            accum = 0
            phi2_i = 2
            for j in phi_array_sort_ind_2
                while phi_array[2,j] > psi_mesh.psi_2[phi2_i]
                    f_phi_cell[phi1_i-1, phi2_i-1, cell_row, cell_column, t_index] = float(accum)
                    accum = 0
                    phi2_i += 1
                end
                accum +=1
            end
            try 
                f_phi_cell[phi1_i-1, phi2_i-1, cell_row, cell_column, t_index] = float(accum)
            catch 
                println(phi1_i,' ', phi2_i,' ', cell_row,' ', cell_column,' ', t_index)
                println(maximum(phi_array[1,:]),' ',maximum(phi_array[2,:]))
                println(psi_mesh.phi_domain)
                rethrow()
            end
            prev_stop_1 = current_i
            phi1_i +=1
        end
    end
    #catch those for which there are no higher points
    phi_array_sort_ind_2 = phi_array_sort_ind_1[sortperm(phi_array[2,phi_array_sort_ind_1[prev_stop_1:current_i]]).+prev_stop_1.-1]
    accum = 0
    phi2_i = 2
    for j in phi_array_sort_ind_2
        while phi_array[2,j] > psi_mesh.psi_2[phi2_i]
            f_phi_cell[phi1_i-1, phi2_i-1, cell_row, cell_column, t_index] = float(accum)
            accum = 0
            phi2_i += 1
        end
        accum +=1
    end
    f_phi_cell[phi1_i-1, phi2_i-1, cell_row, cell_column, t_index] = float(accum)

    f_phi_cell[:, :,cell_row, cell_column, t_index] = f_phi_cell[:,:,cell_row, cell_column, t_index]./(size(phi_array)[2])
    return nothing
end 

function assign_f_phi!(f_phi_t::AbstractArray{TF},phi_array::AbstractArray{TF}, xp::AbstractVector{TF}, yp::AbstractVector{TF}, psi_mesh::PsiGrid{TF}, space_cells::CellGrid{TF}, t_index::Int) where TF <: AbstractFloat
    x_sort = sortperm(xp)
    x_i = 2 #assuming no values outside range
    prev_stop_x = 1
    current_i = 0
    for i in x_sort
        current_i += 1
        while xp[i] > space_cells.x_edges[x_i]
            y_sort = x_sort[sortperm(yp[x_sort[prev_stop_x:current_i-1]]).+(prev_stop_x-1)]
            prev_stop_y = 1
            y_i = 2
            current_j = 0
            for j in y_sort
                current_j += 1
                while yp[j] > space_cells.y_edges[y_i]
                    assign_f_phi_cell!(f_phi_t, phi_array[:,y_sort[prev_stop_y:current_j-1]], psi_mesh, y_i-1, x_i-1, t_index)
                    prev_stop_y = current_j
                    y_i+=1
                end
            end
            assign_f_phi_cell!(f_phi_t, phi_array[:,y_sort[prev_stop_y:current_j]], psi_mesh, y_i-1, x_i-1, t_index)
            prev_stop_x = current_i
            x_i +=1
        end
    end
    #catch final cell
    y_sort = x_sort[sortperm(yp[x_sort[prev_stop_x:current_i]]).+(prev_stop_x-1)]
    prev_stop_y = 1
    y_i = 2
    current_j = 0
    for j in y_sort
        current_j += 1
        while yp[j] > space_cells.y_edges[y_i]
            assign_f_phi_cell!(f_phi_t, phi_array[:,y_sort[prev_stop_y:current_j-1]], psi_mesh, y_i-1, x_i-1, t_index)
            prev_stop_y = current_j
            y_i+=1
        end
    end
    assign_f_phi_cell!(f_phi_t, phi_array[:,y_sort[prev_stop_y:current_j]], psi_mesh, y_i-1, x_i-1, t_index)
    return nothing
end 

function assign_f_edge_cell!(f_edge_cell::AbstractArray{TF,4},phi_array::AbstractArray{TF,2},psi_mesh::PsiGrid{TF}, cell_row::Int, cell_column::Int, t_index::Int) where TF <: AbstractFloat
    "use if cell_points has alreday been determined"
    phi_array_sort_ind_1 = sortperm(phi_array[1,:])
    phi1_i = 2 #upper bound index #assuming no values outside range
    prev_stop_1 = 1
    current_i = 0
    for i in phi_array_sort_ind_1
        current_i += 1
        while phi_array[1,i] > psi_mesh.psi_1[phi1_i]
            phi_array_sort_ind_2 = phi_array_sort_ind_1[sortperm(phi_array[2,phi_array_sort_ind_1[prev_stop_1:current_i-1]]).+prev_stop_1.-1]
            accum = 0
            phi2_i = 2
            for j in phi_array_sort_ind_2
                while phi_array[2,j] > psi_mesh.psi_2[phi2_i]
                    f_edge_cell[phi1_i-1, phi2_i-1, cell_row, t_index] = float(accum)
                    accum = 0
                    phi2_i += 1
                end
                accum +=1
            end
            try 
                f_edge_cell[phi1_i-1, phi2_i-1, cell_row, t_index] = float(accum)
            catch 
                println(phi1_i,' ', phi2_i,' ', cell_row,' ', cell_column,' ', t_index)
                println(maximum(phi_array[1,:]),' ',maximum(phi_array[2,:]))
                println(psi_mesh.phi_domain)
                rethrow()
            end
            prev_stop_1 = current_i
            phi1_i +=1
        end
    end
    #catch those for which there are no higher points
    phi_array_sort_ind_2 = phi_array_sort_ind_1[sortperm(phi_array[2,phi_array_sort_ind_1[prev_stop_1:current_i]]).+prev_stop_1.-1]
    accum = 0
    phi2_i = 2
    for j in phi_array_sort_ind_2
        while phi_array[2,j] > psi_mesh.psi_2[phi2_i]
            f_phi_cell[phi1_i-1, phi2_i-1, cell_row, t_index] = float(accum)
            accum = 0
            phi2_i += 1
        end
        accum +=1
    end
    f_edge_cell[phi1_i-1, phi2_i-1, cell_row, t_index] = float(accum)

    f_edge_cell[:, :,cell_row, t_index] = f_edge_cell[:,:,cell_row, t_index]./(size(phi_array)[2])
    return nothing
end 

function eval_by_cell!(func!::Function, xp::Vector{TF}, yp::Vector{TF}, space_cells::CellGrid{TF}) where TF <: AbstractFloat
    "func! is expected to be a function of signature func(cell_row,cell_column,cell_particles)->nothing"
    x_sort = sortperm(xp)
    x_i = 2 #assuming no values outside range
    prev_stop_x = 1
    current_i = 0
    for i in x_sort
        current_i += 1
        while xp[i] > space_cells.x_edges[x_i]
            y_sort = x_sort[sortperm(yp[x_sort[prev_stop_x:current_i-1]]).+(prev_stop_x-1)]
            prev_stop_y = 1
            y_i = 2
            current_j = 0
            for j in y_sort
                current_j += 1
                while yp[j] > space_cells.y_edges[y_i]
                    func!(y_i-1,x_i-1,y_sort[prev_stop_y:current_j-1])#calling function
                    prev_stop_y = current_j
                    y_i+=1
                end
            end
            func!(y_i-1,x_i-1,y_sort[prev_stop_y:current_j])#calling function
            prev_stop_x = current_i
            x_i +=1
        end
    end
    #catch final cell
    y_sort = x_sort[sortperm(yp[x_sort[prev_stop_x:current_i]]).+(prev_stop_x-1)]
    prev_stop_y = 1
    y_i = 2
    current_j = 0
    for j in y_sort
        current_j += 1
        while yp[j] > space_cells.y_edges[y_i]
            func!(y_i-1,x_i-1,y_sort[prev_stop_y:current_j-1])#calling function
            prev_stop_y = current_j
            y_i+=1
        end
    end
    func!(y_i-1,x_i-1,y_sort[prev_stop_y:current_j])#calling function
    return nothing
end 

#CLt/normal
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, turb_k_e::Vector{TF}, bc_params::BCParams{TF,Int, Nothing,false}, t_index::Int) where TF<:AbstractFloat
    n_abs = sum(abs_points)
    abs_k = bc_params.bc_k.*ones(TF, 2,n_abs)
    effective_v_particles =( phip[:,abs_points,t_index].*bc_params.num_vp)
    #K for Erban and Chapman approximation 
    P = zeros(TF, 2,n_abs)
    P[1,:] = min.(abs_k[1,:].*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    P[2,:] = min.(abs_k[2,:].*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    #by CLT approx dist for number of virtual particles to have reacted
    xi = randn(TF, 2,n_abs).*sqrt.((P.*(1 .-P)))
    #catching places where all mass has been removed
    xi = [effective_v_particles[i,j]>0 ? xi[i,j]./sqrt(effective_v_particles[i,j]) : 0 for i in 1:2, j in 1:n_abs]
    ratios = max.(min.((1 .-P) + xi,1),0)
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

#CLt/normal Precomp
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,Int, Nothing,false}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat
    n_abs = sum(abs_points)
    effective_v_particles =( phip[:,abs_points,t_index].*bc_params.num_vp)
    #K for Erban and Chapman approximation 
    P = zeros(TF, 2,n_abs)
    P .= Precomp_P
    #by CLT approx dist for number of virtual particles to have reacted
    xi = randn(TF, 2,n_abs).*sqrt.((P.*(1 .-P)))
    #catching places where all mass has been removed
    xi = [effective_v_particles[i,j]>0 ? xi[i,j]./sqrt(effective_v_particles[i,j]) : 0 for i in 1:2, j in 1:n_abs]
    ratios = max.(min.((1 .-P) + xi,1),0)
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

####Not being updated ####
#using binomal noise for small numbers of vparticles
#disabled to allow precomp to work
# function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, turb_k_e::Vector{TF}, bc_params::BCParams{TF,Int, Nothing,false}, t_index::Int) where TF<:AbstractFloat
#     n_abs = sum(abs_points)
#     abs_k = bc_params.bc_k.*ones(TF,2,n_abs)
#     effective_v_particles =( phip[:,abs_points,t_index].*bc_params.num_vp)
#     #K for Erban and Chapman approximation 
#     P = zeros(TF,2,n_abs)
#     P[1,:] = min.(abs_k[1,:].*sqrt.(bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
#     P[2,:] = min.(abs_k[2,:].*sqrt.(bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
#     #Binomal dist for number of virtual particles to have reacted
#     xi_dist = Binomial.(ceil.(effective_v_particles),1 .-P)
#     xi = [rand(xi_dist[i,j]) for i in 1:2, j in 1:n_abs]
#     ratios = [effective_v_particles[i,j]>0 ? xi[i,j]./ceil.(effective_v_particles[i,j]) : 0 for i in 1:2, j in 1:n_abs]
#     phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
#     return nothing
# end

# #binomal precomp
# function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,Int, Nothing,false}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat
#     n_abs = sum(abs_points)
#     effective_v_particles =( phip[:,abs_points,t_index].*bc_params.num_vp)
#     #K for Erban and Chapman approximation 
#     P = zeros(TF,2,n_abs)
#     P .= Precomp_P
#     #Binomal dist for number of virtual particles to have reacted
#     xi_dist = Binomial.(ceil.(effective_v_particles),1 .-P)
#     xi = [rand(xi_dist[i,j]) for i in 1:2, j in 1:n_abs]
#     ratios = [effective_v_particles[i,j]>0 ? xi[i,j]./ceil.(effective_v_particles[i,j]) : 0 for i in 1:2, j in 1:n_abs]
#     phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
#     return nothing
# end

#binary Precomp, need to edit this file to enable this, (comment out binomal precomp)
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,Int, Nothing,true}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat 
    #K for Erban and Chapman approximation 
    n_abs = sum(abs_points)
    xi=ones(size(phip)[2])
    xi[abs_points]=rand(n_abs)
    phip[:, xi.<Precomp_P, t_index] .=0 
    return nothing
end
####updates resume ####

#mean
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, turb_k_e::Vector{TF}, bc_params::BCParams{TF,TF, Nothing}, t_index::Int) where TF<:AbstractFloat 
    n_abs = sum(abs_points)
    abs_k = bc_params.bc_k.*ones(2,n_abs)
    #K for Erban and Chapman approximation 
    P = zeros(TF,2,n_abs)
    P[1,:] = min.(abs_k[1,:].*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    P[2,:] = min.(abs_k[2,:].*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)
    ratios = 1 .-P #taking mean for limiting case
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

#mean Precomp
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,TF, Nothing}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat 
    #K for Erban and Chapman approximation 
    ratios = 1 - Precomp_P #taking mean for limiting case
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

###add in non-liner sorbtion as a correction factor from linear

#CLt/normal Precomp
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,Int,func_T,false}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat where func_T<:Function
    n_abs = sum(abs_points)
    effective_v_particles =( phip[:,abs_points,t_index].*bc_params.num_vp)
    #K for Erban and Chapman approximation 
    P = zeros(TF, 2,n_abs)
    phi_vec = [phip[:,abs_points,t_index][:,i] for i=1:n_abs]
    corr_vec = bc_params.corr_factor.(phi_vec)
    corr_arr = [corr_vec[j][i] for i=1:2, j=1:n_abs]
    P .= Precomp_P.*corr_arr
    #by CLT approx dist for number of virtual particles to have reacted
    xi = randn(TF, 2,n_abs).*sqrt.(abs.(P.*(1 .-P)))
    #catching places where all mass has been removed
    xi = [effective_v_particles[i,j]>0 ? xi[i,j]./sqrt(effective_v_particles[i,j]) : 0 for i in 1:2, j in 1:n_abs]
    ratios = max.(min.((1 .-P) + xi,1),0)
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

#mean Precomp
function bc_absorbtion!(phip::Array{TF,3}, abs_points::BitVector, bc_params::BCParams{TF,TF,func_T}, t_index::Int, Precomp_P::TF) where TF<:AbstractFloat where func_T<:Function
    #K for Erban and Chapman approximation 
    n_abs=sum(abs_points)
    phi_vec = [phip[:,abs_points,t_index][:,i] for i=1:n_abs]
    corr_vec = bc_params.corr_factor.(phi_vec)
    corr_arr = [corr_vec[j][i] for i=1:2, j=1:n_abs]
    ratios = 1 .- Precomp_P.*corr_arr #taking mean for limiting case
    phip[:, abs_points, t_index] = phip[:, abs_points, t_index].*ratios
    return nothing
end

###

function particle_motion_model(x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::Array{T,2}, m_params::MotionParams{T}, dt::T, space_cells::CellGrid{T}) where T<:AbstractFloat
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    u_mean=m_params.u_mean
    np = size(x_pos)[1]
    nt = size(x_pos)[2]-1
    bc_interact = falses(np, nt, 4)
    #intitial vaules of velocity, maintaining consitancy with energy
    uxp = randn(T, np).*sqrt.(2/3 .*turb_k_e[:,1])
    uyp = randn(T, np).*sqrt.(2/3 .*turb_k_e[:,1])
    for t=1:nt
        x_pos[:,t+1] = x_pos[:,t] + (uxp_full)*dt # random walk in x-direction
        y_pos[:,t+1] = y_pos[:,t] + uyp*dt # random walk in y-direction
        uxp = uxp+(-T(0.5)*B*omega_bar*uxp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e[:,t].*omega_bar.*dt); 
        uyp = uyp+(-T(0.5)*B*omega_bar*uyp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e[:,t].*omega_bar.*dt); 
        for i in 1:space_cells.y_res
            in_y = space_cells.y_edges[i].<y_pos[:,t].<space_cells.y_edges[i+1]
            for j in 1:x_res
                in_x = space_cells.x_edges[j].<x_pos[:,t].<space_cells.x_edges[j+1]
                cell_particles = findall(in_x.&in_y)
                turb_k_e[cell_particles,t+1].=T(0.5)*(st.mean(uxp[cell_particles].^2)+st.mean(uyp[cell_particles].^2))*1.5 #turb_e_init;
            end
        end
        uxp_full = u_mean .+ uxp

            # Reflection particles at boundaries

        # Reflection at upper boundary y>height_domain
        # doing closed on top open on bottom, as cell detection is open on top,
        # closed on bottom
        mag = findall(y_pos[:,t+1].>=space_cells.height_domain) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        V1 = space_cells.height_domain.*ones(T, dim_mag) 

        ypr_mag = V1*2 .- y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp>1 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,1] .= true

        # Reflection at lower boundary y<0
        mag = findall(y_pos[:,t+1].<=0) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        ypr_mag = - y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp>1 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,2] .= true

        #bc at end (y=length_domain) of domain
        end_indicies = x_pos[:,t+1].>=space_cells.length_domain #index of particle with xp>length

        end_x = x_pos[end_indicies,t+1]
        xpr_end = end_x .- space_cells.length_domain #shifting particles back to begining
        x_pos[end_indicies,t+1] = xpr_end #replacing x coords

        bc_interact[end_indicies,t,3] .= true

        #bc at start (x=0) of domain
        start_indicies = x_pos[:,t+1].<=0 #index of particle with xp>length

        xpr_start = space_cells.length_domain .+ x_pos[start_indicies,t+1] 
        x_pos[start_indicies,t+1] = xpr_start #replacing x coords
        bc_interact[start_indicies,t,4] .= true
    end
    return bc_interact
end

function particle_motion_model(x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::T, m_params::MotionParams{T}, dt::T, space_cells::CellGrid{T}) where T<:AbstractFloat
    #for constant kinetic energy
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    u_mean=m_params.u_mean
    np = size(x_pos)[1]
    nt = size(x_pos)[2]-1
    bc_interact = falses(np, nt, 4)#3rd index is for: upper, lower, right, left
    #intitial vaules of velocity, maintaining consitancy with energy
    uxp = randn(T, np).*sqrt.(2/3 .*turb_k_e)
    uyp = randn(T, np).*sqrt.(2/3 .*turb_k_e)
    for t=1:nt
        x_pos[:,t+1] = x_pos[:,t] + (uxp_full)*dt # random walk in x-direction
        y_pos[:,t+1] = y_pos[:,t] + uyp*dt # random walk in y-direction
        uxp = uxp+(-T(0.5)*B*omega_bar*uxp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 
        uyp = uyp+(-T(0.5)*B*omega_bar*uyp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 
        uxp_full = u_mean .+ uxp

            # Reflection particles at boundaries

        # Reflection at upper boundary y>height_domain
        # doing closed on top open on bottom, as cell detection is open on top,
        # closed on bottom
        mag = findall(y_pos[:,t+1].>=space_cells.height_domain) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        V1 = space_cells.height_domain.*ones(T, dim_mag) 

        ypr_mag = V1*2 .- y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp>1 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,1] .= true

        # Reflection at lower boundary y<0
        mag = findall(y_pos[:,t+1].<=0) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        ypr_mag = - y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp<0 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,2] .= true

        #bc at end (y=length_domain) of domain
        end_indicies = x_pos[:,t+1].>=space_cells.length_domain #index of particle with xp>length

        end_x = x_pos[end_indicies,t+1]
        xpr_end = end_x .- space_cells.length_domain #shifting particles back to begining
        x_pos[end_indicies,t+1] = xpr_end #replacing x coords

        bc_interact[end_indicies,t,3] .= true

        #bc at start (x=0) of domain
        start_indicies = x_pos[:,t+1].<=0 #index of particle with xp>length

        xpr_start = space_cells.length_domain .+ x_pos[start_indicies,t+1] 
        x_pos[start_indicies,t+1] = xpr_start #replacing x coords
        bc_interact[start_indicies,t,4] .= true
    end
    return bc_interact
end

#reflective bc at begining
function particle_motion_model_ref_start(x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::T, m_params::MotionParams{T}, dt::T, space_cells::CellGrid{T}) where T<:AbstractFloat
    #for constant kinetic energy
    omega_bar=m_params.omega_bar
    C_0=m_params.C_0
    B=m_params.B
    u_mean=m_params.u_mean
    np = size(x_pos)[1]
    nt = size(x_pos)[2]-1
    bc_interact = falses(np, nt, 4)
    #intitial vaules of velocity, maintaining consitancy with energy
    uxp = randn(T, np).*sqrt.(2/3 .*turb_k_e)
    uyp = randn(T, np).*sqrt.(2/3 .*turb_k_e)
    for t=1:nt
        uxp_full = u_mean .+ uxp
        x_pos[:,t+1] = x_pos[:,t] + (uxp_full)*dt # random walk in x-direction
        y_pos[:,t+1] = y_pos[:,t] + uyp*dt # random walk in y-direction
        uxp = uxp+(-T(0.5)*B*omega_bar*uxp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 
        uyp = uyp+(-T(0.5)*B*omega_bar*uyp)*dt+randn(T, np).*sqrt.(C_0.*turb_k_e.*omega_bar.*dt); 

            # Reflection particles at boundaries

        # Reflection at upper boundary y>height_domain
        # doing closed on top open on bottom, as cell detection is open on top,
        # closed on bottom
        mag = findall(y_pos[:,t+1].>=space_cells.height_domain) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        V1 = space_cells.height_domain.*ones(T, dim_mag) 

        ypr_mag = V1*2 .- y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp>1 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,1] .= true

        # Reflection at lower boundary y<0
        mag = findall(y_pos[:,t+1].<=0) # index of particle with yp>height_domain
        dim_mag = size(mag) # dimension of array "mag"

        y_mag_succ = y_pos[mag,t+1] # yp at time t+1 corresponding to the index "mag"

        ypr_mag = - y_mag_succ  # yp at time t+1 of the reflected particle

        y_pos[mag,t+1]= ypr_mag #replacement of yp<0 with yp of reflected particle
        uyp[mag] = -uyp[mag] #reflecting velocity
        bc_interact[mag,t,2] .= true

        #bc at end (y=length_domain) of domain
        end_indicies = x_pos[:,t+1].>=space_cells.length_domain #index of particle with xp>length

        end_x = x_pos[end_indicies,t+1]
        xpr_end = end_x .- space_cells.length_domain #shifting particles back to begining
        x_pos[end_indicies,t+1] = xpr_end #replacing x coords

        bc_interact[end_indicies,t,3] .= true

        #bc at start (x=0) of domain
        start_indicies = x_pos[:,t+1].<=0 #index of particle with xp>length

        xpr_start = .- x_pos[start_indicies,t+1] 
        x_pos[start_indicies,t+1] = xpr_start #replacing x coords
        uxp[start_indicies] = -uxp[start_indicies] #reflecting velocity
        bc_interact[start_indicies,t,4] .= true
    end
    return bc_interact
end


function PSP_model!(f_phi::Array{T,5},x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::Array{T,2}, bc_interact::BitArray{3}, dt::T, initial_condition::Union{String,Tuple{String,Vararg}},  p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    np, nt = size(x_pos)
    nt-=1

    phip = zeros((2, np, 1+1)) #scalar concentration at these points
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle

    set_phi_as_ic!(phip,initial_condition,x_pos[:,1],y_pos[:,1],space_cells,1)
    assign_f_phi!(f_phi,phip[:,:,1], x_pos[:,1], y_pos[:,1], psi_mesh, space_cells,1)

    omegap = zeros(T, np,nt+1) #turbulence frequency
    omega0_dist = Gamma((omega_mean-p_params.omega_min)^2/(omega_sigma_2),(omega_sigma_2)/(omega_mean-p_params.omega_min)) #this should now match long term distribution of omega
    omegap[:,1] = rand(T, omega0_dist, np)
    
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip[:,:,1], cell_particles, cell_particles)
    ;assign_f_phi_cell!(f_phi,phip[:,cell_particles,1],psi_mesh,i,j,1);return nothing) , x_pos[:,1], y_pos[:,1], space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed
    t_decorr_p = 1 ./(c_t.*omegap[phi_pm[1,:],1]).*rand(T,np)
    t_decorr_m = 1 ./(c_t.*omegap[phi_pm[2,:],1]).*rand(T,np)

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle],1]-phip[:,particle,1]),(phip[:,phi_pm[2,particle],1]-phip[:,particle,1]))
    end

    for t in 1:nt
        verbose && print(t,' ')
        # print(maximum(phip[:,:,t]),' ')
        #E-M solver for omega 
        dw = sqrt(dt).*randn(T, np) #random draws
        omegap[:,t+1] = omegap[:,t]-(omegap[:,t].-omega_mean)./T_omega.*dt + sqrt.((omegap[:,t].-p_params.omega_min).*(2*omega_sigma_2*omega_mean/T_omega)).*dw
        omegap[:,t+1] = omegap[:,t+1].*(omegap[:,t+1].>=p_params.omega_min)+p_params.omega_min.*(omegap[:,t+1].<=p_params.omega_min) #enforcing positivity

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
            (length(t_p0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_p0_cell, cell_particles, 1)
            (length(t_m0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_m0_cell, cell_particles, 2)
            (length(t_pm0_cell)>0)&&assign_pm!(phi_pm, phip[:,:,1], t_pm0_cell, cell_particles)
            #update pairs to ensure all are within the same bounds
            pm_check_and_recal_for_cell_change!(phi_pm, phip[:,:,1], cell_particles)
            return nothing
        end, x_pos[:,t], y_pos[:,t], space_cells)
        #reset decorrelation time for particles it had run out on
        t_decorr_p[t_p0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[1,t_p0.&t_pm0],t])
        t_decorr_m[t_m0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[2,t_m0.&t_pm0],t])

        phi_c = T(0.5).*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
        diffusion = zeros(2,np)
        diffusion[1,:] = (phip[1,:,1]-phi_c[1,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        diffusion[2,:] = (phip[2,:,1]-phi_c[2,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        # reaction = dt.*(reaction).*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
        dphi = diffusion 

        # ensuring mean 0 change
        # generating a random orthonormal basis
        # is 2-d so genrate a random unit vector from an angle and proceed based
        # on that
        angle = 2*pi*rand(1)[1];
        e_1 = [cos(angle),sin(angle)]
        handedness = sb.sample([-1,1],1)[1] #randomly choose betwen left or right handed system
        e_2 = handedness*[e_1[2],-e_1[1]]
        T_mat=zeros(T,2,2)
        T_mat[:,1] = e_1  #coord transform matrix
        T_mat[:,2] = e_2
        dphi = T_mat\dphi  #transform to new coords
        #performing adjustment to mean 0
        corr_factor = zeros(T, 2,np)
        
        eval_by_cell!(function (i,j,cell_particles)any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1]
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
        end, x_pos[:,t], y_pos[:,t], space_cells)
        dphi = corr_factor.*dphi
        dphi = T_mat*dphi #return to old coords

        #reaction has to be done after mean zero correction - or it has no effect
        reaction = zeros(T, 2,np) # body reaction
        reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:,1],phip[2,:,1]))#.*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
        reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:,1],phip[2,:,1]))
        dphi .+= reaction
        phip[:,:,1+1] = phip[:,:,1]+dphi
        if !(initial_condition == "triple delta")
            phip[:,:,1+1] = phip[:,:,1+1].*(phip[:,:,1+1].>0) #forcing positive concentration
        end

        bc_absorbtion!(phip,any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],turb_k_e[any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],t+1],bc_params,1+1) #currently only reacting on bottom bc

        assign_f_phi!(f_phi,phip[:,:,1+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
        phip[:,:,1] = phip[:,:,2]
        # print(maximum(phip[:,:,t]),' ')
    end
    verbose && println("end")
    return nothing
end

#constant turb_k_e
function PSP_model!(f_phi::Array{T,5},x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::T, bc_interact::BitArray{3}, dt::T, initial_condition::Union{String,Tuple{String,Vararg}},  p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    np, nt = size(x_pos)
    nt-=1
    precomp_P = min.(bc_params.bc_k.*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)

    phip = zeros(T, (2, np, 2)) #scalar concentration at these points
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle

    set_phi_as_ic!(phip,initial_condition,x_pos[:,1],y_pos[:,1],space_cells,1)
    assign_f_phi!(f_phi,phip[:,:,1], x_pos[:,1], y_pos[:,1], psi_mesh, space_cells,1)

    omegap = zeros(T, np,nt+1) #turbulence frequency
    omega0_dist = Gamma((omega_mean-p_params.omega_min)^2/(omega_sigma_2),(omega_sigma_2)/(omega_mean-p_params.omega_min)) #this should now match long term distribution of omega
    omegap[:,1] = rand(omega0_dist, np).+p_params.omega_min
    
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip[:,:,1], cell_particles, cell_particles)
    ;assign_f_phi_cell!(f_phi,phip[:,cell_particles,1],psi_mesh,i,j,1);return nothing) , x_pos[:,1], y_pos[:,1], space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed
    t_decorr_p = 1 ./(c_t.*omegap[phi_pm[1,:],1]).*rand(T,np)
    t_decorr_m = 1 ./(c_t.*omegap[phi_pm[2,:],1]).*rand(T,np)

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle],1]-phip[:,particle,1]),(phip[:,phi_pm[2,particle],1]-phip[:,particle,1]))
    end

    for t in 1:nt
        verbose && print(t,' ')
        #E-M solver for omega 
        dw = sqrt(dt).*randn(T, np) #random draws
        omegap[:,t+1] = omegap[:,t]-(omegap[:,t].-omega_mean)./T_omega.*dt + sqrt.((omegap[:,t].-p_params.omega_min).*(2*omega_sigma_2*omega_mean/T_omega)).*dw
        omegap[:,t+1] = omegap[:,t+1].*(omegap[:,t+1].>=p_params.omega_min)+p_params.omega_min.*(omegap[:,t+1].<=p_params.omega_min) #enforcing positivity

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
            (length(t_p0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_p0_cell, cell_particles, 1)
            (length(t_m0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_m0_cell, cell_particles, 2)
            (length(t_pm0_cell)>0)&&assign_pm!(phi_pm, phip[:,:,1], t_pm0_cell, cell_particles)
            #update pairs to ensure all are within the same bounds
            pm_check_and_recal_for_cell_change!(phi_pm, phip[:,:,1], cell_particles)
            return nothing
        end, x_pos[:,t], y_pos[:,t], space_cells)
        #reset decorrelation time for particles it had run out on
        t_decorr_p[t_p0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[1,t_p0.&t_pm0],t])
        t_decorr_m[t_m0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[2,t_m0.&t_pm0],t])

        phi_c = T(0.5).*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
        diffusion = zeros(T, 2,np)
        diffusion[1,:] = (phip[1,:,1]-phi_c[1,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,1].*dt).-1.0)
        diffusion[2,:] = (phip[2,:,1]-phi_c[2,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,1].*dt).-1.0)
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
        end, x_pos[:,t], y_pos[:,t], space_cells)

        dphi = corr_factor.*dphi
        dphi = T_mat*dphi #return to old coords

        #reaction has to be done after mean zero correction - or it has no effect
        reaction = zeros(T, 2,np) # bulk reaction
        reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:,1],phip[2,:,1]))#.*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
        reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:,1],phip[2,:,1]))
        dphi .+= reaction
        phip[:,:,1+1] = phip[:,:,1]+dphi
        if !(initial_condition == "triple delta")
            phip[:,:,1+1] = phip[:,:,1+1].*(phip[:,:,1+1].>0) #forcing positive concentration
        end

        bc_absorbtion!(phip,any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1+1, precomp_P) #currently only reacting on bottom bc

        assign_f_phi!(f_phi,phip[:,:,1+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
        # print(maximum(phip[:,:,t]),' ')
        phip[:,:,1] = phip[:,:,2]
    end
    verbose && println("end")
    return nothing
end

function PSP_model_record_phi_local_diff!(gphi::AbstractArray{T,3},f_phi::AbstractArray{T,5},x_pos::AbstractArray{T,2},y_pos::AbstractArray{T,2}, turb_k_e::T, bc_interact::BitArray{3}, dt::T, initial_condition::Union{String,Tuple{String,Vararg}},  p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    np, nt = size(x_pos)
    nt-=1
    precomp_P = min.(bc_params.bc_k.*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)

    phip = zeros(T, (2, np, 2)) #scalar concentration at these points, 2 is at t+1, 1 is at t
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle
    gamma = zeros(T, (2, np))

    set_phi_as_ic!(phip,initial_condition,x_pos[:,1],y_pos[:,1],space_cells,1)
    assign_f_phi!(f_phi,phip[:,:,1], x_pos[:,1], y_pos[:,1], psi_mesh, space_cells,1)

    omegap = zeros(T, np,nt+1) #turbulence frequency
    omega0_dist = Gamma((omega_mean-p_params.omega_min)^2/(omega_sigma_2),(omega_sigma_2)/(omega_mean-p_params.omega_min)) #this should now match long term distribution of omega
    omegap[:,1] = rand(omega0_dist, np).+p_params.omega_min
    
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip[:,:,1], cell_particles, cell_particles)
    ;assign_f_phi_cell!(f_phi,phip[:,cell_particles,1],psi_mesh,i,j,1);return nothing) , x_pos[:,1], y_pos[:,1], space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed + uniform so doesn't depend on start time 
    t_decorr_p = 1 ./(c_t.*omegap[phi_pm[1,:],1]).*rand(T,np)
    t_decorr_m = 1 ./(c_t.*omegap[phi_pm[2,:],1]).*rand(T,np)

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle],1]-phip[:,particle,1]),(phip[:,phi_pm[2,particle],1]-phip[:,particle,1]))
    end

    mean_phi = zeros(T,2,nt+1)
    mean_phi[:,1] = mean(phip[:,:,1],dims=2)[:,1]

    for t in 1:nt
        verbose && print(t,' ')
        #E-M solver for omega 
        dw = sqrt(dt).*randn(T, np) #random draws
        omegap[:,t+1] = omegap[:,t]-(omegap[:,t].-omega_mean)./T_omega.*dt + sqrt.((omegap[:,t].-p_params.omega_min).*(2*omega_sigma_2*omega_mean/T_omega)).*dw
        omegap[:,t+1] = omegap[:,t+1].*(omegap[:,t+1].>=p_params.omega_min)+p_params.omega_min.*(omegap[:,t+1].<=p_params.omega_min) #enforcing positivity

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
            (length(t_p0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_p0_cell, cell_particles, 1)
            (length(t_m0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_m0_cell, cell_particles, 2)
            (length(t_pm0_cell)>0)&&assign_pm!(phi_pm, phip[:,:,1], t_pm0_cell, cell_particles)
            #update pairs to ensure all are within the same bounds
            pm_check_and_recal_for_cell_change!(phi_pm, phip[:,:,1], cell_particles)
            return nothing
        end, x_pos[:,t], y_pos[:,t], space_cells)
        #reset decorrelation time for particles it had run out on
        t_decorr_p[t_p0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[1,t_p0.&t_pm0],t])
        t_decorr_m[t_m0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[2,t_m0.&t_pm0],t])

        phi_c = T(0.5).*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
        diffusion = zeros(T, 2,np)
        diffusion[1,:] = (phip[1,:,1]-phi_c[1,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        diffusion[2,:] = (phip[2,:,1]-phi_c[2,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        dphi = diffusion
        #the local diffusion for each particle
        gamma[1,:] = c_phi.*T(0.5).*omegap[:,t].*(phip[1,:,1]-phi_c[1,:])
        gamma[2,:] = c_phi.*T(0.5).*omegap[:,t].*(phip[2,:,1]-phi_c[2,:])

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
        end, x_pos[:,t], y_pos[:,t], space_cells)

        dphi = corr_factor.*dphi
        dphi = T_mat*dphi #return to old coords

        #reaction has to be done after mean zero correction - or it has no effect
        reaction = zeros(T, 2,np) # body reaction
        reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:,1],phip[2,:,1]))#.*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
        reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:,1],phip[2,:,1]))
        dphi .+= reaction
        phip[:,:,1+1] = phip[:,:,1]+dphi
        mean_phi[:,t+1] = mean(phip[:,:,2],dims=2)[:,1]
        if !(initial_condition == "triple delta")
            phip[:,:,1+1] = phip[:,:,1+1].*(phip[:,:,1+1].>0) #forcing positive concentration
        end

        eval_by_cell!(function (i,j,cell_particles)
            gphi[i,j,t]=mean(gamma[1,cell_particles].*phip[1,cell_particles,1])
            return nothing
        end, x_pos[:,t], y_pos[:,t], space_cells)
        
        bc_absorbtion!(phip,any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,2, precomp_P) #currently only reacting on bottom bc
        
        assign_f_phi!(f_phi, phip[:,:,1+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
        phip[:,:,1] = phip[:,:,2]
        # print(maximum(phip[:,:,t]),' ')
    end
    # plt=plot(permutedims( mean_phi))
    # display(plt)
    verbose && println("end")
    return nothing
end

function PSP_model_record_reacting_mass!(edge_mean::AbstractArray{T,1}, edge_squared::AbstractArray{T,1}, edge_squared_v::AbstractArray{T,1}, f_phi::Array{T,5},x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::T, bc_interact::BitArray{3}, dt::T, initial_condition::Union{String,Tuple{String,Vararg}},  p_params::PSPParams{T}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    omega_mean=p_params.omega_bar
    omega_sigma_2 = p_params.omega_sigma_2
    T_omega = p_params.T_omega
    c_phi = p_params.c_phi
    c_t = p_params.c_t
    np, nt = size(x_pos)
    nt-=1
    precomp_P = min.(bc_params.bc_k*sqrt.(bc_params.B*pi*m_params.omega_bar/(bc_params.D_mol+T(0.5)*bc_params.omega_bar*bc_params.C_0*turb_k_e)),1)

    phip = zeros(T, (2, np, 1+1)) #scalar concentration at these points
    phi_pm = zeros(Int, 2, np) #pm pairs for each particle
    set_phi_as_ic!(phip,initial_condition,x_pos[:,1],y_pos[:,1],space_cells,1)

    omegap = zeros(T, np,nt+1) #turbulence frequency
    omega0_dist = Gamma((omega_mean-p_params.omega_min)^2/(omega_sigma_2),(omega_sigma_2)/(omega_mean-p_params.omega_min)) #this should now match long term distribution of omega
    omegap[:,1] = rand(omega0_dist, np).+p_params.omega_min
    
    eval_by_cell!((i,j,cell_particles)-> (assign_pm!(phi_pm, phip[:,:,1], cell_particles, cell_particles)
    ;assign_f_phi_cell!(f_phi,phip[:,cell_particles,1],psi_mesh,i,j,1);return nothing) , x_pos[:,1], y_pos[:,1], space_cells)

    #time stamp until new p/m bound found, needed to ensure particles are
    #decorreltaed
    t_decorr_p = 1 ./(c_t.*omegap[phi_pm[1,:],1]).*rand(T,np)
    t_decorr_m = 1 ./(c_t.*omegap[phi_pm[2,:],1]).*rand(T,np)

    function test_dot(particle)#used to check if bounding condition is fulfilled
        la.dot((phip[:,phi_pm[1,particle],1]-phip[:,particle,1]),(phip[:,phi_pm[2,particle],1]-phip[:,particle,1]))
    end

    for t in 1:nt
        verbose && print(t,' ')
        #E-M solver for omega 
        dw = sqrt(dt).*randn(T, np) #random draws
        omegap[:,t+1] = omegap[:,t]-(omegap[:,t].-omega_mean)./T_omega.*dt + sqrt.((omegap[:,t].-p_params.omega_min).*(2*omega_sigma_2*omega_mean/T_omega)).*dw
        omegap[:,t+1] = omegap[:,t+1].*(omegap[:,t+1].>=p_params.omega_min)+p_params.omega_min.*(omegap[:,t+1].<=p_params.omega_min) #enforcing positivity

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
            (length(t_p0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_p0_cell, cell_particles, 1)
            (length(t_m0_cell)>0)&&assign_pm_single!(phi_pm, phip[:,:,1],t_m0_cell, cell_particles, 2)
            (length(t_pm0_cell)>0)&&assign_pm!(phi_pm, phip[:,:,1], t_pm0_cell, cell_particles)
            #update pairs to ensure all are within the same bounds
            pm_check_and_recal_for_cell_change!(phi_pm, phip[:,:,1], cell_particles)
            return nothing
        end, x_pos[:,t], y_pos[:,t], space_cells)
        #reset decorrelation time for particles it had run out on
        t_decorr_p[t_p0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[1,t_p0.&t_pm0],t])
        t_decorr_m[t_m0.&t_pm0] = 1 ./(c_t.*omegap[phi_pm[2,t_m0.&t_pm0],t])

        phi_c = T(0.5).*(phip[:,phi_pm[1,:],1]+phip[:,phi_pm[2,:],1])
        diffusion = zeros(T, 2,np)
        diffusion[1,:] = (phip[1,:,1]-phi_c[1,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        diffusion[2,:] = (phip[2,:,1]-phi_c[2,:]).*(exp.(-c_phi.*T(0.5).*omegap[:,t].*dt).-1.0)
        # reaction = dt.*(reaction).*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
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
        end, x_pos[:,t], y_pos[:,t], space_cells)

        dphi = corr_factor.*dphi
        dphi = T_mat*dphi #return to old coords

        #reaction has to be done after mean zero correction - or it has no effect
        reaction = zeros(T, 2,np) # body reaction
        reaction[1,:] = dt.*(p_params.reaction_form[1].(phip[1,:,1],phip[2,:,1]))#.*exp.(c_phi.*T(0.5).*omegap[:,t].*dt) #integration of reation term to match diffusion scheme, uncomment if reaction !=0
        reaction[2,:] = dt.*(p_params.reaction_form[2].(phip[1,:,1],phip[2,:,1]))
        dphi .+= reaction
        phip[:,:,1+1] = phip[:,:,1]+dphi
        if !(initial_condition == "triple delta")
            phip[:,:,1+1] = phip[:,:,1+1].*(phip[:,:,1+1].>0) #forcing positive concentration
        end
        eval_by_cell!(function (i,j,cell_p)
            if i==1
                edge_mean[t] += sum(phip[1,cell_p[bc_interact[cell_p,t,2]],1+1]) / (length(cell_p)*space_cells.x_res)
                # println(length(cell_p),' ', edge_mean[t],' ',sum(phip[1,cell_p[bc_interact[cell_p,t,2]],t+1]) / (length(cell_p)*space_cells.x_res))
                edge_squared[t] += sum(phip[1,cell_p[bc_interact[cell_p,t,2]],1+1].^2) / (length(cell_p)*space_cells.x_res)
                if t>=2
                    v_t=y_pos[cell_p[bc_interact[cell_p,t,2]],t]-y_pos[cell_p[bc_interact[cell_p,t,2]],t-1]
                    println(maximum(v_t),' ',minimum(v_t))
                    edge_squared_v[t-1] += sum(v_t.*phip[1,cell_p[bc_interact[cell_p,t,2]],1].^2) / (length(cell_p)*space_cells.x_res)
                end
            end
            return nothing
        end,  x_pos[:,t], y_pos[:,t], space_cells)

        bc_absorbtion!(phip,any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],bc_params,1+1, precomp_P) #currently only reacting on bottom bc

        assign_f_phi!(f_phi,phip[:,:,1+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
        # print(maximum(phip[:,:,t]),' ')
        phip[:,:,1] = phip[:,:,2]
    end
    verbose && println("end")
    return nothing
end

function make_f_phi_no_PSP!(f_phi::Array{T,5},x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::Array{T,2}, bc_interact::BitArray{3}, initial_condition::Union{String,Tuple{String,Vararg}}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    np, nt = size(x_pos)
    nt-=1
    
    phip = zeros(T, (2, np, nt+1)) #scalar concentration at these points

    isa(initial_condition, String) && (initial_condition=(initial_condition,))
    if initial_condition[1] == "Uniform phi_1"
        set_phi_as_ic_up1!(phip,1)
    elseif initial_condition[1] == "triple delta"
        set_phi_as_ic_td!(phip,1)
    elseif initial_condition[1] == "2 layers"
        set_phi_as_ic_2l!(phip,y_pos[:,1],space_cells,1)
    elseif initial_condition[1] == "double delta"
        set_phi_as_ic_dd!(phip,1)
    elseif initial_condition[1] == "centred normal"
        set_phi_as_ic_norm1!(phip,1)
    elseif initial_condition[1] == "centred 2 normal"
        set_phi_as_ic_normboth!(phip,1)
    elseif initial_condition[1] == "double delta difference"
        set_phi_as_ic_dd_diff!(phip,initial_condition[2],1)
        throw(ArgumentError("Not a valid intitial condition"))
    end
    assign_f_phi!(f_phi,phip[:,:,1], x_pos[:,1], y_pos[:,1], psi_mesh, space_cells,1)
    
    for t in 1:nt
        verbose && print(t,' ')
        bc_absorbtion!(phip,any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],turb_k_e[any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1],t],bc_params,t) #currently only reacting on bottom bc
        phip[:,:,t+1] = phip[:,:,t]
        assign_f_phi!(f_phi,phip[:,:,t+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
    end
    verbose && println("end")
    return nothing
end

#constant turb_k_e TODO:integrate with rewritten api 
function make_f_phi_no_PSP!(f_phi::Array{T,5},x_pos::Array{T,2},y_pos::Array{T,2}, turb_k_e::T, bc_interact::BitArray{3}, initial_condition::Union{String,Tuple{String,Vararg}}, psi_mesh::PsiGrid{T}, space_cells::CellGrid{T}, bc_params::BCParams{T}, verbose::Bool=false) where T<:AbstractFloat
    np, nt = size(x_pos)
    nt-=1
    precomp_P = min.(bc_params.bc_k.*sqrt.(2*bc_params.B.*pi./(bc_params.C_0.*turb_k_e)),1)

    phip = zeros(T, (2, np, nt+1)) #scalar concentration at these points

    isa(initial_condition, String) && (initial_condition=(initial_condition,))
    if initial_condition[1] == "Uniform phi_1"
        set_phi_as_ic_up1!(phip,1)
    elseif initial_condition[1] == "triple delta"
        set_phi_as_ic_td!(phip,1)
    elseif initial_condition[1] == "2 layers"
        set_phi_as_ic_2l!(phip,y_pos[:,1],space_cells,1)
    elseif initial_condition[1] == "double delta"
        set_phi_as_ic_dd!(phip,1)
    elseif initial_condition[1] == "centred normal"
        set_phi_as_ic_norm1!(phip,1)
    elseif initial_condition[1] == "centred 2 normal"
        set_phi_as_ic_normboth!(phip,1)
    elseif initial_condition[1] == "double delta difference"
        set_phi_as_ic_dd_diff!(phip,initial_condition[2],1)
    elseif initial_condition[1] == "2 layers difference"
        set_phi_as_ic_2l_diff!(phip,initial_condition[2],y_pos[:,1],space_cells,1)
    else
        throw(ArgumentError("Not a valid intitial condition"))
    end
    assign_f_phi!(f_phi,phip[:,:,1], x_pos[:,1], y_pos[:,1], psi_mesh, space_cells,1)
    
    for t in 1:nt
        verbose && print(t,' ')
        bc_absorbtion!(phip, any(bc_interact[:,t,bc_params.reacting_boundaries], dims=2)[:,1], bc_params, t, precomp_P) #currently only reacting on bottom bc
        phip[:,:,t+1] = phip[:,:,t]
        assign_f_phi!(f_phi,phip[:,:,t+1], x_pos[:,t+1], y_pos[:,t+1], psi_mesh, space_cells,t+1)
    end
    verbose && println("end")
    return nothing
end

# end