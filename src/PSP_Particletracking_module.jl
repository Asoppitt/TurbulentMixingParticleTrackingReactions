# module PSPParticleTrackingReactions

using Random, Distributions#, Plots
import StatsBase as sb
import LinearAlgebra as la
import Statistics as st
import Base.getindex, Base.setindex!, Base.size, Base.length, Base.iterate, Base.ndims, Base.copyto!

export cell_grid,BC_params,psi_grid,motion_params,PSP_motion_bc_params

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
abstract type Omega{AbstractFloat,Distribution,Logical} end

struct OmegaGl{T}<:Omega{T,Gamma, false}
    omega::Vector{T}
    log_omega_normed::Vector{T}
    dist::Gamma
    omega_bar::T
    omega_sigma_2::T
    omega_min::T
    T_omega::T
    inv_T_omega::T
end

struct OmegaG{T}<:Omega{T,Gamma,true}
    omega::Vector{T}
    dist::Gamma
    omega_bar::T
    omega_sigma_2::T
    omega_min::T
    T_omega::T
    inv_T_omega::T
end

struct OmegaL{T}<:Omega{T,LogNormal,false}
    omega::Vector{T}
    log_omega_normed::Vector{T}
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
    p_params.omega_min>0 && return OmegaG(omega,dist,p_params.omega_bar,p_params.omega_sigma_2,p_params.omega_min,p_params.T_omega,1/p_params.T_omega)
    p_params.omega_min<=0 &&return OmegaGl(omega,log.(omega),dist,p_params.omega_bar,p_params.omega_sigma_2,p_params.omega_min,p_params.T_omega,1/p_params.T_omega)
end

function Omega(dist::D,np::I,p_params::PSPParams{T}) where I<:Integer where D<:LogNormal where T<:AbstractFloat
    omega=T.(rand(dist,np))
    return OmegaL(omega,log.(omega/p_params.omega_bar),dist,p_params.omega_bar,p_params.omega_sigma_2,p_params.omega_min,p_params.T_omega,log(p_params.omega_bar),varlogx(dist),1/p_params.T_omega)
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