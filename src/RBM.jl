module RBM

using PyPlot

include("CRBM.jl")

export RBM_r, rbm_copy
export rbm_create, rbm_create_with_standard_values
export rbm_init_weights_random!, rbm_init_visible_bias!
export rbm_init_output_bias_random!, rbm_init_hidden_bias_random!
export rbm_rescale_weights!
export rbm_calculate_L1, rbm_calculate_L2
export rbm_write, rbm_read
export crbm_binary_update!
export sigm
export rbm_visualise_learning_progress
export crbm_binary_train_plain!
# export crbm_binary_train_L2!             # TODO
# export crbm_binary_train_L1!             # TODO
# export crbm_binary_train_weight_scaling! # TODO



type RBM_t 
  n::Int64            # number of output nodes
  m::Int64            # number of hidden nodes
  k::Int64            # number of input nodes
  uditer::Int64       # number of up-down passes when sampling
  alpha::Float64      # learning rate
  momentum::Float64   # contribution of previous gradient
  weightcost::Float64 # weight cost for L2
  numepochs::Int64    # number of training epochs
  batchsize::Int64    # size of training data batches
  W::Matrix{Float64}  # interaction weights between hidden and outputs
  V::Matrix{Float64}  # interaction weights between hidden and inputs
  b::Vector{Float64}  # bias weights for outputs
  c::Vector{Float64}  # bias weights for hiddens
  vW::Matrix{Float64} # for the momentum
  vV::Matrix{Float64} # for the momentum
  vb::Vector{Float64} # for the momentum
  vc::Vector{Float64} # for the momentum
end



function rbm_create(n::Int64, m::Int64, k::Int64, uditer::Int64,
  alpha::Float64, momentum::Float64, weightcost::Float64,
  numepochs::Int64, batchsize::Int64)

  W = zeros((m, n))
  V = zeros((m, k))
  b = zeros(n)
  c = zeros(m)

  vW = zeros((m, n))
  vV = zeros((m, k))
  vb = zeros(n)
  vc = zeros(m)

  return RBM_t(n, m, k, uditer, alpha,
               momentum, weightcost, numepochs, batchsize,
               W, V, b, c, vW, vV, vb, vc)
end



function rbm_create_with_standard_values(n::Int64, m::Int64, k::Int64)
  rbm_create(n, m, k, 10, 0.1, 0.5, 0.001, 10000, 50)
end



function rbm_copy(src::RBM_t)
  copy = rbm_create(src.n, src.m, src.k, src.uditer, src.alpha, src.momentum, src.numepochs, src.batchsize)
  copy.W  = deepcopy(src.W)
  copy.V  = deepcopy(src.V)
  copy.b  = deepcopy(src.b)
  copy.c  = deepcopy(src.c)
  copy.vW = deepcopy(src.vW)
  copy.vV = deepcopy(src.vV)
  copy.vb = deepcopy(src.vb)
  copy.vc = deepcopy(src.vc)
  return copy
end



function rbm_init_weights_random!(rbm::RBM_t)
  rbm.W = rand((rbm.m,rbm.n)) .* 0.01
  rbm.V = rand((rbm.m,rbm.k)) .* 0.01
end



function rbm_init_hidden_bias_random!(rbm::RBM_t)
  rbm.b = rand(rbm.n) .* 0.01
end



function rbm_init_output_bias_random!(rbm::RBM_t)
  rbm.c = rand(rbm.m) .* 0.01
end



function rbm_init_visible_bias!(rbm::RBM_t, data::Array{Int64,2})
  # each row of the data must contain be of type {0,1}^k
  @assert (size(data)[2] == rbm.n) "rbm_init_visible_bias!: Each row of the data array must be of type {0,1}^k"

  l = size(data)[1]
  for i=1:rbm.n
    s = sum(data[:,i])
    if s == l
      p =  (l-1) / l
      rbm.b[i] = log2(p/(1.0 - p))
    elseif s == 0
      rbm.b[i] = 0.0
    else
      p = s / l
      rbm.b[i] = log2(p/(1.0 - p))
    end
  end
end



function rbm_rescale_weights!(rbm::RBM_t, abs_maximum::Float64)
  @assert (abs_maximum > 0.0) "rbm_rescale_weights!: abs_maximum must be > 0.0"
  mw = maximum(abs(rbm.W))
  mv = maximum(abs(rbm.V))
  if mw > mv
    m = mw
  else
    m = mv
  end
  if m > abs_maximum
    factor = abs_maximum / m
    rbm.W = rbm.W .* factor
    rbm.V = rbm.V .* factor
  end
end



function rbm_calculate_L2(rbm::RBM_t, last::Float64)
  current = (sum(rbm.W^2) + sum(rbm.V^2)) * 0.5
  diff    = rbm.weightcost * (current - last) * rbm.alpha
  return diff
end



function rbm_calculate_L1(rbm::RBM_t, last::Float64)
  current = (sum(abs(rbm.W)) + sum(abs(rbm.V))) * 0.5
  diff    = rbm.weightcost * (current - last) * rbm.alpha
  return diff
end



function rbm_write(filename, rbm)
  f = open(filename, "w")
  serialize(f, rbm)
  close(f)
end



function rbm_read(filename)
  f = open(filename, "r")
  rbm = deserialize(f)
  close(f)
  return rbm
end



function rbm_visualise_learning_progress(rbm::RBM_t)
end

end # module
