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
