module RBM

type RBM_t # checked
  bins::Int64        # number of bins for discretized data
  n::Int64           # number of output nodes
  m::Int64           # number of hidden nodes
  k::Int64           # number of input nodes
  uditer::Int64      # number of up-down passes when sampling
  alpha::Float64     # learning rate
  momentum::Float64  # contribution of previous gradient
  numepochs::Int64   # number of training epochs
  batchsize::Int64   # size of training data batches
  mem::Int64         # time window of inputs (standard ist 1)
  W::Matrix{Float64} # interaction weigths between hidden and outputs
  V::Matrix{Float64} # interaction weights between hidden and inputs
  b::Vector{Float64} # bias weights for outputs
  c::Vector{Float64} # bias weights for hiddens
  vW::Matrix{Float64} # for the momentum
  vV::Matrix{Float64} # for the momentum
  vb::Vector{Float64} # for the momentum
  vc::Vector{Float64} # for the momentum
end


function rbm_create(bins, n, m, k, uditer, alpha, momentum, numepochs, batchsize, mem, randomise)
  if randomise
    W = (2.0 .* rand((m,n)) .- 1.0) .* 0.1 # random numbers in [-0.1, 0.1]
    V = (2.0 .* rand((m,k)) .- 1.0) .* 0.1 # random numbers in [-0.1, 0.1]
    b = (2.0 .* rand(n)     .- 1.0) .* 0.1 # random numbers in [-0.1, 0.1]
    c = (2.0 .* rand(m)     .- 1.0) .* 0.1 # random numbers in [-0.1, 0.1]
  else
    W = zeros((m, n))
    V = zeros((m, k))
    b = zeros(n)
    c = zeros(m)
  end

  vW = zeros((m, n))
  vV = zeros((m, k))
  vb = zeros(n)
  vc = zeros(m)

  return RBM_t(bins, n, m, k, uditer, alpha, momentum, numepochs, batchsize,
               mem, W, V, b, c, vW, vV, vb, vc)
end


end # module
