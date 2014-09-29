module RBM

#using PyPlot

export RBM_t
export rbm_copy
export rbm_create
export rbm_init_weights_random!
export rbm_init_visible_bias!
export rbm_init_output_bias_random!, rbm_init_hidden_bias_random!
export rbm_rescale_weights!
#= export rbm_calculate_L1, rbm_calculate_L2 =#
export rbm_write, rbm_read, rbm_read_old
export sigm
export rbm_visualise

type RBM_t 
  bins::Int64
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
  dropout::Float64
  perturbation::Float64
end

function rbm_create(n::Int64, m::Int64, k::Int64, uditer::Int64,
  alpha::Float64, momentum::Float64, weightcost::Float64,
  numepochs::Int64, batchsize::Int64, bins::Int64, dropout::Float64,
  perturbation::Float64)

  W = zeros((m, n))
  V = zeros((m, k))
  b = zeros(n)
  c = zeros(m)

  vW = zeros((m, n))
  vV = zeros((m, k))
  vb = zeros(n)
  vc = zeros(m)

  return RBM_t(bins, n, m, k, uditer, alpha,
               momentum, weightcost, numepochs, batchsize,
               W, V, b, c, vW, vV, vb, vc, dropout, perturbation)
end

rbm_create(n::Int64, m::Int64, k::Int64) = rbm_create(n, m, k, 15, 0.01, 0.5, 0.001, 10000, 100, 16, 0.0, 0.1)






function rbm_copy(src::RBM_t)
  copy = rbm_create(src.n, src.m, src.k, src.uditer, src.alpha, src.momentum, src.weightcost, src.numepochs, src.batchsize, src.bins, src.dropout)
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
  rbm.W = randn((rbm.m,rbm.n)) .* 0.01
  rbm.V = randn((rbm.m,rbm.k)) .* 0.01
end



function rbm_init_hidden_bias_random!(rbm::RBM_t)
  rbm.b = rand(rbm.n) .* 0.01
end



function rbm_init_output_bias_random!(rbm::RBM_t)
  rbm.c = rand(rbm.m) .* 0.01
end



function rbm_init_visible_bias!(rbm::RBM_t, data::Array{Int64,2})
  # each row of the data must contain be of type {0,1}^k
  n = size(data)[2]
  N = rbm.n
  @assert (size(data)[2] == rbm.n) "rbm_init_visible_bias!: Each row of the data array must be of type {0,1}^n. It has $n and not $N"

  l = size(data)[1]
  for i=1:rbm.n
    s = sum(data[:,i])
    if s == l # only 1 in data
      p =  (l-1) / l
    elseif s == 0 # no 1 in data
      p =  1 / l
    else
      p = s / l
    end
    rbm.b[i] = log2(p/(1.0 - p))
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



#= function rbm_calculate_L2(rbm::RBM_t, last::Float64) =#
  #= current = sum(rbm.W^2) + sum(rbm.V^2) =#
  #= diff    = rbm.weightcost * (current - last) * rbm.alpha =#
  #= return diff =#
#= end =#



#= function rbm_calculate_L1(rbm::RBM_t, last::Float64) =#
  #= current = sum(abs(rbm.W)) + sum(abs(rbm.V)) =#
  #diff    = rbm.weightcost * (current - last) * rbm.alpha
  #= return diff =#
#= end =#



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

function rbm_read_old(filename)
  f        = open(filename, "r")
  s        = split(strip(readline(f)))

  bins     = int(eval(parse(s[1])))
  n        = int(eval(parse(s[2])))
  m        = int(eval(parse(s[3])))
  k        = int(eval(parse(s[4])))
  uditer   = int(eval(parse(s[5])))
  alpha    = float64(eval(parse(s[6])))
  momentum = float64(eval(parse(s[7])))
  epochs   = int(eval(parse(s[8])))
  batch    = int(eval(parse(s[9])))
  memory   = int(eval(parse(s[10])))

  println("n: ", n, " bins: ", bins, " m: ", m, " k: ", k, " uditer: ", uditer)
  rbm = rbm_create(n, m, k, uditer, alpha, momentum, 0.0, epochs, batch, bins, 0.0, 0.0)

  # reading W
  for i=1:m
    l = split(strip(readline(f)))
    for j=1:n
      rbm.W[i,j] = eval(parse(l[j]))
    end
  end
  # reading V
  for i=1:m
    l = split(strip(readline(f)))
    for j=1:k
      rbm.V[i,j]=eval(parse(l[j]))
    end
  end
  # reading b
  bs = split(strip(readline(f)))
  for i=1:length(bs)
    rbm.b[i] = eval(parse(bs[i]))
  end
  cs = split(strip(readline(f)))
  for i=1:length(cs)
    rbm.c[i] = eval(parse(cs[i]))
  end
  close(f)
  return rbm
end



function rbm_visualise(rbm::RBM_t)

end

end # module
