
module CRBM

function crbm_binary_up(rbm::RBM_t, y::Array{Float64}, x::Array{Float64})
  r = repmat(rbm.c, 1, size(y)[1]) + rbm.V * y' + rbm.W * x'
  convert(Matrix{Float64}, crbm_random_binary_draw(r'))
end



function crbm_binary_down(rbm::RBM_t, z::Array{Float64})
  r = repmat(rbm.b, 1, size(z)[1]) + rbm.W' * z'
  convert(Matrix{Float64}, crbm_random_binary_draw(r'))
end



function crbm_random_binary_draw(p::Matrix{Float64})
  sigm(p) .> rand(size(p))
end



function crbm_binary_update!(rbm::RBM_t, y::Array{Float64}, X::Array{Float64})
  Z = crbm_binary_up(rbm, y, X)
  for i=1:rbm.uditer
    X = crbm_binary_down(rbm, Z)
    Z = crbm_binary_up(rbm, y, X)
  end       
  return X,Z
end



function sigm(p::Matrix{Float64})
  1./(1 .+ exp(-p))
end



function int2binary(v::Int64, n::Int64) # checked
  r=zeros(n)
  for i=1:n
    r[i] = (((1 << (n-i)) & v)>0)?1.0:0.0
  end
  return r
end



function binarise_matrix(A::Matrix{Float64}, bins::Int64)
  N = int(ceil(log2(bins)))
  B=zeros(size(A)[1], size(A)[2]* N)
  for i=1:size(A)[1]
    for j=1:size(A)[2]
      value = A[i,j]
      d     = dvalue(value, bins)
      b     = int2binary(d, N)
      for u = 1:N
        B[i,(j-1)*N+u] = b[u]
      end
    end
  end
  B
end



function crbm_binary_train_plain!(rbm, S, A, bins, perturbation)
  N = ceil(log2(bins))   # nr units pro sensor
  P = perturbation .* randn(size(S))
  ss= binarise_matrix(S + P, bins) # binarisation of training data
  aa= binarise_matrix(A, bins) # binarisation of training data
  for t=1:rbm.numepochs
    # extract data batch for current epoch
    r = rand(rbm.batchsize);
    s = ss[int64(ceil(size(ss)[1] * r)),:] 
    a = aa[int64(ceil(size(aa)[1] * r)),:]

    # generate hiddens given the data
    z = crbmup(rbm,s,a) 

    # generate random outputs to start sampler
    A=zeros(size(s)[1], rbm.n) 
    for i=1:size(s)[1]
      A[i,:] = transpose(int2binary(int(floor(2^rbm.n * rand())), rbm.n )) 
    end

    (A, Z) = crbm_binary_update(rbm, s, A) 

    Eb  = transpose(mean(a,1) - mean(A,1))
    Ec  = transpose(mean(z,1) - mean(Z,1))
    EW  = (z' * a - Z' * A)/size(s)[1]
    EV  = (z' * s - Z' * s)/size(s)[1]

    Eb = squeeze(Eb,2)
    Ec = squeeze(Ec,2)

    if rbm.momentum == 0
      rbm.b = rbm.b + rbm.alpha * Eb  
      rbm.c = rbm.c + rbm.alpha * Ec  
      rbm.W = rbm.W + rbm.alpha * EW  
      rbm.V = rbm.V + rbm.alpha * EV     
    else 
      rbm.b = rbm.b + rbm.alpha * Eb + rbm.momentum * rbm.vb 
      rbm.c = rbm.c + rbm.alpha * Ec + rbm.momentum * rbm.vc 
      rbm.W = rbm.W + rbm.alpha * EW + rbm.momentum * rbm.vW 
      rbm.V = rbm.V + rbm.alpha * EV + rbm.momentum * rbm.vV

      rbm.vb = Eb
      rbm.vc = Ec
      rbm.vW = EW
      rbm.vV = EV
    end

  end # training iteration
end 

end
