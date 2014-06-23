using RBM
using Base.Test

# write your own tests here

# test initialisation
rbm = rbm_create(11, 12, 13, 14, 0.1, 0.2, 100, 15)
@test rbm.n         == 11
@test rbm.m         == 12
@test rbm.k         == 13
@test rbm.uditer    == 14
@test rbm.alpha     == 0.1
@test rbm.momentum  == 0.2
@test rbm.numepochs == 100
@test rbm.batchsize == 15

# test standard initialisation
rbm = rbm_create_with_standard_values(1, 2, 3)
@test rbm.n         == 1
@test rbm.m         == 2
@test rbm.k         == 3
@test rbm.uditer    == 10
@test rbm.alpha     == 0.1
@test rbm.momentum  == 0.5
@test rbm.numepochs == 10000
@test rbm.batchsize == 50

# test random initialisation functions
rbm = rbm_create_with_standard_values(10, 10, 10)
@test rbm.W == zeros(rbm.m, rbm.n)
@test rbm.V == zeros(rbm.m, rbm.k)
@test rbm.b == zeros(rbm.n)
@test rbm.c == zeros(rbm.m)

rbm_init_random_weights!(rbm)

@test rbm.W != zeros(rbm.m, rbm.n)
@test rbm.V != zeros(rbm.m, rbm.k)
@test size(rbm.W) == (rbm.m,rbm.n)
@test size(rbm.V) == (rbm.m,rbm.k)
@test rbm.b == zeros(rbm.n)
@test rbm.c == zeros(rbm.m)

rbm_init_output_bias_random!(rbm)
@test rbm.c != zeros(rbm.m)
@test size(rbm.c) == (rbm.m,)

rbm_init_hidden_bias_random!(rbm)
@test rbm.b != zeros(rbm.n)
@test size(rbm.b) == (rbm.n,)

#function rbm_init_hidden_bias_random!(rbm::RBM_t)

# test visible bias initialisation
r = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] [1, 1, 1, 1, 1, 1, 1, 0, 0, 0] [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] [1, 1, 1, 1, 1, 1, 1, 1, 1, 0] [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

rbm = rbm_create_with_standard_values(11, 1, 1)
rbm_init_visible_bias!(rbm, r)

for i=1:11
  if i == 1
    s = 0.0
  elseif i == 11
    p = (10 - 1.0) / 10.0
    s = log2( p / ( 1 - p))
  else
    p = (i - 1.0) / 10.0
    s = log2( p / ( 1 - p))
  end
  @test rbm.b[i] == s
end

# test rbm copying
src   = rbm_create_with_standard_values(10, 10, 10)
src.W = rand(10,10)
src.V = rand(10,10)
src.b = rand(10)
src.c = rand(10)
dest  = rbm_copy(src)

@test dest.n  == src.n
@test dest.m  == src.m
@test dest.k  == src.k
@test dest.W  == src.W
@test dest.V  == src.V
@test dest.b  == src.b
@test dest.c  == src.c
@test dest.vW == src.vW
@test dest.vV == src.vV
@test dest.vb == src.vb
@test dest.vc == src.vc

dest.W[1,1] = dest.W[1,1] * 2
@test dest.W != src.W

# test weight scaling
rbm = rbm_create_with_standard_values(10, 10, 10)
rbm.W = rand(10,10) * 100
rbm.V = rand(10,10) * 100
cpy = rbm_copy(rbm)
rbm_rescale_weights!(rbm, 10.0)
mw = maximum(abs(rbm.W))
mv = maximum(abs(rbm.V))
if mw > mv
  m = mw
else
  m = mv
end
@test m == 10.0

mw = maximum(abs(cpy.W))
mv = maximum(abs(cpy.V))
if mw > mv
  mx = mw
else
  mx = mv
end

factor = m / mx

for i=1:size(rbm.W)[1]
  for j=1:size(rbm.W)[2]
    rbm.W[i,j]  = cpy.W[i,j] * factor
    rbm.W[i,j] != cpy.W[i,j]
  end
end

for i=1:size(rbm.V)[1]
  for j=1:size(rbm.V)[2]
    rbm.V[i,j]  = cpy.V[i,j] * factor
    rbm.V[i,j] != cpy.V[i,j]
  end
end

# test of assertion
#= rbm = rbm_create(11, 12, 13, 14, 0.1, 0.2, 100, 15) =#
