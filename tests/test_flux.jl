using Flux
using Flux.Optimise
using Flux.Data
using Test

model = Chain(
  Dense(10, 5, relu),
  Dense(5, 1, relu))

loss(x, y) = Flux.mse(model(x), y)

X = rand(10,100)
y = rand(1, 100)
# y = θ'*X


loss(X, y)


gs = gradient(()->loss(X,y), params(model))
for p in params(model)
  println(sum(gs[p]))
  p .-= 0.01*gs[p]
end
loss(X, y)



model = Chain(
  Dense(10, 5, relu),
  Dense(5, 1, relu))

loss(x, y) = Flux.mse(model(x), y)

X = rand(10,100) # Dummy data
θ = rand(10)
y = θ'*X
loss(X, y) # ~ 3


gs = gradient(() -> loss(X, y), params(model))
for p in params(model)
  println(sum(gs[p]))
  p .-= 0.01 * gs[p]
end
loss(X,y)

opt = ADAM(0.01, (0.9, 0.999))
d = DataLoader(X, y, batchsize = 10)
train!(loss, params(model), d, opt)
train!(loss, params(model), [(X, y)], opt)


loss(X,y)

