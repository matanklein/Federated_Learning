from collaborative.participant import *
import sparsechem as sc

mc = sc.ModelConfig(10, [40], 2, 1)
mc.lr = 0.01
trunk = sc.Trunk(mc)
model = sc.TrunkAndHead(trunk, mc)

p = Server(model, mc, X=[[1,0], [0,1], [1,1]], y=[1,1,0])

print(p.get_weights('trunk'))