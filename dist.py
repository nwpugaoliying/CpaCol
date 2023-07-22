import torch
import pdb
# ma = torch.tensor([0,1,2,3,4,5]).float().unsqueeze(0)
# ma = ma.repeat(11, 6)
# mb = ma
# mc = torch.zeros(11, 6, 11, 6) 
# for i in range(11):
#     for j in range(6):
#         for p in range(11):
#             for q in range(6):
#                 x = (i-p) * (i-p) + (j-q) * (j-q)
#                 if x > 2:
#                     mc[i, j, p, q] = torch.tensor(x).float().sqrt()
#                 # if xx > 2.3

# print(mc)

# pdb.set_trace()
# dist = torch.exp(-mc/5)


ma = torch.ones(22,22)
ma[:5, 5:11] *= 0.9
ma[:5, 11:17] *= 0.8
ma[:5, 17:] *= 0.7

ma[5:11, :5] *= 0.9
ma[5:11, 11:17] *= 0.9
ma[5:11, 17:] *= 0.8

for i in range(11, 17):
    for j in range(22):
        ma[i, j] = ma[6, 21-j]

for i in range(17, 22):
    for j in range(22):
        ma[i, j] = ma[0, 21-j]

print(ma)
torch.save(ma, 'weight_reid.pt')
pdb.set_trace()