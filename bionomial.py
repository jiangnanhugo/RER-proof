# this file is used to debug the bionomial equalities in the proof.

from sympy import binomial_coefficients
from sympy import binomial as bi



L=20; k=10;

sumed=0
for l in range(1,L+1):
  sumed+=bi(L-l, k-1)
print(bi(L,k))
# Out[7]: 184756

print(sumed)
# Out[8]: 184756

