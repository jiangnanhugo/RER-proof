# this file is used to debug the bionomial equalities in the proof.

from sympy import binomial_coefficients
from sympy import binomial as bi



L=20; k=10;
#
# $\sum_{l=1}^L {L-l \choose k-1}$ 
# should be equal to: 
# ${L\choose k}$ 
#
sumed=0
for l in range(1,L+1):
  sumed+=bi(L-l, k-1)
print(bi(L,k))
# Out[7]: 184756

print(sumed)
# Out[8]: 184756



#
# $\sum_{l=1}^L {L+l-2 \choose k-1 }$ 
# should be equal to: 
# ${2L-1\choose k} - {L-2\choose k}$ 
#
sumed=0
for l in range(1,L+1):
  sumed+=bi(L+l-2, k-1)

print(bi(2*L-1,k)-bi(L-2,k))
# Out[11]: 635701638
print(sumed)
# Out[12]: 635653018
