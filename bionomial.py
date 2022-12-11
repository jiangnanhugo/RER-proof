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
# ${2L-1\choose k} - {L-1\choose k}$ 
#
sumed=0
for l in range(1,L+1):
  sumed+=bi(L+l-2, k-1)

print(bi(2*L-1,k)-bi(L-1,k))
# Out[11]: 635653018
print(sumed)
# Out[12]: 635653018

#
# $\sum_{l=1}^L {2*l-2 \choose k-1 }$ 
# should be less than: 
# ${2L-1\choose k} - {2L-3\choose k-2}$ 
#


In [28]: sumed=0
    ...: for l in range(1,L+1):
    ...:     sumed+=bi(2*l-2, k-2)
    ...:

In [29]: bi(2*L-1,k-1)-bi(2*L-3,k-2)
Out[29]: 173307112

In [30]: sumed
Out[30]: 119603024
  
  
# the following part of code evaluates this equation:
# (1-\eta)^{2L}=\sum_{k=0}^{2L}{2L\choose k} (-\eta)^k=1+\sum_{k=1}^{2L}{2L\choose k} (-\eta)^k,
In [39]: from sympy import binomial as bi

In [40]: bi(10,11)
Out[40]: 0

In [41]: sumed=0;

In [42]: for k in range(2*L+1):
    ...:     sumed+=bi(2*L, k)*(-eta)**k
    ...:

In [44]: sumed
Out[44]: 0.000976562500000000

In [45]: sumed=[];
    ...: for k in range(2*L+1):
    ...:     sumed.append(bi(2*L, k)*(-eta)**k)
    ...:

In [46]: sumed
Out[46]:
[1.00000000000000,
 -5.00000000000000,
 11.2500000000000,
 -15.0000000000000,
 13.1250000000000,
 -7.87500000000000,
 3.28125000000000,
 -0.937500000000000,
 0.175781250000000,
 -0.0195312500000000,
 0.000976562500000000]

In [47]: (1-eta)**(2*L)
Out[47]: 0.0009765625

  
# comparison between our lemma and the existing bounds
# our bound is way better.
In [53]: eta=0.05;L=4;

In [54]: -eta*L
Out[54]: -0.2

In [55]: (1-eta)**(2*L) - (1-eta)**(2*L-3) + (1-eta)**(L) - (1-eta)**(L-2)
Out[55]: -0.1983542562109376

In [56]: eta=0.05;L=40;

In [57]: -eta*L
Out[57]: -2.0

In [58]: (1-eta)**(2*L) - (1-eta)**(2*L-3) + (1-eta)**(L) - (1-eta)**(L-2)
Out[58]: -0.016630930192162413
