# \phi_{l_1}\phi^\top_{l_1}\ldots\phi_{l_k} is a symmetric square matrix


In [22]: phi=[np.random.randn(12,1) for li in range(k)]

In [23]: A=np.eye(12);
    ...: for li in range(k):
    ...:     A*=phi[li]@np.transpose(phi[li])
    ...:

In [24]: np.transpose(A)==A

    
    
    
# upper bound comparison
 etas,Ls,z_comb=[],[],[];
     ...: for eta in np.linspace(0.01, 0.99, 99, endpoint=True):
     ...:     for L in np.linspace(2,100, 98, endpoint=True):
     ...:         etas.append(eta); Ls.append(L);
     ...:         z_comb.append((1-eta)**(2*L) - (1-eta)**(2*L-3) + (1-eta)**(L) - (1-eta)**(L-2))
