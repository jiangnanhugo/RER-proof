# \phi_{l_1}\phi^\top_{l_1}\ldots\phi_{l_k} is a symmetric square matrix


In [22]: phi=[np.random.randn(12,1) for li in range(k)]

In [23]: A=np.eye(12);
    ...: for li in range(k):
    ...:     A*=phi[li]@np.transpose(phi[li])
    ...:

In [24]: np.transpose(A)==A
