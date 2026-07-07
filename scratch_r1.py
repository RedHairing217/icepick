import sympy as sp

theta,p = sp.symbols('theta p', positive=True)
# Weinstein-type: minimize J[u] = ||sqrt(L) u||^theta ||u||^{1-theta} / ||(I*|u|^p)|u|^p||^{1/(2p)}
# The claim: sharp constant C = theta^{theta/2} (1-theta)^{1/(2p)-theta/2} ||Q||^{(p-1)/p}
# Check dimensional/homogeneity consistency under scaling u -> c u.
# LHS scales: (c^theta)(c^{1-theta}) = c^1.  RHS: (c^{2p})^{1/(2p)} = c^1. So inequality is scale-invariant in amplitude c. Good, consistent - C is a pure attained constant.
# The exponent (p-1)/p on ||Q||_{L2}: at optimizer u=Q, Pohozaev/Nehari relates the three quantities.
# Standard Weinstein best constant has form involving powers of the Lagrange multipliers = theta and (1-theta) balance.
# The factor theta^{theta/2}(1-theta)^{(1/(2p)-theta/2)} is exactly the min over lambda of the AM-GM two-term balance:
lam = sp.symbols('lambda', positive=True)
# minimize over scaling of the ratio a^theta b^{1-theta} type -> gives theta^{theta}(1-theta)^{1-theta} style factors.
# We just sanity-check the two exponents on theta and (1-theta) are 'balanced':
e1 = theta/2
e2 = sp.Rational(1,1)/(2*p) - theta/2
print("exp on theta:", e1)
print("exp on (1-theta):", e2)
print("sum:", sp.simplify(e1+e2))  # = 1/(2p)
