"""Generate figures/sample_points.tex: the sample points of the drift fit
(density_filter.tex, Section 5.3) in 2D for rank r = 3.

Window [-1, 1]^2, 4 spectral elements of order 5 per direction (Dirichlet:
19 stored nodes per direction). For the update of direction d the samples
are K = r^2 = 9 fibres: the anchors are the 9 nodes of the *other*
coordinate closest to the mode, and each fibre runs over all nodes of
direction d. Run:  python3 make_sample_points.py > sample_points.tex
"""
import numpy as np

L, n_el, order = 1.0, 4, 5
r = 3
K = r * r
# Gauss–Lobatto nodes on [-1, 1] for order 5 (roots of P_5' and the endpoints).
ref = np.array([-1.0, -0.7650553239294647, -0.2852315164806451,
                0.2852315164806451, 0.7650553239294647, 1.0])
h = 2 * L / n_el
nodes = sorted({round(-L + (e + 0.5 * (t + 1)) * h, 12)
                for e in range(n_el) for t in ref})
interior = [x for x in nodes if abs(abs(x) - L) > 1e-9]  # Dirichlet: ends unstored
bounds = [-L + e * h for e in range(n_el + 1)]
# Anchors: the K nodes of the other coordinate nearest to the mode (xi = 0).
anchors = sorted(sorted(interior, key=abs)[:K])

def panel(d, x0, label):
    out = []
    s = 2.6  # cm per unit
    out.append(r"\begin{scope}[shift={(%.2f,0)}]" % x0)
    # element boundaries
    for b in bounds:
        out.append(r"\draw[gray!45,thin] (%.4f,%.4f) -- (%.4f,%.4f);" % (b*s, -L*s, b*s, L*s))
        out.append(r"\draw[gray!45,thin] (%.4f,%.4f) -- (%.4f,%.4f);" % (-L*s, b*s, L*s, b*s))
    out.append(r"\draw[black,thick] (%.4f,%.4f) rectangle (%.4f,%.4f);" % (-L*s, -L*s, L*s, L*s))
    # density: tilted Gaussian level sets (drawn as rotated ellipses)
    for k, lev in enumerate([0.9, 0.6, 0.3]):
        a = 0.62 * s * (k + 1) / 3 * 1.35
        b = 0.30 * s * (k + 1) / 3 * 1.35
        out.append(r"\draw[red!70!black,thin,rotate=35] (0,0) ellipse (%.3f and %.3f);" % (a, b))
    # all grid nodes, faint
    for x in interior:
        for y in interior:
            out.append(r"\fill[gray!60] (%.4f,%.4f) circle (0.35pt);" % (x*s, y*s))
    # fibres
    for m, a in enumerate(anchors):
        if d == 1:
            out.append(r"\draw[blue!70!black,line width=0.4pt] (%.4f,%.4f) -- (%.4f,%.4f);" % (-L*s, a*s, L*s, a*s))
            for x in interior:
                out.append(r"\fill[blue!70!black] (%.4f,%.4f) circle (0.9pt);" % (x*s, a*s))
            out.append(r"\draw[blue!70!black,thick] (%.4f,%.4f) -- (%.4f,%.4f);" % (-L*s-0.12, a*s, -L*s-0.04, a*s))
        else:
            out.append(r"\draw[green!45!black,line width=0.4pt] (%.4f,%.4f) -- (%.4f,%.4f);" % (a*s, -L*s, a*s, L*s))
            for y in interior:
                out.append(r"\fill[green!45!black] (%.4f,%.4f) circle (0.9pt);" % (a*s, y*s))
            out.append(r"\draw[green!45!black,thick] (%.4f,%.4f) -- (%.4f,%.4f);" % (a*s, -L*s-0.12, a*s, -L*s-0.04))
    out.append(r"\node[below] at (%.3f,%.3f) {$\xi_1$};" % (0, -L*s-0.18))
    out.append(r"\node[left] at (%.3f,%.3f) {$\xi_2$};" % (-L*s-0.18, 0))
    out.append(r"\node[above] at (0,%.3f) {%s};" % (L*s+0.05, label))
    out.append(r"\end{scope}")
    return "\n".join(out)

print(r"\begin{tikzpicture}")
print(panel(1, 0.0, r"(a) update of $G_1^+$"))
print(panel(2, 6.8, r"(b) update of $G_2^+$"))
print(r"\end{tikzpicture}")
