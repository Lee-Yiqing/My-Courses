---
id: B11
title: "A Survey of Deep Learning for Code Generation"
domain: B
year: 2024
arxiv_id: "2310.09043"
confidence: verified
source: "arXiv:2310.09043 / ACM Computing Surveys"
node_type: paper
---

# A Survey of Deep Learning for Code Generation

**Domain**: [[domain_B|LLM Code Generation]] | **Year**: 2024 | **Confidence**: [x] verified


## Authors
[[author_Jie Huang|Jie Huang]], [[author_Shaojun Jin|Shaojun Jin]], [[author_Hanqi Yan|Hanqi Yan]], [[author_Feng Zhao|Feng Zhao]], [[author_Qi Zhu|Qi Zhu]], [[author_Yanjie Jiang|Yanjie Jiang]], [[author_Yutian Tang|Yutian Tang]], et al.


## Keywords
- [[kw_survey|survey]]
- [[kw_deep learning|deep learning]]
- [[kw_code generation|code generation]]
- [[kw_program synthesis|program synthesis]]
- [[kw_taxonomy|taxonomy]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

Midpoint geometric integrators for inertial magnetization dynamics

M. d’Aquinoa,∗, S. Pernaa, C. Serpicoa

aDepartment of Electrical Engineering and Information Technology, University of Naples Federico II, Via Claudio
21, I-80125, Naples, Italy

Abstract

We consider the numerical solution of the inertial version of Landau-Lifshitz-Gilbert equation (iLLG), which
describes high-frequency nutation on top of magnetization precession due to angular momentum relaxation.
The iLLG equation defines a higher-order nonlinear dynamical system with very different nature compared
to the classical LLG equation, requiring twice as many degrees of freedom for space-time discretization.
It exhibits essential conservation properties, namely magnetization amplitude preservation, magnetization
projection conservation, and a balance equation for generalized free energy, leading to a Lyapunov structure
(i.e. the free energy is a decreasing function of time) when the external magnetic field is constant in time.
We propose two second-order numerical schemes for integrating the iLLG dynamics over time, both based on
implicit midpoint rule. The first scheme unconditionally preserves all the conservation properties, making
it the preferred choice for simulating inertial magnetization dynamics. However, it implies doubling the
number of unknowns, necessitating significant changes in numerical micromagnetic codes and increasing
computational costs especially for spatially inhomogeneous dynamics simulations. To address this issue, we
present a second time-stepping method that retains the same computational cost as the implicit midpoint
rule for classical LLG dynamics while unconditionally preserving magnetization amplitude and projection.
Special quasi-Newton techniques are developed for solving the nonlinear system of equations required at
each time step due to the implicit nature of both time-steppings. The numerical schemes are validated on
analytical solution for macrospin terahertz frequency response and the effectiveness of the second scheme is
demonstrated with full micromagnetic simulation of inertial spin waves propagation in a magnetic thin-film.

Keywords:
magnetic inertia, terahertz spin nutation, micromagnetic simulations, inertial
Landau-Lifshitz-Gilbert (iLLG) equation, implicit midpoint rule, numerical methods.

1. Introduction

arXiv:2310.09043v1  [physics.comp-ph]  13 Oct 2023

The study of ultra-fast magnetization processes has become increasingly important in recent years,
particularly for its potential applications to future generations of nanomagnetic and spintronic devices [1].
Since the pioneering experiment by Beaurepaire et al. [2] that revealed subpicosecond spin dynamics, the
investigation of ultra-fast magnetization processes has attracted the attention of many research groups,
leading to a considerable body of research [3, 4, 5, 6, 7, 8, 9, 10].
Recently, there have been exciting experimental developments in the direct detection of spin nutation in
ferromagnets in the terahertz range [11, 12]. This has confirmed the presence of inertial effects in magnetization dynamics, which were theoretically predicted several years ago [13, 14, 15]. Nutation-like magnetization motions in nanomagnets occurring at gigahertz frequencies under the action of time-harmonic applied
external magnetic fields were also studied theoretically in past decades within the classical precessional
dynamics[16].

∗Corresponding author
Email addresses: mdaquino@unina.it (M. d’Aquino), salvatore.perna@unina.it (S. Perna), serpico@unina.it (C.
Serpico)

From a technological perspective, the observation of terahertz spin nutation opens up new possibilities
for exploiting novel ultra-fast regimes. For instance, it may be possible to use strong picosecond field pulses
to drive ballistic magnetization switching into the inertial regime [17, 18, 19, 20, 21, 22]. This has important
implications for the development of ultra-fast magnetic devices, and it also has fundamental implications
for the physics of magnetism.
From a theoretical point of view, inertial magnetization dynamics can be described by augmenting the
classical Landau-Lifshitz-Gilbert (LLG) precessional dynamics with a torque term modeling intrinsic angular
momentum relaxation [13, 14]. This approach has been successful in explaining the observed high-frequency
spin nutation in uniformly-magnetized ferromagnetic samples [11], for which magnetization dynamics is
governed by the following inertial version of the Landau-Lifshitz-Gilbert equation[13, 14]:

dM

dM


,
(1)

dt −τ 2 d2M

dt2

dt
= −γM ×

Heff −
α
γMs

where M(t) is the magnetization vector field (Ms is the saturation magnetization of the material), Heff is
the magnetic effective field, α is the Gilbert damping, γ is the absolute value of the gyromagnetic ratio and
τ defines the time scale of inertial magnetic phenomena.
However, when spatial changes of magnetization do occur in magnetic systems of nano- and micro-scale,
the description of spatially-inhomogeneous ultra-fast magnetization dynamics occurring at sub-picosecond
time scales becomes a challenging problem that requires appropriate extension of eq.(1) to take into account
space-varying vector fields in the region Ωoccupied by the ferromagnetic body. This extension leads to the
formulation of a novel equation where formally the total derivatives with respect to time become partial
and the effective field is given by the variational derivative of the Gibbs-Landau free energy functional[23],
resulting in the following:

∂M

∂M


,
(2)

∂t −τ 2 ∂2M

∂t2

∂t
= −γM ×

Heff −
α
γMs

where generally the natural (homogeneous Neumann) boundary conditions ∂M/∂n = 0 are inherited by
the classical LLG when no surface anisotropy is present at the body surface ∂Ω. Equation (2) reduces to
the purely precessional classical LLG equation when no inertia is considered (i.e. τ = 0). Nevertheless,
despite this apparent similarity, eq.(2) has profoundly different nature in that it has hyperbolic (wavelike) character (instead of parabolic as the classical LLG equation) and admits the possibility of travelling
solutions (spin waves) with finite propagation speed[24]. For this reason, the iLLG dynamics deserves a
dedicated investigation in his own rights.
In this respect, based on equations (1),(2), a number of theoretical studies have been proposed in the latest
years to characterize terahertz spin nutation[25, 26, 27, 28, 29, 30, 31, 32]. Most of these interesting studies
rely on analytical approaches valid in idealized situations such as, for instance, analysis of magnetization
oscillations in single-domain particles (macrospin) or small-amplitude spin waves propagation in infinite
media. Very recently, the possibility to observe propagation of ultra-short inertial spin waves in confined
ferromagnetic thin-films driven by ac terahertz fields has been also theoretically demonstrated[24]. These
waves exhibit behavior that deviates significantly from classical exchange spin-waves and can propagate at
a finite speed up to a limit of several thousands meters per second, which is comparable with the velocity
of surface acoustic waves.
While such phenomena occurring in confined micromagnetic systems mainly involve magnetization oscillations around equilibria and can be investigated by analyzing the inertial LLG dynamics in the linear
regime, no such possibility exists when far from equilibrium dynamics such as nonlinear oscillations[33],
magnetization switching[21] or even chaos[34] are considered.
In these situations where no analytical techniques can be applied, one has to resort to numerical simulation. In this respect, after the experimental evidence of the terahertz spin nutation[11], the study of
inertial effects in magnetization dynamics is rapidly becoming an emergent field of research and, consequently, the need for accurate and efficient computational techniques that exploit the intrinsic properties of
the nutation dynamics beyond off-the-shelf time-stepping schemes is growing fast, too. Nonetheless, at the

2

present moment, very few works[35, 36] address ad-hoc numerical techniques for the inertial magnetization
dynamics.
In this paper, after illustrating the general qualitative conservation properties of the continuous inertial
magnetization dynamics, we propose suitable time-integration schemes based on the implicit midpoint rule
technique[37] for the numerical solution of the inertial LLG (iLLG) equation and their relevant properties
are discussed. The midpoint rule is an unconditionally stable second order accurate scheme which preserves
the fundamental geometrical properties of the classical LLG dynamics[38]. The first time-stepping proposed
here is shown to preserve all relevant conservation properties of the iLLG dynamics unconditionally, i.e.
regardless of the time step amplitude. Despite these remarkable properties, we show that, in general, the
numerical integration of inertial magnetization dynamics must address the issue of the higher order of the
dynamical system that it describes, which implies dramatic changes of micromagnetic codes and results
anyway in at least doubling the computational cost of the numerical scheme as compared to classical LLG
dynamics.
This has a huge impact when micromagnetic simulations with full spatial discretization on
hundred thousands (or more) computational cells have to be performed, such as in the case of (sub)micronsized magnetic systems. For this reason, we develop an additional efficient implementation of the midpoint
rule technique for iLLG dynamics, based on suitable multistep method for the inertial term, which can be
built on the top of that associated with classical LLG dynamics and, therefore, retaining a computational
cost with the same order of magnitude.
The proposed techniques are first validated by computing the
frequency response of a magnetic thin-film modeled as single spin (macrospin) magnetized along the easy
direction and subject to out-of-plane ac field, and comparing the results with the analytical solution. Then,
full micromagnetic simulations of inertial spin wave propagation in a ferromagnetic nanodot are performed
in order to demonstrate the accuracy and effectiveness of the second proposed time-stepping in reproducing
spatially-inhomogeneous ultra-fast spin nutation dynamics.

2. Inertial magnetization dynamics and qualitative properties

The starting point of the discussion is the inertial Landau-Lifshitz-Gilbert (iLLG) equation (2), expressed
in dimensionless form[13, 24]:

∂m


,
(3)

∂t = −m ×

heff −α∂m

∂t −ξ ∂2m

∂t2

where m(r, t) is the magnetization unit-vector (normalized by the saturation magnetization Ms) at each
location r ∈Ω(Ωis the region occupied by the magnetic body), time is measured in units of (γMs)−1

(corresponding to 5.7 ps for γ = 2.21 × 105A−1s−1m and µ0Ms = 1T), α is the Gilbert (dimensionless and
positive, typically in the order ∼10−3 ÷ 10−2) damping parameter, the parameter ξ measures the strength
of inertial effects in magnetization dynamics. It is worthwhile noting (see eq.(2)) that the dimensionless
quantity ξ can be expressed as ξ = (γMsτ)2 where τ determines the physical time-scale of magnetic inertia,
for which previous works[13, 11, 21] assessed its order of magnitude as fractions of picosecond (this implies
that typically ξ ∼10−2). Thus, the inertial effects in magnetization dynamics are governed by a quantity
with the same smallness as usual Gilbert damping α ∼10−2. The effective field heff(r, t) is given by[23]:

heff = −δg

δm ,
(4)

which takes into account interactions (exchange, anisotropy, magnetostatics, Zeeman) among magnetic
moments and is expressed as the variational derivative of the free energy functional (the dimensionless
energy is measured in units of µ0M 2
s V , with V being the volume of region Ω)

Z

g(m, ha) = 1

V

l2
ex
2 (∇m)2 + fan −1

2hm · m −ha · m dV ,
(5)

Ω

where A and lex =
p

(2A)/(µ0M 2s ) are the exchange stiffness constant and length, respectively, fan is the
anisotropy energy density, hm is the magnetostatic (demagnetizing) field and ha(r, t) the external applied
field.

3

When the anisotropy is of uniaxial type, such that fan = κan[1 −(m · ean)2] with κan and ean being the
uniaxial anisotropy constant and unit-vector, respectively, the effective field can be expressed by the sum of
a linear operator C acting on magnetization vector field plus the applied field:

heff(r, t) = −Cm + ha ,
(6)

where C = −l2
ex∇2 + N + κanean ⊗ean and N is the (symmetric-positive definite) demagnetizing operator
such that:

hm(r) = 1

4π ∇∇·
Z

m(r′)
|r −r′| dV = −Nm .
(7)

Ω

As mentioned in the previous section, eq. (3) is usually complemented with the natural boundary conditions
∂m/∂n = 0 at the body surface ∂Ω, which is typical when no surface anisotropy is considered. It can be
shown that the operator C with the aforementioned boundary conditions is self-adjoint and positive-definite
in the appropriate subspace of square-integrable vector fields[23].
It is also worth remarking that, for eq.(3), equilibrium magnetization fields are characterized by simultaneously vanishing time-derivatives of first and second order:

∂m

∂t = 0
,
∂2m

∂t2 = 0
.
(8)

Equation (3) describes a nonlinear dynamical system of higher order compared to that associated with
the classical LLG equation (obtained by setting ξ = 0 in eq.(3)). In fact, by defining a new variable w
resembling, in a purely formal fashion, the ’angular momentum’ of a point-particle of unitary mass, position
vector m and velocity ∂m/∂t such that

w = m × ∂m

∂t ,
(9)

one has:
∂w

∂t = m × ∂2m

∂t2
.
(10)

First, by dot-multiplying both sides of eq.(3) by m, we observe that magnetization vector evolves on the
unit-sphere |m|2 = 1 since

m · ∂m

∂t = 0 .
(11)

Then, by cross-multiplying both sides of eq.(3) by m, one obtains:

w = −m × (m × heff) + αm × w + ξm × ∂w

∂t
.
(12)

By performing further cross-multiplication of both sides of the latter equation by m, one ends up with:

m × w = (m × heff) −αw −ξ dw

dt
,
(13)

where the property m · ∂w/∂t = 0 has been used.
Consequently, iLLG eq.(3) can be rewritten as a set two coupled nonlinear equations for variables m
and w as follows:

∂m

∂t = w × m
,
(14)

ξ ∂w

∂t = −m × w −αw + m × heff
,
(15)

where eq.(14) comes from eq.(9) cross-multiplied by m combined with the fact that |m|2 = 1, and eq.(15)
from eq.(13). We point out that, as a consequence of eq.(8) and the definition of w from eq.(9), equilibrium
solutions of eqs.(14)-(15) are such that:

∂m

∂t = 0
,
∂w

∂t = 0
.
(16)

4

In this way, the implicit equation (3) has been transformed into a higher-order equation in standard explicit
form, which is amenable of general considerations concerning the properties of the dynamical systems that
it describes.
To this end, we now focus on the dynamical system expressed by eqs.(14)-(15) where the state variables
m, w are considered independent of each other, remembering that it is equivalent to the original iLLG eq.(3)
when eq.(9) holds. First of all, by dot-multiplying eq.(14) by m, one can immediately see that the motion
of vector m occurs on the unit-sphere |m| = 1:

m · ∂m

∂t = 0
⇒
|m(r, t)| = 1
∀r ∈Ω, t ≥t0,
(17)

provided that m has unit-amplitude at initial time t0.
The latter will be referred to as magnetization
amplitude conservation property.
Now, let us sum eq.(14) dot-multiplied by w and eq.(15) divided by ξ and dot-multiplied by m. One
has:

w · ∂m

∂t + m · ∂w

∂t = ∂(w · m)

∂t
= −α

ξ w · m
.
(18)

This means that, in any spatial location r ∈Ω, the scalar product w · m, termed as ’angular momentum’
projection on magnetization, will have to decay exponentially to zero as follows:

ξ t
∀r ∈Ω, t ≥t0,
(19)

w(r, t) · m(r, t) = w(r, t0) · m(r, t0)e−α

where the time decay constant is controlled by the ratio ξ/α > 0 between the intensities of damping and
inertia. Thus, for t ≫t0 (practically t > t0 + 5ξ/α), the ’angular momentum’ variable w is asymptotically
constrained to evolve on the manifold defined by w · m = 0. Interestingly, for zero damping α = 0, the
latter equation implies exact conservation of the product w · m at any time:

w(r, t) · m(r, t) = w(r, t0) · m(r, t0)
∀r ∈Ω, t ≥t0.
(20)

From equation (19) it is also worth noting that, for any value of α ≥0 and initially vanishing magnetization time-derivative ∂m/∂t(r, t0) = 0 at any location r ∈Ω, which therefore implies w(r, t0) = 0, the
iLLG dynamics will occur such that the product w · m is always zero:

w(r, t) · m(r, t) = w(r, t0) · m(r, t0) = 0
∀r ∈Ω, t ≥t0.
(21)

From the above discussion, being that the inertial magnetization dynamics must fulfill the two constraints
(17),(19), one can conclude that, in general, the dynamical system obtained by the iLLG eq.(3) and expressed
by eqs.(14)-(15) has, in each spatial location r ∈Ω, four independent state variables evolving on a fourdimensional state space. This means that the iLLG dynamics requires a double number of degrees of freedom
compared to the classical LLG for its description.
Furthermore, eq.(3) admits an additional conservation property. In fact, by dot-multiplying eq.(15) by
w and integrating over the region Ω, one has:

Z

Z

1
V

ξ
2
∂|w|2

∂t
dV = 1

V

Ω

Ω
−α|w|2 + w · (m × heff) dV ⇔
(22)

Z

Z

1
V

ξ
2
∂|w|2

∂t
dV = 1

V

∂t dV.
(23)

Ω

Ω
−α|w|2 + heff · ∂m

By using the fact that

dg

Z

Z

dt = 1

V

δg
δm · ∂m

∂t + δg

V

∂t −m · ∂ha

∂t dV ,
(24)

δha
· ∂ha
∂t dV = 1

Ω

Ω
−heff · ∂m

5

and remembering from (9) that |w| = |∂m/∂t|, one obtains the following energy balance equation:

Z

Z


g + 1

d
dt

V

ξ
2 |w|2 dV

= −1

V

∂t −α |w|2 dV ⇔

Ω

Ω
m · ∂ha

!

Z

Z

g + 1

2
dV

= −1

2
dV .
(25)

d
dt

V

ξ
2

∂t

V

∂t

Ω

Ω
m · dha

∂m

dt −α

∂m

The latter equation can be put in a more compact form by defining the following generalized free energy:

Z

Z

2
dV
,
(26)

˜g(m, w, ha) = g(m, ha) + 1

V

ξ
2 |w|2 dV = g(m, ha) + 1

V

ξ
2

∂t

Ω

Ω

∂m

where the second term, in the framework of the purely formal mechanical analogy introduced before, can be
seen as a sort of ’kinetic’ energy (see the last equality in eq.(26)) augmenting the classical micromagnetic
free energy interpreted as ’potential’ energy. Thus, the balance equation (25) becomes

d˜g

Z

2
dV ,
(27)

dt = −1

V

∂t

Ω
m · ∂ha

∂t −α

∂m

where the first term at the right-hand side describes energy pumping via time-varying external applied
magnetic field and the second term takes into account the intrinsic dissipation of magnetic materials.
It is apparent that, under the assumption of constant-in-time (even spatially-inhomogeneous) applied
field (∂ha/∂t = 0), the generalized free energy ˜g must be a decreasing function of time:

d˜g

Z

2
dV ≤0 ,
(28)

dt = −1

V

∂t

Ω
α

∂m

which reveals a Lyapunov structure for the iLLG in terms of the generalized free energy ˜g similarly to
what happens for the LLG dynamics in terms of then classical free energy g. This means, that, under
the above assumptions, the only possible attractors of the dynamics are stable equilibria (i.e. such that
∂m/∂t = 0, ∂w/∂t = 0 and ˜g is minimum).
In addition, in the absence of dissipation (α = 0), one has the conservation property for the quantity ˜g:

!

d˜g

Z

2
dV

= 0
,
(29)

g + 1

dt = d

dt

V

ξ
2

∂t

Ω

∂m

|m(r, t)| = 1
∀r ∈Ω, t ≥t0
(30a)

w(r, t) · m(r, t) = w(r, t0) · m(r, t0)
∀r ∈Ω, t ≥t0
(30b)

which is analogous to the conservation of ’total’ (potential + ’kinetic’) energy ˜g in mechanical systems and
here strikingly expresses the conservative nature of the (lossless) spin nutation dynamics.
We remark that the balance equation (25),(27) could have been derived directly from eq.(3) by dotmultiplying both sides by the quantity in parentheses and integrating over Ω.
Finally, we observe that, in the absence of dissipation (i.e. α = 0), eqs.(14)-(15) admit three integrals of
motion:






Z

˜g = g + 1

2
dV = ˜g0 , t ≥t0
(30c)

V

ξ
2

∂t






Ω

∂m

that we term amplitude, ’angular momentum’ projection on magnetization and ’total’ free energy conservation, respectively. The former two hold in a pointwise fashion, that is in any location and time instant
(provided that they are fulfilled at initial time t0), while the last is an integral constraint on magnetization
motion (we remark that ˜g(t0) = ˜g0 is the initial ’total’ free energy).

6

The above conservation laws hold for spatially-inhomogeneous magnetization processes, but one can also
consider ’sufficiently small’ particles where the exchange interaction strongly penalizes spatial magnetization
gradients and, thus, approximately treat them as uniformly-magnetized (macrospin) anisotropic particles,
which eliminates the dependence on the spatial location r within the ferromagnet. This makes sense when
dealing with magnetic nanosystems of dimensions in the order of the exchange length, such as those used as
elementary cells for magnetic memories and other spintronic devices[1]. Under the assumption of spatiallyuniform magnetization and anisotropy of uniaxial type, the free energy (5) has the simple expression[39]:

g(m, ha) = 1

2Dxm2
x + 1

2Dym2
y + 1

2Dzm2
z −m · ha
,
(31)

|m(t)| = 1
∀, t ≥t0 ,
(32a)

w(t) · m(t) = w(t0) · m(t0)
∀, t ≥t0 ,
(32b)

where Dx, Dy, Dz are effective demagnetizing factors taking into account shape and crystalline anisotropy.
The aforementioned integrals of motion (30a)-(30c) become






˜g = g + ξ

2
= ˜g0 , t ≥t0 ,
(32c)

2 |w|2 = g + ξ

2

dt






dm

with g given by eq.(31) and will be instrumental in the validation of time-stepping techniques that we will
discuss in the following sections.

3. Spatially semi-discretized iLLG equation

Now we proceed to the numerical discretization of the iLLG equation. In the following, we will refer
to spatially semi-discretized equations on a collection of N mesh points (rj)N
j=1 associated with the related
computational cells of volume Vj. This description is quite general and works both for finite-difference and
finite-element methods.
We will denote as m(t) = (m1, . . . , mN)T , w(t) = (w1, . . . , wN)T ∈R3N (the notation T means matrix
transpose) the mesh vectors containing all cell vectors mj(t), wj(t) ∈R3 with j = 1, . . . , N.
Moreover, we will use the operator notation for the cross-product for both cell and mesh vectors, namely:

Λ(v) · w = v × w ,
Λ(v) · w = (v1 × w1, . . . , vN × wN)T ,
(33)

meaning that the latter operator is a skew-symmetric 3N × 3N block-diagonal operator that provides cross
product of homologous cell vectors.
Thus, the semi-discretized iLLG equation will read as:

dm

dt = Λ(w) · m
,
(34)

ξ dw

dt = −Λ(m) · w −αw + Λ(m) · heff
,
(35)

where the discrete effective field heff is given by:

heff(m, t) = −∂g

∂m = −C · m(t) + ha(t) ,
(36)

the symmetric positive-definite matrix C plays the role of the effective field operator C and g(m, ha) is the
discrete counterpart of the free energy defined by eq. (5):

g(m, ha) = 1

2mT · C · m −hT
a · m .
(37)

7

By using the same line of reasoning as for the continuous iLLG equation (14)-(15), one can derive the
following conservation properties:

|mj(t)| = |mj(t0)|
∀t ≥t0,
j = 1, . . . , N ,
(38)

(wj(t) · mj(t)) = (wj(t0) · mj(t0))e−α

ξ t
∀t ≥t0,
j = 1, . . . , N ,
(39)

2!

2
−m · dha

= d

g(m(t), ha(t)) + ξ


g(m(t), ha(t)) + ξ

d
dt

2

dt

dt

2 |w|2

= d˜g

dt

dt ,
(40)

dm

dt = −α

dm

where ˜g(m, w, ha) = g(m, ha) + ξ

2|w|2 is the discrete ’total’ energy corresponding to ˜g in the continuous
iLLG dynamics (see eq.(26)).

4. Midpoint time-steppings for iLLG dynamics

The numerical solution of eqs.(34)-(35) with classical time-stepping techniques in general will corrupt the
conservation properties (38)-(40) of semi-discretized inertial magnetization dynamics. Thus, such properties
will be fulfilled with an accuracy depending on the amplitude of the time-step ∆t. For the classical purely
precessional LLG equation, it has been shown[38] that the implicit midpoint rule technique preserves the
properties of discrete magnetization dynamics regardless of the time-step. Here we propose two schemes
based on such technique for the iLLG spin nutation dynamics.

4.1. Implicit midpoint rule (IMR)

The first is based on discretiztion of eqs.(34)-(35) at time tn+ 1

2 = tn + ∆t/2 with the following secondorder midpoint formulas:

mn+ 1

2 = mn+1 + mn

2 = wn+1 + wn

2
wn+ 1

2
,
(41)

where mn, wn denote m(tn), w(tn), which leads to the following time-stepping for the j−th computational
cell:

mjn+1 −mjn

2
j
× m
n+ 1

2
j
,
(42)

∆t
= w
n+ 1

2
j
× w
n+ 1

2
j
−αw
n+ 1

2
j

ξ
wn+1
j
−wn
j
∆t
= −m
n+ 1

+ m
n+ 1

2 , tn+ 1

2 )
,
j = 1, . . . , N .
(43)

2
j
× heff,j(mn+ 1

Now, by dot-multiplying the first equation by mn+1/2
j
, one can easily see that

|mn+1
j
|2 −|mn
j |2 = 0 ,
j = 1, . . . , N ,
(44)

2 and eq.(43) by mn+ 1

meaning that magnetization amplitude will be preserved unconditionally, namely independently of ∆t in each
computational cell. In addition, by dot-multiplying eq.(42) by wn+ 1

2 and summing
both sides of the result, one immediately ends up with:

2
j
· m
n+ 1

2
j
,
j = 1, . . . , N ,
(45)

wn+1
j
+ wn
j
2
·
mn+1
j
−mn
j
∆t
+
mn+1
j
+ mn
j
2
·
wn+1
j
−wn
j
∆t
= −α

ξ w
n+ 1

which expresses the reproduction of the property (39) in its mid-point time discretized version:

2
j
· m
n+ 1

2
j
,
j = 1, . . . , N .
(46)

wn+1
j
· mn+1
j
−wn
j · mn
j
∆t
= −α

ξ w
n+ 1

8

Remarkably enough, in the conservative case α = 0, the latter equation becomes:

wn+1
j
· mn+1
j
−wn
j · mn
j
∆t
= 0
⇒
wn+1
j
· mn+1
j
= wn
j · mn
j ,
j = 1, . . . , N
(47)

which means that the ’angular momentum’ projection conservation property is fulfilled for any choice of the
time step ∆t.
Now let us consider the midpoint rule time-stepping for the mesh vectors:

mn+1 −mn

2 ) · mn+ 1

2
,
(48)

∆t
= Λ(wn+ 1

ξ wn+1 −wn

2 ) · wn+ 1

2 −αwn+ 1

2 + Λ(mn+ 1

2 , tn+ 1

2 )
.
(49)

2 ) · heff(mn+ 1

∆t
= −Λ(mn+ 1

By assuming constant applied field, dot-multiplying the second equation by wn+1/2 and taking into
account eq.(36) and the symmetry of the matrix C, one obtains the following discretized energy balance:

2 |2 −gn+1 −gn

ξ
2
|wn+1|2 −|wn|2

∆t
= −α|wn+ 1

∆t
⇔
(50)

˜gn+1 −˜gn

2 |2
,

∆t
= −α|wn+ 1

where ˜gn = ˜g(mn), which implies that the total (discrete) energy ˜g must be either decreasing when α > 0
or being conserved when α = 0, both regardless of the time-step.
Equations (42)-(43) represent a nonlinear system of 6N coupled scalar equations, which must be solved
at each time step. They can be regarded as the following two vector equations in u = mn+1, v = wn+1:

F (u, v) = 0
,
G(u, v) = 0 ,
(51)

and can be solved by using Newton-Raphson iteration:

∂F

∂F

!−1

∂u


−


,
(52)

∂v
∂G

∂G

 uk+1
vk+1


=
 uk
vk

·
 F (uk)
G(vk)

∂u

∂v

where the partial Jacobian matrices are given by:

∂F

∂u = I

∆t −1

4Λ(v + wn) ,
(53)

∂F

∂v = 1

4Λ(u + mn) ,
(54)

∂G

u + mn


,
(55)


heff

∂u = −1

4Λ(v + wn) + 1

2Λ(u + mn) · C −Λ

2

∂G

∂v = ξ

∆t I + 1

4Λ(u + mn) + α

2 I ,
(56)

and the linear operator notation Λ has been used for the cross product involving mesh vectors.
The above time-stepping has remarkable qualitative properties that reproduce those of the continuous
iLLG dynamics and, therefore, represents the preferred choice to realize inertial micromagnetic numerical
codes for the analysis of terahertz magnetization dynamics.
However, it evidently requires to double the state variables and, consequently, the number of unknowns,
owing to the introduction of the vector field w although one is mainly interested to compute the dynamics
of magnetization vector field m. This issue becomes even more pronounced when large-scale micromagnetic
simulations with full spatial discretization are considered, which would require dramatic modification of

9

numerical codes in order to introduce the auxiliary variable w and to solve a system of 6N nonlinear
coupled equations at each time-step. Moreover, we remark that in the latter situation, the 3N × 3N matrix
C involved in the Newton iteration (see eq.(55)) is also fully-populated owing to the long-range nature of
magnetostatic interactions. Therefore, following what is done for the classical LLG equation[38], a quasiNewton technique is required to solve the large nonlinear system, implemented by considering reasonable and
sparse approximation of the matrix C (e.g. obtained retaining only exchange and anisotropy contributions
Cex and Can, respectively) and, in turn, of the full Jacobian defined by eqs.(53)-(56).
Of course, the
computational cost of such quasi-Newton method, involving the solution of several non-symmetric large
linear systems (e.g. by using GMRES methods[40]), will be at least double with respect to that associated
with LLG equation, posing a strong limit to the capability of solving large-scale iLLG dynamics.

4.2. Implicit midpoint with multi-step inertial term (IMR-MS)
For these reasons, in order to obtain an alternative efficient numerical technique with minimum modification of existing micromagnetic codes, we propose a second time-stepping based on the implicit midpoint
rule combined with a multi-step method for the inertial term. This technique is based on direct space-time
discretization of eq.(3) at time tn+ 1

2 = tn + ∆t/2:

mn+1 −mn

2 , tn+ 1

2

−αmn+1 −mn


.
(57)

2 ) ·

heff

mn+ 1

2

∆t
= −Λ(mn+ 1

∆t
−ξ d2m

dt2

n+ 1

Then, in order to retain the amplitude conservation property (44) of the aforementioned IMR scheme,
we use the first of midpoint formulas (41) in eq.(57) in a way that it is rewritten as system of 3N nonlinear
equations in the unknowns mn+1:
F n(mn+1) = 0
,
(58)

where F n(y) : R3N →R3N is the following vector function:

y + mn

F n(y) =

I −αΛ

  
y −mn
−∆t f n
y + mn


−∆t ξ d2m


,
(59)

2

2

2

dt2

n+ 1

and where


(60)


m, tn + ∆t


= Λ(m) · ∂g


tn + ∆t

f n(m) = −Λ(m) · heff


m, ha

2

∂m

2

is the purely precessional term in the right-hand-side of the conservative semi-discretized iLLG equation
(35) expressed by using the definition (36) of the discrete effective field.
Then, we adopt a multi-step approach with a p−points backward formula for the second derivative in
the inertial term appearing in eqs.(57) and (59):

d2m

p≥3
X

dt2

2 ≈
1
∆t2

n+ 1

k=1
an+2−kmn+2−k = ∆2
p ,
(61)

where the coefficients an+2−k are determined from truncation error analysis in order to control the accuracy
of the approximation. This technique implies a slight modification of existing numerical codes based on
implicit midpoint rule time-stepping. In fact, once formula (61) is plugged into the time-stepping equation
(57), the solution of the nonlinear coupled equations (58) can be obtained by using the Newton-Raphson
technique[38] as follows:

y0 = mn ,
yk+1 = yk + ∆yk+1
with
Jn
F (yk, tn)∆yk+1 = −F n(yk) ,
(62)

by simply considering the following augmented Jacobian matrix of the iteration:

p≥3
X

−

JF (u, t) = I

∆t + α

∆tΛ(mn) +
ξ
2∆t2 a1Λ(u) +
ξ
2∆t2 Λ

k=2
an+2−kmn+2−k
!

u + mn


(63)

−
ξ
2∆t2 Λ(u + mn)a1 −1

2Jf

2
, t + ∆t

2

10

p
order
Coefficients
scheme
3
O(∆t)
a1 = 1, a0 = −2, a−1 = 1
IMR-MS1
4
O(∆t2)
a1 = 3/2, a0 = −7/2, a−1 = 5/2, a−2 = −1/2
IMR-MS2

Table 1: Table of coefficients for multi-step formula (61).

where Jf(u, t) = Λ(u) · C + Λ[heff(u, t)]. The linear system in eq.(62) is solved at each iteration k by
considering the sparse approximation of the full matrix C as C ≈Cex + Can in the Jacobian Jf and using
the GMRES method[40] until the residual ∥F n(yk)∥decreases below a prescribed tolerance.
Now, if one assumes that initially magnetization has zero velocity dm/dt(t = 0) = 0, which is reasonable
in simulation of experimental situations, then at the first time step n = 1 one has mn+2−k = 0, k > 2. For
the subsequent steps n > 1, one can use magnetization samples from the previous steps as mn+2−k, k > 2
in eq.(57). The only cost of such operation is for the storage of p −2 magnetization vectors.
In this respect, the simplest choice is the classical p = 3 points formula:

d2m

2 ≈mn+1 + mn−1 −2mn

dt2

∆t2
= ∆2
p=3 ,
(64)

n+ 1

which, plugged into eq.(57), defines the IMR-MS1 scheme.
However, an analysis of truncation error reveals that:

∆2
p=3 = d2m

2 −1

2 ∆t + 5

2 ∆t2 + . . . ,
(65)

dt2

2
d3m

dt3

24
d4m

dt4

n+ 1

n+ 1

n+ 1

meaning that the accuracy is just of first order O(∆t) (it would be O(∆t2) if the second derivative was
computed at t = tn). Thus, in order to be consistent with discretization of other terms in eq.(57) at second
order with respect to ∆t, one can derive a more accurate formula using p = 4 points. To this end, we
compute a different second derivative formula:

˜∆2
p=3 = d2m

2 ≈2mn+1 −3mn + mn−2

dt2

3∆t2
,
(66)

n+ 1

which has truncation error such that:

˜∆2
p=3 = d2m

2 −5

2 ∆t + 13

2 ∆t2 + . . . .
(67)

dt2

6
d3m

dt3

24
d4m

dt4

n+ 1

n+ 1

n+ 1

Now we use Richardson extrapolation[41] to cancel O(∆t) order terms in the truncation and define the
following new difference formula (defining the IMR-MS2 scheme):

∆2
p=4 = 5

2∆2
p=3 −3

2
˜∆2
p=3 = 3mn+1 −7mn + 5mn−1 −mn−2

2∆t2
,
(68)

for which the truncation error is O(∆t2):

∆2
p=4 = d2m

2 −7

2 ∆t2 + . . . .
(69)

dt2

24
d4m

dt4

n+ 1

n+ 1

The coefficients for the above multi-step formulas with p = 3, 4 are summarized in table 1.
In order to assess the order of accuracy of the proposed schemes, we consider the conservative iLLG
dynamics (α = 0) and numerically integrate eq.(3) using IMR, IMR-MS1, IMR-MS2 time-steppings for
different time step ∆t and compare the results with a benchmark reference solution obtained by using
standard adaptive time step Dormand-Prince Runge-Kutta (RK45) scheme[42, 43]. Absolute tolerances are
set to 10−14 both for RK45 and for Newton iterations solving eq.(57) with IMR, IMR-MS1, IMR-MS2.
In the left panel of figure 1, we report the global truncation error ||∆m|| arising from time-integration
of iLLG eq.(3) in the interval [0,1] between the proposed schemes and the reference RK45 solution.
A

11

10
0
1−|m|

RK45
IMR
IMR−MS1
IMR−MS2

10
0

10
−10

10
−1

0
5
10
15
20
25
10
−20

10
−2

10
0
∆g/g

10
−3

10
−10

||∆m||

10
−4

0
5
10
15
20
25
10
−20

∆(w⋅ m)

10
−5

10
0

10
−6

10
−10

IMR
IMR−MS1
IMR−MS2
O(∆t)

O(∆t2)

0
5
10
15
20
25
10
−20

10
−4
10
−3
10
−2
10
−1
10
−7

∆t

t  [(γ Ms)−1]

Figure 1: Accuracy tests on conservative (α = 0) iLLG dynamics. The values of parameters are Dx = 0, 1, Dy = 0.2, Dz =
0.7, ha = (0, 0, 0.1), m(t = 0) = (1, 0, 0), ξ = 0.03. (left) Global error ||∆m|| at t = 1 between IMR, IMR-MS1, IMR-MS2
schemes and reference RK45 solution showing first-order O(∆t) behavior for IMR-MS1 and second-order O(∆t2) for IMR and
IMR-MS2. (right) Conservation of properties of iLLG dynamics versus time (time step ∆t = 0.001 for all IMR schemes).
Top panel refers to amplitude 1 −|m| conservation, middle panel to relative error ∆˜g/˜g = (˜g(t) −˜g(0))/˜g(0) in ’total’ free
energy conservation, bottom panel refers to error ∆(w · m) = w(t) · m(t) −w(0) · m(0) in ’angular momentum’ projection on
magnetization conservation. One can see that all IMR, IMR-MS1, IMR-MS2 schemes preserve amplitude |m| and projection
w · m with (double-precision) machine accuracy and only IMR also preserves energy. IMR-MS2 outperforms IMR-MS1 and
RK45 in energy conservation.

quick inspection of the figure confirms the expected first and second orders of accuracy for IMR-MS1 and
IMR,IMR-MS2, respectively. Remarkably, IMR-MS2 has performance quite similar to the fully-implicit IMR
without doubling the number of degrees of freedom. On the other hand, in the right panel of fig.1, one can
look at the conservation properties for the proposed schemes (with ∆t = 0.001) and the reference solution. As
expected, all the three IMR schemes are able to preserve amplitude |m| and ’angular momentum’ projection
on magnetization w · m with (double) machine precision, while only the fully-implicit IMR is able to do so
for the ’total’ energy ˜g. Nevertheless, it is apparent (middle panel, blue and cyan solid lines) that IMR-MS2
is able to guarantee the same precision as the RK45 concerning energy conservation while being significantly
lower-order than RK45.
For the evaluation of the variable w and energy ˜g when considering IMR-MS
schemes, we have used the central difference formula wn = mn × (mn+1 −mn−1)/(2∆t).

5. Numerical results

In order to validate the proposed techniques on physically relevant situations, we perform two different
simulations. The first describes the ultra-fast resonant spin nutation of a uniformly-magnetized thin-film
driven by ac terahertz appled field, similar to the experiment[11] that provided direct evidence of the
presence of inertial effects. This will also be a basic testbed to compare the accuracy of the developed IMR
and IMR-MS schemes. The second simulation will address the ultra-fast spatially-inhomogeneous dynamics
of magnetization in a microscale nanodot excited with terahertz applied field and will demonstrate the
efficiency of the IMR-MS time-stepping in full micromagnetic simulations.

5.1. Nutation frequency response of a single-domain particle

We analyze the frequency response of a thin-film magnetized along the easy y direction and subject
to an out-of-plane ac field with small amplitude. In this situation, one can assume that macrospin iLLG
dynamics occurs in the linear regime and analytical theory can be developed. To this end, let us assume

12

Figure 2: Time-domain linear relaxation of mz computed with IMR, IMR-MS with different time steps and compared with
analytical solution of linear iLLG eq.(70).

that the applied field is decomposed in a nonzero constant bias field plus a time-harmonic component
ha(t) = hdc + hac(t) ,
|hac| ≪|hdc|. We also assume that the free energy has the simple form (31) under
the macrospin approximation.
Then, the iLLG eq.(3) can be linearized around an equilibrium m0 such that m(t) = m0 + ∆m(t) in
the following way:
d∆m


,
(70)

dt
= m0 ×

(D + h0I) · ∆m −αd∆m

dt
−ξ d2∆m

dt2

where D denotes the diagonal matrix D = diag[Dx, Dy, Dz], h0 = heff(m0) · m0 = (−D · m0 + ha) · m0 is
the projection of the equilibrium effective field on equilibrium magnetization.
We first observe that the dynamics fulfills the constraint m0 · ∆m = 0 (to the first order), therefore
we can consider only the dyanamics of the component ∆m⊥of ∆m living in the plane perpendicular to
the equilibrium m0. We also refer to the projection of the demag tensor D as D⊥. As a consequence, we
will deal with vectors having only two components associated with axes transverse to the equilibrium m0.
We observe that the skew-symmetric operator Λ is invertible (such that Λ · Λ = −I) when restricted to
the plane orthogonal to m0 and we express both ∆m⊥, hac using complex (phasor) domain as ∆m⊥(t) =
∆˜mejωt , hac = ˜hacejωt. By using these formulas in eq.(70), one ends up with:

˜hac ,
(71)

∆˜m =

jωΛ(m0) + (D⊥+ h0I) + jωαI −ξω2I
−1
|
{z
}
χ(ω)

which defines the magnetic susceptibility tensor χ(ω). When referred to principal axes, χ(ω) is a 2 × 2
matrix which can be easily computed.
It can be shown that the resonance frequencies associated with the above linear dynamical system are
the roots of the following fourth-degree polynomial:

ξ2ω4 −2jαξω3 −(α2 + ξ(ω0y + ω0z) + 1)ω2 + jα(ω0y + ω0z)ω + ω0yω0z = 0 ,
(72)

where ω0y = Dy −Dx + hdc and ω0y = Dz −Dx + hdc. Equation (72) can be solved by using appropriate
perturbation theory leading to the following resonance frequencies (computed in the conservative case when

13

Figure 3: Frequency response power spectrum |∆˜mz(ω)|2. detail of the nutation peak. The FMR frequency is around 18
GHz whereas the nutation frequency is about 634 GHz. The respective linewidths are about 1 GHz and 27 GHz. Blue line
is eq.(71), red dots and dashed line are analytical formulas (73),(75), black symbols are the result of numerical simulations of
iLLG dynamics with IMR-MS1.

α = 0):

ωFMR ≈±
ωK
p

1 + ξ(ω0y + ω0z)
, ωK = √ω0yω0z ,
(73)

sp

2ξ(ω0y + ω0z) + 1 + ξ(ω0y + ω0z) + 1

ωN ≈±

2ξ2
,
(74)

where ωK =
p

(Dy −Dx + hdc)(Dz −Dx + hdc) is the classical Kittel ferromagnetic resonance (FMR)
frequency. The former equation describes the influence of inertial effects on the FMR frequency, while the
second formula gives the nutation resonance frequency (typically in the THz range). We observe that the
above formulas take into account the dependence on the external bias field through the parameters ω0y, ω0z.
It is also possible to determine closed-form expressions for the half-power (Full Width at Half Maximum,
FWHM) linewidths:

∆ωFMR ≈α(ω0y + ω0z) −ξ(ω2
0y + 4ω0yω0z + ω2
0z) ,
(75)

"

#

.
(76)

∆ωN ≈α

ξ

1 +
1
p

2ξ(ω0y + ω0z) + 1

Here we consider an infinite thin-film (Dx = 0, Dy = 0, Dz = 1) with material parameters: damping
α = 0.023, saturation magnetization such that µ0Ms = 0.93T, inertial time scale τ = 1.26 ps.
In figure 2, we report the time-evolution of mz during relaxation under zero bias field starting from an
initial state tilted in the x −z plane such that mx = 0.01, mz = 0.01. The analytical solution of eq.(70) is
used to benchmark the proposed numerical techniques with different time step amplitudes. It is apparent
that IMR technique allows the use of the largest time steps (up to ∆t = 0.01, around 25 samples per nutation
period) yielding no significant loss of accuracy with respect to the analytical solution. One can also see that
IMR-MS with first-order formula (64) (IMR-MS1) is accurate when ∆t ∼0.0001 (corresponding to 0.61 fs
in physical units), while is not able to follow nutation dynamics after 5-6 periods (the period is 0.27) with
50 times larger time step ∆t = 0.005. Conversely, IMR-MS with second order formula (68) (IMR-MS2)
performs well with ∆t = 0.005 (slightly above 50 samples per period) and provides a measured speedup of

14

about 30 times compared to the former. This occurs since the average number of Newton iterations remains
of the same order of magnitude, namely 2 for IMR-MS1 with ∆t = 0.0001 and 3 for IMR-MS2 with 50
times larger ∆t (the tolerance for Newton-Raphson iteration was set to 10−14). Finally, comparing IMR
and IMR-MS2 methods, one can see that, despite correctly reproducing the nutation oscillation, IMR-MS2
produces a small phase-shift when the largest time step ∆t = 0.01 is chosen.
Next, by using eqs.(70), the frequency response power spectrum of the out-of-plane magnetization component mz has been computed under a bias field hax = 0.35T and ac field directed along y. The iLLG
equation (3) has been solved numerically in order to determine the frequency response power spectrum.
Namely, given the susceptibility χ(ω) in eq.(71) and the cross-power spectrum matrix (∆˜m) · (∆˜m)H =
(χ(ω) · ˜hac) · (χ(ω) · ˜hac)H (H means conjugate transpose) assuming that the only nonzero component of the
input field ˜hac = (˜hy, ˜hz)T is ˜hy = 1 (Fourier Transform of a Dirac delta) and the output response is the
out-of-plane magnetization ∆˜mz, the output power spectrum is:

|∆˜
mz(ω)|2 = |χzy(ω)|2
,
(77)

which is reported in fig.3 and compared with analytical formulas (73),(75). In order to compute |χzy|2 from
time-domain numerical simulations, we apply a sinusoidal field hy(t) at frequency ω0 with sufficiently small
amplitude (in order to stay in the linear regime) and measure the steady-state oscillation of mz(t) after
sufficiently long time so that the transient response has vanished. This has been performed by choosing a
simulated time T = 2ns and ac field amplitude equal to 0.06T. Then, the Fast Fourier Transform Mz(ω) =
F[mz(t)] has been computed and the maximum of the power spectrum |Mz(ω)|2 has been determined.
This procedure has been repeated for several values of ω0. The so determined points form samples of the
frequency response power spectrum |χzy(ω)|2.
In figure 3, numerical simulations performed with IMR-MS1 with ∆t corresponding to 1 fs are used to
compute the output power spectrum (black symbols). The results are in excellent agreement with analytical
theory.

5.2. Spatially-inhomogeneous spin nutation driven by terahertz applied field

In order to perform efficient time-domain micromagnetic simulations of iLLG dynamics with full spatial
discretization, the proposed IMR-MS2 time-stepping has been implemented in the finite-difference numerical
code MaGICo[38, 44], which retains the same computational cost as the simulation of classical precessional
LLG dynamics while keeping important conservation properties as outlined above.
To validate the code, we explore typical spatio-temporal patterns of iLLG dynamics considering the
time-domain simulation of ultra-short inertial spin waves in a confined ferromagnetic thin-film. At terahertz frequencies, the behavior of small magnetization oscillations significantly deviates from the classical
description of exchange-dominated spin-waves, in that an ultimate limiting propagation speed appears[24].
This difference is mostly due to the mathematical structure of eq.(3) compared with the classical LLG precessional dynamics, i.e. the same equation where one sets ξ = 0. In fact, when inertial effects are taken into
account, the torque proportional to the second-order time-derivative transforms the classical LLG equation
into a wave-like equation with hyperbolic mathematical character. In this respect, on short time scales,
finite time delays are expected in magnetization response propagation far from local external excitation.
The considered sample is made of Cobalt and has a thin-film shape with square cross-section 200×200 nm2

and thickness 5 nm. The ferromagnetic nanodot is initially at equilibrium, being saturated along the x axis
by a static field µ0Hax = 100 mT. The value of material parameters are γ = 2.211 × 105 m A−1 s−1,
µ0Ms = 1.6 T, A = 13 pJ/m (lex = 3.57 nm), τ = 0.653 ps (ξ = 0.0338) and α = 0.005.
The applied field is a spatially-uniform sine wave step (turned on at t = 0) along the y axis transverse
to the equilibrium configuration with amplitude µ0Hay = 100 mT and frequency f = 1386 GHz. For the
above choice of parameters, this excitation frequency is slightly above the nutation resonance frequency and
corresponds to inertial spin waves with wavelength around 20 nanometers[24].
The numerical simulation of iLLG equation (3) is performed with a time-step of 25 fs, a 2.5×2.5×5 nm3

computational cell, which corresponds to discretize the thin-film into 80 × 80 square prims cell. In order
to isolate short-wavelength spin wave propagation from the rest of the simulated spatial pattern, high-pass

15

Figure 4: Spatial profiles of short-wavelength magnetization out-of-plane component mz obtained by FFT high-pass filtering
at time t = 1, 10, 30, 49 ps. The magenta solid line is a guide for the eye to follow the spin wave oscillation profile along the x
direction.

spatial filtering via two-dimensional Fast Fourier Transform is performed on magnetization components.
The simulated time is 100 ps and some snapshots of the magnetization out-of-plane component mz, taken
at different time instants, are reported in figure 4.
The magnetization is initially at the equilibrium and mostly aligned with the static field along the x
direction except close to the square corners where there is the most pronounced deviation. Such a tilting acts
as local excitation for inertial spin waves when the time-varying ac field step is applied[24]. In fact, as it can
be seen in the various panels of fig.4, two wavepacket with wavelength ≈20 nm propagate from the edges of
the nanodot toward its center after the application of the ac field, consistently showing a propagation with
finite speed ≈2000 m/s compatible with that predicted by the theory[24].

6. Conclusion

In this paper, we have proposed second-order accurate and efficient numerical schemes for the timeintegration of the ultra-fast inertial magnetization dynamics.
We have shown that the iLLG equation
describes a higher-order dynamical system compared to the classical precessional dynamics which requires
to double the degrees of freedom for its desription. We have derived the fundamental properties of the iLLG
dynamics, namely conservation of magnetization amplitude and ’angular momentum’ projection, Lyapunov
structure and generalized free energy balance properties, and demonstrated that the proposed implicit

16

midpoint rule (IMR) time-stepping is able to correctly reproduce them unconditionally. Suitable Newton
technique has been developed for the inversion of the nonlinearly coupled system of equations to be solved at
each time-step. For large-scale micromagnetic simulations with full spatial discretization, efficient numerical
time-stepping schemes based on implicit midpoint rule combined with appropriate multi-step method for
the inertial term, termed IMR-MS of order 1 and 2, have been proposed. These schemes retain the same
computational cost of the IMR for the classical LLG dynamics while providing conservation of magnetization
amplitude and accurate reproduction of the high frequency nutation oscillations. In particular, thanks to
the unconditional stability due to its implicit nature along with the second-order accuracy on the inertial
term, both the IMR and IMR-MS2 allow choosing moderately large time-steps only based on accuracy
requirements for the description of the nutation dynamics. The proposed techniques have been successfully
validated against test cases of spatially-homogeneous and inhomogeneous magnetization iLLG dynamics
demonstrating their effectiveness. For these reasons, we believe that these numerical schemes can become a
standard de facto in the micromagnetic simulation of inertial magnetization dynamics in nano- and microscale magnetic systems.

Acknowledgements

M.d’A., S.P. and C.S. acknowledge support from the Italian Ministry of University and Research,
PRIN2020 funding program, grant number 2020PY8KTC.

References

[1] B. Dieny, I. L. Prejbeanu, K. Garello, P. Gambardella, P. Freitas, R. Lehndorff, W. Raberg, U. Ebels, S. O. Demokritov,
J. Akerman, A. Deac, P. Pirro, C. Adelmann, A. Anane, A. V. Chumak, A. Hirohata, S. Mangin, S. O. Valenzuela,
M. C. Onba¸slı, M. d’Aquino, G. Prenat, G. Finocchio, L. Lopez-Diaz, R. Chantrell, O. Chubykalo-Fesenko, P. Bortolotti,
Opportunities and challenges for spintronics in the microelectronics industry, Nature Electronics 3 (8) (2020) 446–459.
doi:10.1038/s41928-020-0461-5.
URL https://doi.org/10.1038/s41928-020-0461-5
[2] E. Beaurepaire, J.-C. Merle, A. Daunois, J.-Y. Bigot, Ultrafast spin dynamics in ferromagnetic nickel, Physical Review
Letters 76 (22) (1996) 4250–4253. doi:10.1103/physrevlett.76.4250.
URL https://doi.org/10.1103/physrevlett.76.4250
[3] B. Koopmans, M. van Kampen, J. T. Kohlhepp, W. J. M. de Jonge, Ultrafast magneto-optics in nickel: Magnetism or
optics?, Physical Review Letters 85 (4) (2000) 844–847. doi:10.1103/physrevlett.85.844.
URL https://doi.org/10.1103/physrevlett.85.844
[4] C. Stamm, T. Kachel, N. Pontius, R. Mitzner, T. Quast, K. Holldack, S. Khan, C. Lupulescu, E. F. Aziz, M. Wietstruk,
H. A. D¨urr, W. Eberhardt, Femtosecond modification of electron localization and transfer of angular momentum in nickel,
Nature Materials 6 (10) (2007) 740–743. doi:10.1038/nmat1985.
URL https://doi.org/10.1038/nmat1985
[5] C. D. Stanciu, F. Hansteen, A. V. Kimel, A. Kirilyuk, A. Tsukamoto, A. Itoh, T. Rasing, All-optical magnetic recording
with circularly polarized light, Physical Review Letters 99 (4) (2007) 047601. doi:10.1103/physrevlett.99.047601.
URL https://doi.org/10.1103/physrevlett.99.047601
[6] A. V. Kimel, B. A. Ivanov, R. V. Pisarev, P. A. Usachev, A. Kirilyuk, T. Rasing, Inertia-driven spin switching in
antiferromagnets, Nature Physics 5 (10) (2009) 727–731. doi:10.1038/nphys1369.
URL https://doi.org/10.1038/nphys1369
[7] A. Kirilyuk, A. V. Kimel, T. Rasing, Ultrafast optical manipulation of magnetic order, Reviews of Modern Physics 82 (3)
(2010) 2731–2784. doi:10.1103/revmodphys.82.2731.
URL https://doi.org/10.1103/revmodphys.82.2731
[8] C.-H. Lambert, S. Mangin, B. S. D. C. S. Varaprasad, Y. K. Takahashi, M. Hehn, M. Cinchetti, G. Malinowski, K. Hono,
Y. Fainman, M. Aeschlimann, E. E. Fullerton, All-optical control of ferromagnetic thin films and nanostructures, Science
345 (6202) (2014) 1337–1340. doi:10.1126/science.1253493.
URL https://doi.org/10.1126/science.1253493
[9] C. Dornes, Y. Acremann, M. Savoini, M. Kubli, M. J. Neugebauer, E. Abreu, L. Huber, G. Lantz, C. A. F. Vaz, H. Lemke,
E. M. Bothschafter, M. Porer, V. Esposito, L. Rettig, M. Buzzi, A. Alberca, Y. W. Windsor, P. Beaud, U. Staub,
D. Zhu, S. Song, J. M. Glownia, S. L. Johnson, The ultrafast Einstein–de Haas effect, Nature 565 (7738) (2019) 209–212.
doi:10.1038/s41586-018-0822-7.
URL https://doi.org/10.1038/s41586-018-0822-7
[10] M. Hudl, M. d’Aquino, M. Pancaldi, S.-H. Yang, M. G. Samant, S. S. Parkin, H. A. D¨urr, C. Serpico, M. C. Hoffmann,
S. Bonetti, Nonlinear magnetization dynamics driven by strong terahertz fields, Physical Review Letters 123 (19) (2019)
197204. doi:10.1103/physrevlett.123.197204.
URL https://doi.org/10.1103/physrevlett.123.197204

17

[11] K. Neeraj, N. Awari, S. Kovalev, D. Polley, N. Z. Hagstr¨om, S. S. P. K. Arekapudi, A. Semisalova, K. Lenz,
B. Green, J.-C. Deinert, I. Ilyakov, M. Chen, M. Bawatna, V. Scalera, M. d’Aquino, C. Serpico, O. Hellwig, J.E. Wegrowe, M. Gensch, S. Bonetti, Inertial spin dynamics in ferromagnets, Nature Physics 17 (2) (2020) 245–250.
doi:10.1038/s41567-020-01040-y.
URL https://doi.org/10.1038/s41567-020-01040-y
[12] V. Unikandanunni, R. Medapalli, M. Asa, E. Albisetti, D. Petti, R. Bertacco, E. E. Fullerton, S. Bonetti, Inertial spin
dynamics in epitaxial cobalt films, Physical Review Letters 129 (2022) 237201. doi:10.1103/PhysRevLett.129.237201.
URL https://link.aps.org/doi/10.1103/PhysRevLett.129.237201
[13] M.-C. Ciornei, J. M. Rub´ı, J.-E. Wegrowe, Magnetization dynamics in the inertial regime: Nutation predicted at short
time scales, Physical Review B 83 (2) (2011) 020410(R). doi:10.1103/physrevb.83.020410.
URL https://doi.org/10.1103/physrevb.83.020410
[14] E. Olive, Y. Lansac, J.-E. Wegrowe, Beyond ferromagnetic resonance: The inertial regime of the magnetization, Applied
Physics Letters 100 (19) (2012) 192407. doi:10.1063/1.4712056.
URL https://doi.org/10.1063/1.4712056
[15] R. Mondal, M. Berritta, A. K. Nandy, P. M. Oppeneer, Relativistic theory of magnetic inertia in ultrafast spin dynamics,
Physical Review B 96 (2) (2017) 024425. doi:10.1103/physrevb.96.024425.
URL https://doi.org/10.1103/physrevb.96.024425
[16] C. Serpico, M. d’Aquino, G. Bertotti, I. D. Mayergoyz, Quasiperiodic magnetization dynamics in uniformly magnetized
particles and films, Journal of Applied Physics 95 (11) (2004) 7052–7054. doi:10.1063/1.1689211.
URL http://aip.scitation.org/doi/10.1063/1.1689211
[17] M. Bauer, J. Fassbender, B. Hillebrands, R. L. Stamps, Switching behavior of a stoner particle beyond the relaxation time
limit, Physical Review B 61 (5) (2000) 3410–3416. doi:10.1103/physrevb.61.3410.
URL https://doi.org/10.1103/physrevb.61.3410
[18] G. Bertotti, I. Mayergoyz, C. Serpico, M. d’Aquino, Geometrical analysis of precessional switching and relaxation in
uniformly magnetized bodies, IEEE Transactions on Magnetics 39 (5) (2003) 2501–2503. doi:10.1109/TMAG.2003.816453.
URL http://ieeexplore.ieee.org/document/1233123/
[19] M. d’Aquino, W. Scholz, T. Schrefl, C. Serpico, J. Fidler, Numerical and analytical study of fast precessional switching,
Journal of Applied Physics 95 (11) (2004) 7055–7057. doi:10.1063/1.1689910.
URL http://aip.scitation.org/doi/10.1063/1.1689910
[20] T. Devolder, H. W. Schumacher, C. Chappert, Precessional Switching of Thin Nanomagnets with Uniaxial Anisotropy,
Springer Berlin Heidelberg, Berlin, Heidelberg, 2006, pp. 1–55. doi:10.1007/10938171\_1.
URL https://doi.org/10.1007/10938171_1
[21] K. Neeraj, M. Pancaldi, V. Scalera, S. Perna, M. d’Aquino, C. Serpico, S. Bonetti, Magnetization switching in the inertial
regime, Physical Review B 105 (2022) 054415. doi:10.1103/PhysRevB.105.054415.
URL https://link.aps.org/doi/10.1103/PhysRevB.105.054415
[22] L. Winter, S. Großenbach, U. Nowak, L. R´ozsa, Nutational switching in ferromagnets and antiferromagnets, Physical
Review B 106 (21) (2022) 214403. doi:10.1103/physrevb.106.214403.
URL https://doi.org/10.1103/physrevb.106.214403
[23] W. F. Brown, Micromagnetics, Interscience Publishers, 1963.
[24] M. d'Aquino, S. Perna, M. Pancaldi, R. Hertel, S. Bonetti, C. Serpico, Micromagnetic study of inertial spin waves in
ferromagnetic nanodots, Physical Review B 107 (14) (Apr. 2023). doi:10.1103/physrevb.107.144412.
URL https://doi.org/10.1103/physrevb.107.144412
[25] T. Kikuchi, G. Tatara, Spin dynamics with inertia in metallic ferromagnets, Physical Review B 92 (18) (2015) 184410.
doi:10.1103/physrevb.92.184410.
URL https://doi.org/10.1103/physrevb.92.184410
[26] S. Giordano, P.-M. D´ejardin, Derivation of magnetic inertial effects from the classical mechanics of a circular current loop,
Physical Review B 102 (21) (2020) 214406. doi:10.1103/physrevb.102.214406.
URL https://doi.org/10.1103/physrevb.102.214406
[27] I. Makhfudz, E. Olive, S. Nicolis, Nutation wave as a platform for ultrafast spin dynamics in ferromagnets, Applied Physics
Letters 117 (13) (2020) 132403. doi:10.1063/5.0013062.
URL https://doi.org/10.1063/5.0013062
[28] A. M. Lomonosov, V. V. Temnov, J.-E. Wegrowe, Anatomy of inertial magnons in ferromagnetic nanostructures, Physical
Review B 104 (2021) 054425. doi:10.1103/PhysRevB.104.054425.
URL https://link.aps.org/doi/10.1103/PhysRevB.104.054425
[29] M. Cherkasskii, M. Farle, A. Semisalova, Dispersion relation of nutation surface spin waves in ferromagnets, Phys. Rev.
B 103 (2021) 174435. doi:10.1103/PhysRevB.103.174435.
URL https://link.aps.org/doi/10.1103/PhysRevB.103.174435
[30] R. Mondal, L. R´ozsa, Inertial spin waves in ferromagnets and antiferromagnets, Physical Review B 106 (13) (2022) 134422.
doi:10.1103/physrevb.106.134422.
URL https://doi.org/10.1103/physrevb.106.134422
[31] S. V. Titov, W. J. Dowling, Y. P. Kalmykov, M. Cherkasskii, Nutation spin waves in ferromagnets, Physical Review B
105 (21) (2022) 214414. doi:10.1103/physrevb.105.214414.
URL https://doi.org/10.1103/physrevb.105.214414
[32] Z. Gareeva, K. Guslienko, Nutation excitations in the gyrotropic vortex dynamics in a circular magnetic nanodot, Nanomaterials 13 (3) (2023) 461. doi:10.3390/nano13030461.

18

URL https://doi.org/10.3390/nano13030461
[33] P. E. Wigen, Nonlinear Phenomena and Chaos in Magnetic Materials, WORLD SCIENTIFIC, 1994. doi:10.1142/1686.
URL https://doi.org/10.1142/1686
[34] E. A. Montoya, S. Perna, Y.-J. Chen, J. A. Katine, M. d’Aquino, C. Serpico, I. N. Krivorotov, Magnetization reversal
driven by low dimensional chaos in a nanoscale ferromagnet, Nature Communications 10 (1) (Feb. 2019). doi:10.1038/
s41467-019-08444-2.
URL https://doi.org/10.1038/s41467-019-08444-2
[35] M. Ruggeri, Numerical analysis of the landau–lifshitz–gilbert equation with inertial effects, ESAIM: Mathematical Modelling and Numerical Analysis 56 (4) (2022) 1199–1222. doi:10.1051/m2an/2022043.
URL https://doi.org/10.1051/m2an/2022043
[36] P. Li, L. Yang, J. Lan, R. D. null, J. Chen, A second-order semi-implicit method for the inertial landau-lifshitz-gilbert
equation, Numerical Mathematics:
Theory, Methods and Applications 16 (1) (2023) 182–203.
doi:10.4208/nmtma.
oa-2022-0080.
URL https://doi.org/10.4208/nmtma.oa-2022-0080
[37] M. d’Aquino, C. Serpico, G. Miano, I. D. Mayergoyz, G. Bertotti, Numerical integration of Landau–Lifshitz–Gilbert
equation based on the midpoint rule, Journal of Applied Physics 97 (10) (2005) 10E319. doi:10.1063/1.1858784.
URL http://aip.scitation.org/doi/10.1063/1.1858784
[38] M. d’Aquino, C. Serpico, G. Miano, Geometrical integration of Landau–Lifshitz–Gilbert equation based on the mid-point
rule, Journal of Computational Physics 209 (2) (2005) 730–753. doi:10.1016/j.jcp.2005.04.001.
URL https://doi.org/10.1016/j.jcp.2005.04.001
[39] I. Mayergoyz, G. Bertotti, C. Serpico, Nonlinear Magnetization Dynamics in Nanosystems, Elsevier Science, 2009.
[40] Y. Saad, M. H. Schultz, GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems,
SIAM Journal on Scientific and Statistical Computing 7 (3) (1986) 856–869. doi:10.1137/0907058.
URL https://doi.org/10.1137/0907058
[41] L. F. Richardson, IX. the approximate arithmetical solution by finite differences of physical problems involving differential
equations, with an application to the stresses in a masonry dam, Philosophical Transactions of the Royal Society of
London. Series A, Containing Papers of a Mathematical or Physical Character 210 (459-470) (1911) 307–357.
doi:
10.1098/rsta.1911.0009.
URL https://doi.org/10.1098/rsta.1911.0009
[42] J. Dormand, P. Prince, A family of embedded runge-kutta formulae, Journal of Computational and Applied Mathematics
6 (1) (1980) 19–26. doi:10.1016/0771-050x(80)90013-3.
URL https://doi.org/10.1016/0771-050x(80)90013-3
[43] L. F. Shampine, M. W. Reichelt, The matlab ode suite, SIAM Journal on Scientific Computing 18 (1) (1997) 1–22.
arXiv:https://doi.org/10.1137/S1064827594276424, doi:10.1137/S1064827594276424.
URL https://doi.org/10.1137/S1064827594276424
[44] M. d’Aquino, Magnetization Geometrical Integration Code (MaGICo), http://wpage.unina.it/mdaquino/index_file/
MaGICo.html.

19

---

*Source: arXiv:2310.09043 / ACM Computing Surveys*
