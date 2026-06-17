---
id: A6
title: "AI-Assisted Software Engineering: A Systematic Literature Review"
domain: A
year: 2024
arxiv_id: "2405.03258"
confidence: verified
source: "IEEE Transactions on Software Engineering"
node_type: paper
---

# AI-Assisted Software Engineering: A Systematic Literature Review

**Domain**: [[domain_A|Vibe Coding / Prompt-Driven Development]] | **Year**: 2024 | **Confidence**: [x] verified




## Keywords
- [[kw_AI-assisted SE|AI-assisted SE]]
- [[kw_systematic review|systematic review]]
- [[kw_software engineering lifecycle|software engineering lifecycle]]

## Abstract

(Abstract not available - see PDF content below)

## Paper Content

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG
CATEGORIES

DOGANCAN KARABAS AND SANGJIN LEE

Abstract. We give a speciﬁc cylinder functor for semifree dg categories. This allows us to construct a homotopy colimit functor explicitly. These two functors are “computable”, speciﬁcally, the
constructed cylinder functor sends a dg category of strictly ﬁnite type, i.e., a semifree dg category
having ﬁnitely many objects and generating morphisms, to a dg category of strictly ﬁnite type.
The homotopy colimit functor has a similar property.
Moreover, using the cylinder functor, we
give a coﬁbration category of semifree dg categories and that of dg categories of strictly ﬁnite type,
independently from the work of Tabuada [Tab05b]. All the results similarly work for semifree dg
algebras. We also describe an application to symplectic topology and provide a toy example.

Contents

1.
Introduction
1

2.
Preliminaries on dg categories
4

3.
Homotopy colimit functor using model structures
7

4.
Cylinder functor for the category of dg categories
11

5.
Homotopy colimit functor on the diagrams of semifree dg categories
17

6.
I-category and coﬁbration category of semifree dg categories
22

7.
Wrapped Fukaya category of T ∗Sn and the reﬂection functor
32

References
40

arXiv:2405.03258v1  [math.CT]  6 May 2024

1. Introduction

Homotopy theory, originating in algebraic topology, plays a pivotal role in numerous areas of modern mathematics. Speciﬁcally, the homotopy theory of diﬀerential graded (dg) categories is prominently featured in ﬁelds such as algebraic geometry, representation theory, higher categories, and
symplectic topology. To explore their homotopy theory up to various weak equivalences, Tabuada
[Tab05b, Tab05a] introduced model structures for the category of dg categories. These model structures come with auxiliary functors called coﬁbrations and ﬁbrations, and provide two functorial
factorizations of functors: the ﬁrst type factors a functor into a coﬁbration followed by a weak
equivalence, and the second type factors it into a weak equivalence followed by a ﬁbration.

There is another approach to the homotopy theory of dg categories without using a model structure: Starting with coﬁbrations, instead of considering the entire factorization data, one can focus
solely on the functorial factorization of codiagonal maps into a coﬁbration and a weak equivalence.

Date: May 7, 2024.
1

2
DOGANCAN KARABAS AND SANGJIN LEE

This functorial factorization yields a construction of a cylinder functor (Deﬁnition 4.1). Once additional axioms (see Deﬁnition 6.1) are established, this construction recovers an entire functorial
factorization of the ﬁrst type, representing “half” of a model structure known as a coﬁbration category in the sense of [Bau89]. Therefore, constructing a cylinder functor allows us to delve into the
homotopy theory of dg categories, speciﬁcally allowing us to describe the homotopy colimit functor
on diagrams of dg categories.

In the current paper, we present a simple construction of a cylinder functor on the category of dg
categories, expanding upon our earlier work [KL21] where the cylinder functor was deﬁned solely at
the level of objects. Moreover, our cylinder functor leads us to a simple homotopy colimit functor
and establishes an I-category and a coﬁbration category of dg categories, oﬀering a computational
approach to the homotopy theory of dg categories. Detailed results can be found in Theorems 1–3,
which will be discussed following the introduction of our framework. We will also comment on the
simplicity of our constructions.

Throughout this paper, we mostly work within the context of the category of semifree dg categories
over a commutative ring k, denoted by dgCats (see Deﬁnition 2.1). We note that every dg category
has a semifree resolution (as discussed in [Dri04]), emphasizing the signiﬁcance of our focus on
dgCats. Furthermore, we ﬁx weak equivalences as quasi-equivalences, pretriangulated equivalences,
or Morita equivalences.

Theorem 1. [Corollary 4.9] Let Cyl: dgCats →dgCats be a functor together with the natural
transformations i1, i2 : 1dgCats ⇒Cyl and p: Cyl ⇒1dgCats deﬁned as follows:

• Cyl(C) for any C ∈dgCats and the natural transformations are as in Theorem 4.5,
• Cyl(F) for any morphism (dg functor) F : A →B in dgCats is as in Theorem 4.8.

Then, Cyl is a cylinder functor, which means the following conditions are satisﬁed:

• p ◦(i1 ∐i2): C ∐C →Cyl(C) →C is the codiagonal of A,
• i1 ∐i2 : C ∐C →Cyl(C) is a coﬁbration,
• p: Cyl(C) →C is a weak equivalence and a ﬁbration.

We note that our cylinder functor diﬀers signiﬁcantly from the natural cylinder functor induced
by Tabuada’s model structures. Tabuada employed Quillen’s “small object argument” (as detailed
in Hovey [Hov07]) to establish a functorial factorization, which results in a cylinder functor as a
special case.
However, this method (or its reﬁnement by Garner [Gar09]) involves a transﬁnite
construction that is overly complicated from a computational standpoint.

In contrast, our cylinder functor Cyl ensures that the number of generating morphisms of a
semifree dg category C and its image Cyl(C) are close in size. Speciﬁcally, if the former is ﬁnite, the
latter is also ﬁnite, a property that does not hold for Tabuada’s natural cylinder. This eﬀectiveness
arises from the construction of Cyl, which is speciﬁcally tailored for semifree dg categories. Moreover,
the computability of our cylinder functor ensures that all our functorial factorizations are computable
as can be seen in Theorem 6.9.

Using the cylinder functor described earlier, we can construct the homotopy colimit (pushout)
functor for the category of dg categories, where weak equivalences are deﬁned as quasi-equivalences,
pretriangulated equivalences, or Morita equivalences:

Theorem 2. [Theorem 5.3] Let J be a category of the form a ←c →b, and dgCatJ denote the
category of J-diagrams in dgCat. Then, the homotopy colimit functor

hocolim: Ho(dgCatJ) →Ho(dgCat)

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
3

can be given such that hocolim(C) for any C ∈dgCats is deﬁned as in Theorem 5.3(1), and
hocolim(F) for any morphism (dg functor) F : A →B in dgCats is deﬁned as in Theorem 5.3(2).

Our homotopy colimit functor is simple in the sense that it produces a semifree dg category of a
size close to the total sizes of semifree dg categories in a given diagram. Speciﬁcally, the ﬁniteness of
the number of generators is preserved under our homotopy colimit functor. In contrast, if we compare
this to the Grothendieck construction for homotopy colimit functor (refer to [GPS18]), our method
oﬀers several computational advantages. The Grothendieck construction is not explicit; rather, it
is expressed as a localization of a non-semifree dg category, which often leads to computational
diﬃculties.

We note that the description in Theorem 2 still holds for the diagrams of the form A ←C →B,
where A and B are not necessarily semifree, if k has ﬂat dimension zero (e.g., if k is a ﬁeld). See
Remark 5.8 for details.

Moreover, the constructions in Theorems 1 and 2 can be adapted for scenarios where the input
of the functors is a localization of a semifree dg category.
Please refer to Theorems 4.12, 4.13,
and 5.7 for more details on these adjustments. The usage of Theorem 5.7 is demonstrated through
computations for the case n = 2 in Sections 7.2 and 7.3.

As mentioned earlier, the constructed cylinder functor can deﬁne a “half” of a model structure, or
more precisely, a coﬁbration category structure on dgCats. This structure diﬀers from that induced
by [Tab05b] due to its simpler functorial factorization:

Theorem 3.

(1) (Theorem 6.2) The category of semifree dg categories dgCats is an I-category with the structure (cof, I), which is deﬁned as follows:
• cof : Coﬁbrations are semifree extensions.
• I: The functor I is the cylinder functor Cyl: dgCats →dgCats from Theorem 1.
(2) (Theorem 6.11) The category of semifree dg categories dgCats forms a coﬁbration category, where weak equivalences are quasi-equivalences and coﬁbrations are semifree extensions. Moreover, every object in dgCats is both ﬁbrant and coﬁbrant. In particular, dgCats
makes a category of coﬁbrant objects.

Theorems 1–3 hold in the setting of semifree dg algebras with slight modiﬁcations, see Remarks
4.11 and 5.6, and Theorem 6.13.

We note that Theorem 3 is not necessary for constructions such as Theorem 2, as we can use
our cylinder functor within Tabuada’s model structures to carry out homotopy theory. However,
we have established additional axioms for the constructed cylinder functor to explicitly construct a
coﬁbration category of semifree dg categories, which is distinct from the approach in [Tab05b]. In
[Tab05b], Quillen’s small object argument is employed to construct a model structure and establish
the existence of factorizations, but they are not eﬀective or concretely expressible in practice. In
contrast, our cylinder functor is explicit and computable, allowing for explicit and computable
constructions within the coﬁbration category dgCats. Consequently, we can derive Theorem 2 from
Theorem 3 without relying on [Tab05b].

Moreover, we can directly restrict the coﬁbration category structure given in Theorem 3 to dg
categories of strictly ﬁnite type, i.e., semifree dg categories with ﬁnitely many objects and generating
morphisms, as the factorizations preserve ﬁniteness. See Remark 6.14. The same is not true for
the model structure of Tabuada [Tab05b], as its given factorizations do not preserve ﬁniteness. We
note that dg categories of strictly ﬁnite type are, in particular, of ﬁnite type. Refer to [Kel07] and
[Kon09] for the deﬁnition of dg categories of ﬁnite type.

4
DOGANCAN KARABAS AND SANGJIN LEE

One can ﬁnd a direct application of the above results in symplectic topology.

Symplectic manifolds are associated with a powerful symplectic invariant known as the Fukaya
category, deﬁned as an A∞-category in [FOOO09a, FOOO09b]. Computing the Fukaya category is
typically a notoriously challenging task. However, Ganatra, Pardon, and Shende [GPS20, GPS18]
proved that for certain open symplectic manifolds (referred as Weinstein manifolds), the (wrapped)
Fukaya category can be computed, after a choice of a covering, as a homotopy colimit of the wrapped
Fukaya categories of the covering elements.

Furthermore, given the relation of wrapped Fukaya categories to microlocal sheaves by [GPS24],
Nadler [Nad17, Nad15] has shown that the wrapped Fukaya category of each covering element can
be regarded as a semifree dg category with ﬁnitely many objects and generating morphisms (a dg
category of strictly ﬁnite type) up to pretriangulated (or Morita) equivalence. This result highlights
the signiﬁcance of Theorem 2 in facilitating such computations.

Now, let us consider a symplectomorphism φ : W1 →W2 between two symplectic manifolds.
It is known that φ induces a functor between Fukaya categories of W1 and W2. If φ respects the
homotopy colimit diagrams that compute Fukaya categories of W1 and W2, one can obtain a speciﬁc
description of the induced functor using Theorem 2. For a detailed example and the symplectic
topological motivations behind this discussion, please refer to Section 7.

1.1. Acknowledgment. We are grateful to Ezra Getzler for his insightful comments, which inspired
the content of Section 6.

The ﬁrst-named author is supported by World Premier International Research Center Initiative (WPI), MEXT, Japan. The second-named author is supported by a KIAS Individual Grant
(MG094401) at Korea Institute for Advanced Study.

2. Preliminaries on dg categories

A diﬀerential graded (dg) category is a category enriched over the symmetric monoidal category
of complexes over a ﬁxed commutative ring k. It can also be viewed as an A∞-category in which
compositions of order greater than 2 are set to vanish. For further details on dg categories, readers
may refer to [Kel07], and for a review of A∞-categories, one can consult [Sei08]. We use d for the
diﬀerential and ◦for compositions of morphisms, and we omit the latter whenever it is convenient.
When introducing a dg category, we follow the convention of providing the following ﬁve items:

(i) Objects: We list the objects in the category.
(ii) Generating morphisms: We give a set of generating morphisms. They generate all the morphisms as an algebra, not as a module. We will not explicitly mention the existence of identity
morphisms, but it should be understood that every object has the identity endomorphism.
(iii) Degrees: For each generating morphism, we specify its degree.
(iv) Diﬀerentials: For each generating morphism, we specify its diﬀerential.
(v) Relations: We specify the relations between generating morphisms. This item will be omitted
if the generating morphisms freely generate all other morphisms.

Given a dg category C, we denote by Ob C (or simply C when it is clear from the context) the
collection of objects in C, and by Mor C the collection of morphisms in C. We use hom∗
C(A, B) (or
simply hom∗(A, B)) to represent the cochain complex of morphisms between the objects A and B
of C.

Next, we introduce a speciﬁc class of dg categories and dg functors that will be fundamental
throughout the paper. For further details, readers can refer to [KL21].

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
5

Deﬁnition 2.1.

(1) A (small) dg category C is called a semifree dg category if its morphisms, treated as an
algebra, are freely generated by a set of morphisms {fi} (indexed by an ordinal), with the
condition that dfi is generated by the set {fj | j < i}. In this case, {fi} is called a set of
generating morphisms of C.
(2) A dg functor F : C →D is called a semifree extension by a set of objects R and a set of
morphisms S = {fi} if it satisﬁes the following conditions:
• F is an inclusion.
• The objects of F(C), along with R, form the objects of D.
• The morphisms of D, treated as an algebra, can be expressed as a free extension of
the morphisms of F(C) by {fi} (indexed by an ordinal), with the condition that dfi is
generated by the morphisms of F(C) and {fj | j < i}.
(3) A dg category D is called a semifree extension of a dg category C by a set of objects R and
a set of morphisms S if there exists a semifree extension F : C →D as in Deﬁnition 2.1(2).

Let dgCat denote the category of small dg categories, where morphisms are dg functors. We aim
to invert certain dg functors, referred to as weak equivalences, in dgCat. The resulting categories
can be studied by introducing model structures on dgCat, making dgCat a model category.
See
[DS95], [Hov07], and [Hir03] for a review of model categories. More precisely, upon inverting weak
equivalences in dgCat, we obtain the homotopy category Ho(dgCat) of the model category dgCat.
Refer to Section 3 for further details.

For a given dg category C, we introduce the following notations:

• Tw C is the dg category of twisted complexes in C, which is a pretriangulated envelope of C.
• Perf C is the split-closure (or idempotent completion) of Tw C.

See [Sei08] for more details. With these notations, we can state the following theorem:

Theorem 2.2 ([Tab05b, Tab05a]). The category dgCat admits the following model structures:

(1) Dwyer-Kan model structure: Weak equivalences are dg functors that are quasi-equivalences,
and any dg category is a ﬁbrant object.
(2) Quasi-equiconic model structure: Weak equivalences are pretriangulated equivalences, which
are dg functors C →D that induce a quasi-equivalence Tw C →Tw D, and ﬁbrant objects
are pretriangulated dg categories.
(3) Morita model structure: Weak equivalences are Morita equivalences, which are dg functors
C →D that induce a quasi-equivalence Perf C →Perf D, and ﬁbrant objects are idempotent
complete pretriangulated dg categories.

All three model structures have the same coﬁbrations, which are retracts of semifree extensions.
Consequently, they also share the same coﬁbrant objects, which are retracts of semifree dg categories.

Remark 2.3. Any quasi-equivalence is a pretriangulated equivalence, and any pretriangulated
equivalence is a Morita equivalence.

It is known that any morphism C →D in the homotopy category Ho(M) of a model category M
can be seen as a chain of objects and morphisms in M

C
∼
←−C′ →D′
∼
←−D

for some coﬁbrant object C′ and ﬁbrant object D′, and arrows
∼
←−are weak equivalences. Consequently, we can characterize the morphisms in Ho(dgCat) through the following proposition:

6
DOGANCAN KARABAS AND SANGJIN LEE

Proposition 2.4. For given dg categories C and D, a morphism C →D in Ho(dgCat) can be
characterized as a chain of dg categories and dg functors in the following ways:

(1) C
∼
←−C′ →D, if dgCat is equipped with the Dwyer-Kan model structure.
(2) C
∼
←−C′ →Tw D, if dgCat is equipped with the quasi-equiconic model structure.
(3) C
∼
←−C′ →Perf D, if dgCat is equipped with the Morita model structure.

Here, C′ is a coﬁbrant dg category, and
∼
←−is a weak equivalence in the corresponding model structure.

Remark 2.5. If C is a coﬁbrant dg category, then we can replace each C
∼
←−C′ with C in Proposition
2.4.

Next, we introduce three distinct types of equivalency between dg categories, characterized by
becoming isomorphic in the corresponding homotopy category Ho(dgCat):

Deﬁnition 2.6. Let C and D be dg categories.

(1) C and D are quasi-equivalent if there is a chain of dg categories and dg functors

C
∼
←−C′ ∼
−→D

for some dg category C′, where each dg functor in the chain is a quasi-equivalence.
(2) C and D are pretriangulated equivalent if Tw C and Tw D are quasi-equivalent.
(3) C and D are Morita equivalent if Perf C and Perf D are quasi-equivalent.

As a result, we have two distinct types of generations for a dg category, deﬁned as follows:

Deﬁnition 2.7. Let C be a dg category. Let {Li} be a collection of objects in C. We say

(1) {Li} generates C if the full dg subcategory of C with the objects {Li} is pretriangulated
equivalent to C,
(2) {Li} split-generates C if the full dg subcategory of C with the objects {Li} is Morita equivalent
to C.

When C is a dg category, and S is a subset of closed degree zero morphisms in C, there exists a
dg category C[S−1], known as the dg localization of C at the morphisms in S. This localization is
essentially obtained from C by inverting morphisms in S. For a precise deﬁnition, one can refer to
sources such as [To¨e11] or [KL21]. The dg localization is unique up to quasi-equivalence, and its
existence is established in [To¨e07].

In the case where C is a semifree dg category, we can explicitly describe C[S−1]:

Proposition 2.8 ([KL21]). When C is a semifree dg category, and S = {gi : Ai →Bi} is a subset of
closed degree zero morphisms in C, the dg localization C[S−1] can be viewed as the semifree extension
of C by the morphisms g′
i, ˆgi, ˇgi, ¯gi

¯gi

gi

ˇgi

Ai
Bi
ˆgi

g′
i

for each i, with the gradings

|g′
i| = 0,
|ˆgi| = |ˇgi| = −1,
|¯gi| = −2,

and with the diﬀerentials

dg′
i = 0,
dˆgi = 1Ai −g′
igi,
dˇgi = 1Bi −gig′
i,
d¯gi = giˆgi −ˇgigi.

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
7

Next, we explore the colimit of diagrams of dg categories:

Proposition 2.9 ([KL21]). Let G: A →C be a dg functor, and F : A →B be a semifree extension
by a set of objects R and a set of morphisms S. Then, there exists a pushout (colimit) square

¯G

B
D

F

¯F

G

A
C

where ¯F : C →D is a semifree extension by the set of objects R and a set of morphisms
¯S := { ¯f : ¯G(A) →¯G(B) | f : A →B in S}

with | ¯f| := |f| and d ¯f := ¯G(df), and

¯G(A) :=

(
G(f)
if f ∈Mor A
¯f
if f ∈S
.

(
G(A)
if A ∈Ob A
A
if A ∈R
,
and
¯G(f) :=

Remark 2.10. In a more casual sense, the colimit D in Proposition 2.9 can be thought as B ∐C
after the identiﬁcation of the images of F with the images of G.

Remark 2.11. Given a semifree dg category C with a set of generating morphisms {fi}, consider
a morphism f : A →B in {fi} where A ̸= B. The dg localization C[f −1] can be given by C with
the identiﬁcations A = B and f = 1A=B. This description relies on a description of dg localization
through a colimit diagram, as presented in [To¨e07], and Proposition 2.9.

Finally, we present two propositions from [Che02] and [EN15], which can be thought as “basis
change” and “cancellation of generators” for the morphisms of semifree dg categories, respectively.
They are useful when we simplify a given semifree dg category.

Proposition 2.12. Let C be a semifree dg category with a set of generating morphisms {fi} (indexed
by an ordinal). Deﬁne the morphisms
˜fi := uifi + gi
where ui is a unit in the coeﬃcient ring k, and gi is a morphism in C generated by the set {fj | j < i}.
Then, the set { ˜fi} also generates the morphisms in C semifreely.

Proposition 2.13. Let C be a semifree dg category, and D be the semifree extension of C by the
morphisms {ai, bi} such that dai = bi for all i. Then, C and D are quasi-equivalent.

In the setting of Proposition 2.13, we say D is obtained from C by stabilization, and C is obtained
from D by destabilization.

3. Homotopy colimit functor using model structures

This section is a review of the homotopy colimit functor and related model-theoretical results.
Our main reference is [DS95].

For a given model category M (more generally, a category M with weak equivalences), we write
Ho(M) for its homotopy category, which is the category

• whose objects are the same as the objects of M, and
• whose morphisms are generated by the morphisms of M and the formal inverses of the weak
equivalences.

8
DOGANCAN KARABAS AND SANGJIN LEE

It comes with the localization functor

l: M →Ho(M)

which is the identity on objects, and sending morphisms to themselves.

From now on, let J be the category given by

a ←c →b

where a, b, c are the objects and the arrows are the morphisms.

Deﬁnition 3.1. Let M be a category with weak equivalences, and MJ be the category of functors
J →M (J-diagrams in M) whose weak equivalences are the objectwise weak equivalences. The
homotopy colimit functor

hocolim: Ho(MJ) →Ho(M)

is deﬁned (up to natural equivalence) as the total left derived functor of the colimit functor

colim: MJ →M.

If M has a model structure, we have a more concrete way to express the homotopy colimit functor.
To describe it, we consider an induced model structure on MJ from the model structure on M:

Proposition 3.2 ([Hir03],[Dug08]). Let M be a model category.

(1) MJ has a model structure, called a Reedy model structure, whose coﬁbrant objects are the
diagrams of the form

A
α
←−C
β−→B

where A, B, C are coﬁbrant objects in M, and β is a coﬁbration.
(2) If MJ is equipped with the Reedy model structure above, then colim: MJ →M preserves
coﬁbrations and acyclic coﬁbrations.

Before describing the homotopy colimit functor using model structures, we need to deﬁne coﬁbrant
resolution functors:

Proposition 3.3 ([DS95]). Let M be a model category. For any X ∈M, there exists a coﬁbrant
object Q(X) ∈M and an acyclic ﬁbration pX : Q(X) →X by model category axioms. Then for any
morphism f : X →Y , there exists a unique morphism ˜f : Q(X) →Q(Y ) up to right homotopy such
that the diagram

˜f

Q(X)
Q(Y )

pX
pY

f

X
Y

commutes.

Deﬁnition 3.4. Let M be a model category.

(1) We deﬁne Mc as the full subcategory of M consisting of coﬁbrant objects, and πMc as
the category with the same objects as Mc whose morphisms are right homotopy classes of
morphisms in Mc.

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
9

(2) For any X ∈M, there exists a coﬁbrant object Q(X) ∈M and an acyclic ﬁbration
pX : Q(X) →X by model category axioms. Then we deﬁne a coﬁbrant resolution functor as

Q: M →πMc
X 7→Q(X)
on objects

f 7→[ ˜f]
on morphisms

where given f : X →Y , the morphism ˜f is determined by Proposition 3.3, and [ ˜f] is the
right homotopy class of ˜f. The well-deﬁnedness of this functor follows from Proposition 3.3.

Now, we present the alternative description of the homotopy colimit functor:

Theorem 3.5 ([DS95]). Let M be a model category, and equip MJ with the Reedy model structure
induced by M as in Proposition 3.2. The unique lift of the composition

MJ
Q
−→π(MJ)c
l ◦colim
−−−−−→Ho(M)

gives the homotopy colimit functor

hocolim: Ho(MJ) →Ho(M)

where Q is a coﬁbrant resolution functor.

Remark 3.6.

(1) Although the colimit functor is not well-deﬁned on π(MJ)c, the functor l ◦colim above is
well-deﬁned since it identiﬁes right homotopic maps between coﬁbrant objects. This follows
from the second item in Proposition 3.2. See [DS95] for the details.
(2) By the lift, we mean that the triangle

l ◦colim ◦Q

MJ
Ho(M)

l

hocolim

Ho(MJ)

commutes. The existence and uniqueness follow from the fact that l ◦colim ◦Q sends weak
equivalences to isomorphisms, which again follows from the second item in Proposition 3.2.

Before presenting a corollary of Theorem 3.5, we describe a functor transforming a given diagram
to a more manageable one:

Proposition 3.7.

(1) There exists a functor T : MJ →MJ such that
• T sends an object of the form

X := (A
α
←−C
β−→B)

to the object

T(X) := (A ∐B
α∐β
←−−−C ∐C
∇C
−−→C)

where ∇C is the codiagonal for C, and

10
DOGANCAN KARABAS AND SANGJIN LEE

• T sends a morphism of the form

α
β

X

FA

FC
FB

F
:=
A
C
B

α′
β′

X′

A′
C′
B′

to the morphism

α∐β
∇C

T(X)

FA∐FB

FC∐FC
FC

T(F )
:=
A ∐B
C ∐C
C

α′∐β′
∇C′
.

A′ ∐B′
C′ ∐C′
C′

T(X′)

(2) The colimit functor satisﬁes

colim ◦T = colim.

(3) T sends weak equivalences to weak equivalences, hence it induces the functor

Ho(T): Ho(MJ) →Ho(MJ)

satisfying

hocolim ◦Ho(T) = hocolim.

Proof. It is straightforward to check.
□

Using Proposition 3.7, we have a slight improvement of Theorem 3.5 for constructing the homotopy
colimit functor, which only requires us to construct a coﬁbrant resolution functor Q for the image
of the functor T. This will be useful in Section 5.

Corollary 3.8. Let M be a model category, and equip MJ with the Reedy model structure induced
by M as in Proposition 3.2. The unique lift of the composition

MJ
T−→MJ
Q
−→π(MJ)c
l ◦colim
−−−−−→Ho(M)

gives the homotopy colimit functor

hocolim: Ho(MJ) →Ho(M)

where T is the functor described in Proposition 3.7, and Q is a coﬁbrant resolution functor.

Proof. The statement follows from the commutativity of the diagram

Q

T

MJ
MJ
π(MJ)c

l
l

l ◦colim

Ho(T)

hocolim

Ho(MJ)
Ho(MJ)
Ho(M)

hocolim

by Theorem 3.5 and Proposition 3.7.
□

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
11

4. Cylinder functor for the category of dg categories

Our goal is to give a formula for the homotopy colimit functor hocolim: Ho(MJ) →Ho(M)
for the case M = dgCat, where dgCat is the model category of small k-linear dg categories for a
commutative ring k with Dwyer-Kan, quasi-equiconic, or Morita model structure. Their weak equivalences are quasi-equivalences, pretriangulated equivalences, and Morita equivalences, respectively
(see Theorem 2.2). Our formula will work for all these model structures. To express the formula,
ﬁrst we need to describe a coﬁbrant resolution functor, which will be done in Section 5.

Here, we will deﬁne a cylinder functor for the semifree dg categories in dgCat, in other words,
we will give a functorial construction of cylinder objects for semifree dg categories. We will use this
construction to describe a coﬁbrant resolution functor in Section 5. Also, in Section 6, the cylinder
functor will play a pivotal role in deﬁning an I-category and a coﬁbration category of semifree dg
categories.

We ﬁrst recall the deﬁnition of a cylinder functor:

Deﬁnition 4.1. Let M be a model category. A functor I : M →M is called a cylinder functor, if
there exists a coﬁbration
iC : C ∐C →I(C)

and an acyclic ﬁbration
pC : I(C) →C

such that pC ◦iC = ∇C (the codiagonal of C) for all C ∈M (in other words, I(C) is a cylinder object
for C), and the diagram

F ∐F

C ∐C
D ∐D

iC
iD
I(F )

(4.1)

I(C)
I(D)

pC
pD

F

C
D

commutes for any morphism F : C →D in M.

From now on, we focus on M = dgCat with Dwyer-Kan model structure. Everything here still
holds if we work with quasi-equiconic or Morita model structure. Before deﬁning a cylinder functor,
recall that [KL21] deﬁned a cylinder object Cyl(C) for any semifree dg category C:

Deﬁnition 4.2 ([KL21]). Let C be a semifree dg category, and i1, i2 : C →C ∐C be the inclusions
to the ﬁrst and second copies, respectively. We deﬁne Cyl(C) as the semifree extension of C ∐C by
the morphisms comprised of

• the morphisms tC, t′
C, ˆtC, ˇtC, ¯tC

¯tC

tC

ˇtC

i1(C)
i2(C)
ˆtC

t′
C

for each C ∈C, with the gradings

|tC| = |t′
C| = 0,
|ˆtC| = |ˇtC| = −1,
|¯tC| = −2,

12
DOGANCAN KARABAS AND SANGJIN LEE

and with the diﬀerentials

dtC = dt′
C = 0,
dˆtC = 1i1(C) −t′
C ◦tC,
dˇtC = 1i2(C) −tC ◦t′
C,
d¯tC = tC ◦ˆtC −ˇtC ◦tC,

• degree |f| −1 morphism tf : i1(A) →i2(B) for each generating morphism f ∈hom∗
C(A, B),
with the diﬀerential

dtf = (−1)|f|(i2(f) ◦tA −tB ◦i1(f)) + correction term

where the correction term is 0 if df = 0.

If df ̸= 0, the correction term is given as follows: If

m
X

df = c1A +

i=1
cifi,ni ◦. . . ◦fi,j ◦. . . ◦fi,1

where fi,j are generating morphisms of C, and c, ci ∈k (c = 0 if A ̸= B), then

ni
X

m
X

correction term =

j=1
(−1)|fi,j−1|+...+|fi,1|i2(fi,ni) . . . i2(fi,j+1) ◦tfi,j ◦i1(fi,j−1) . . . i1(fi,1).

i=1
ci

Remark 4.3. The semifree dg category Cyl(C), associated with a given semifree dg category C,
is well-deﬁned up to isomorphism. In other words, it does not depend on the choice of generating
morphisms of C. This can be veriﬁed directly or by considering Remark 4.10.

Remark 4.4. By the description of the dg localization in Proposition 2.8, Cyl(C) is the dg localization Cyl0(C)[S−1] (up to quasi-equivalence), where Cyl0(C) is the semifree extension of C ∐C by
the morphisms comprised of

• closed degree zero morphism tC : i1(C) →i2(C) for each C ∈C,
• degree |f| −1 morphism tf : i1(A) →i2(B) for each generating morphism f ∈hom∗
C(A, B),
with the diﬀerential given in Deﬁnition 4.2,

and S = {tC | C ∈C}.

Theorem 4.5 ([KL21]). Let C be a semifree dg category. Cyl(C) deﬁned in Deﬁnition 4.2 is a
cylinder object for C. That is, the semifree extension

iC := i1 ∐i2 : C ∐C →Cyl(C)

is a coﬁbration, and the functor

pC : Cyl(C) →C

i1(C), i2(C) 7→C,
tC, t′
C 7→1C,
ˆtC, ˇtC, ¯tC 7→0
for each object C ∈C

i1(f), i2(f) 7→f,
tf 7→0
for each generating morphism f in C

is an acyclic ﬁbration, and they satisfy pC ◦iC = ∇C.

So, Theorem 4.5 builds the cylinder functor on objects of dgCat, but not on morphisms. The
functoriality of the cylinder object construction is not discussed in [KL21]. Here, we will make the
construction functorial. This is the main goal of this section. First, we need some deﬁnitions and
properties.

Given a semifree dg category C, Deﬁnition 4.2 deﬁnes Cyl(C) and for each generating morphism
f of C, it speciﬁes a generating morphism tf of Cyl(C). Here, we extend this and deﬁne a morphism
tθ in Cyl(C) for each morphism θ in C, which will be useful later:

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
13

Deﬁnition 4.6. Let C be a semifree dg category. Let θ ∈hom∗
C(A, B) be given by

m
X

θ = c1A +

i=1
cifi,ni ◦. . . ◦fi,j ◦. . . ◦fi,1

for some generating morphisms fi,j of C and c, ci ∈k (c = 0 if A ̸= B). We deﬁne degree |θ| −1
morphism tθ ∈hom∗
Cyl(C)(i1(A), i2(B)) by

m
X

ni
X

tθ :=

i=1
ci

j=1
(−1)|fi,j−1|+...+|fi,1|i2(fi,ni) ◦. . . ◦i2(fi,j+1) ◦tfi,j ◦i1(fi,j−1) ◦. . . ◦i1(fi,1).

Proposition 4.7. Let C be a semifree dg category, and let θ ∈hom∗
C(A, B). We have the identity

(4.2)
dtθ = (−1)|θ|(i2(θ) ◦tA −tB ◦i1(θ)) + tdθ.

Moreover, if

m
X

(4.3)
θ = c1A +

i=1
ciθi,ni ◦. . . ◦θi,j ◦. . . ◦θi,1

for some morphisms θi,j in C (not necessarily generating morphisms) and c, ci ∈k (c = 0 if A ̸= B),
we have the identity

ni
X

m
X

(4.4)
tθ =

j=1
(−1)|θi,j−1|+...+|θi,1|i2(θi,ni) ◦. . . ◦i2(θi,j+1) ◦tθi,j ◦i1(θi,j−1) ◦. . . ◦i1(θi,1).

i=1
ci

Proof. First, we remark that (4.2) holds by deﬁnition when θ is a generating morphism.
Now,
assume θ is as in Deﬁnition 4.6. Then the assignment θ 7→tθ is linear. Hence, it is enough to prove
(4.2) for θ = 1A and θ = fn ◦. . . ◦f1 for any generating morphisms fi of C. This is straightforward
to check using the remark in the beginning of the proof.

Now, assume θ is given as in (4.3). By the linearity of the assignment θ 7→tθ, we only need to
prove (4.4) for θ = θn ◦. . . ◦θ1 for any given morphisms θi. Furthermore, again by the linearity,
we can assume that each θi is given as a product of generating morphisms fj of C. Then, it is
straightforward to check that (4.4) holds.
□

We are now ready to state one of the main results of this section:

Theorem 4.8. Let F : C →D be a dg functor between semifree dg categories.

(1) There is a dg functor Cyl(F): Cyl(C) →Cyl(D) that is an extension of the dg functor

F ∐F : C ∐C →D ∐D

i.e.

Cyl(F): Cyl(C) →Cyl(D)

i1(C), i2(C) 7→i1(F(C)), i2(F(C))
respectively, for each C ∈C

i1(θ), i2(θ) 7→i1(F(θ)), i2(F(θ))
respectively, for each morphism θ in C

by additionally specifying

tC, t′
C, ˆtC, ˇtC, ¯tC 7→tF (C), t′
F (C), ˆtF (C), ˇtF (C), ¯tF (C)
respectively, for each object C ∈C

tf 7→tF (f)
for each generating morphism f in C

where tF (f) is deﬁned according to Deﬁnition 4.6.

14
DOGANCAN KARABAS AND SANGJIN LEE

(2) The dg functor Cyl(F) satisﬁes

(4.5)
Cyl(F)(tθ) = tF (θ)

for any morphism θ in C.
(3) The diagram

F ∐F

C ∐C
D ∐D

iC
iD
Cyl(F )

(4.6)

Cyl(C)
Cyl(D)

pC
pD

F

C
D

commutes.

Proof. We need to show that Cyl(F) is indeed a dg functor (it is already a functor by deﬁnition).
The only nontrivial part is showing that d(Cyl(F)(tf)) = Cyl(F)(dtf) for any generating morphism
f in C. To verify this, we need to prove the identity

Cyl(F)(tθ) = tF (θ)

for any morphism θ in C. Assume without loss of generality that θ is given by

θ = fn ◦. . . ◦f1

for some generating morphisms fj of C. Then, since Cyl(F) is a functor, we have








n
X

Cyl(F)(tθ) = Cyl(F)

j=1
(−1)|fj−1|+...+|f1|i2(fn) ◦. . . ◦i2(fj+1) ◦tfj ◦i1(fj−1) ◦. . . ◦i1(f1)

n
X

=

j=1
(−1)|fj−1|+...+|f1|i2(F(fn)) ◦. . . ◦i2(F(fj+1)) ◦tF (fj) ◦i1(F(fj−1)) ◦. . . ◦i1(F(f1))

= tF (θ)

by the identity (4.4) since F(θ) = F(fn) ◦. . . ◦F(f1).

Using this and the identity (4.2), for any generating morphism f : A →B in C, we see that

d(Cyl(F)(tf)) = dtF (f)

= (−1)|F (f)|(i2(F(f)) ◦tF (A) −tF (B) ◦i1(F(f))) + td(F (f))

= (−1)|f|(i2(F(f)) ◦tF (A) −tF (B) ◦i1(F(f))) + tF (df)

= Cyl(F)((−1)|f|(i2(f) ◦tA −tB ◦i1(f)) + tdf)

= Cyl(F)(dtf)

which shows that Cyl(F) is a dg functor.

Finally, the commutation of the diagram (4.6) is obvious from the construction.
□

Theorem 4.8 suﬃces for the purpose of deﬁning the homotopy colimit functor in Section 5. However, we proceed to establish a cylinder functor for semifree dg categories in Corollary 4.9. This
construction plays a key role in deﬁning an I-category (a category with a natural cylinder functor),
thus giving rise to a coﬁbration category of semifree dg categories in Section 6.

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
15

Corollary 4.9. The assignment

Cyl: dgCats →dgCats
C 7→Cyl(C)

F 7→Cyl(F)

is a cylinder functor for dgCats, where dgCats is the full subcategory of the model category of
dg categories dgCat (with Dwyer-Kan, quasi-equiconic, or Morita model structure) consisting of
semifree dg categories, and Cyl(C) is as deﬁned in Deﬁnition 4.2, and Cyl(F) is as deﬁned in
Theorem 4.8.

Proof. Clearly, Cyl(1C) = 1Cyl(C) where 1C : C →C is the identity functor on C. Also, the equality
Cyl(G ◦F) = Cyl(G) ◦Cyl(F) directly follows from the deﬁnition of Cyl(F) and the identity (4.5),
hence Cyl is a functor. Then, the commutative diagram (4.6) shows that Cyl is indeed a cylinder
functor in the sense of Deﬁnition 4.1.
□

Remark 4.10. To establish the well-deﬁnedness of the functor Cyl up to natural isomorphism, let
us denote the application of Cyl to a semifree dg category C with a predetermined set of generating
morphisms {fi} by Cyl(C, {fi}). By applying Cyl to the identity functor 1C : C →C, we get an
isomorphism
Cyl(1C): Cyl(C, {fi}) ∼
−→Cyl(C, {gi})

for any set of generating morphisms {gi} of C, which follows from the fact that Cyl is a functor.
Hence, Cyl(C) is well-deﬁned up to isomorphism. Lastly, if there is a functor F : C →D between
semifree dg categories, it induces a commutative square

Cyl(C, {fi})
Cyl(C, {gi})

Cyl(1C)
∼

.

Cyl(F )
Cyl(F )

Cyl(D, {f ′
i})
Cyl(D, {g′
i})

Cyl(1D)
∼

Here, {f ′
i} and {g′
i} are two arbitrary sets of generating morphisms of D. Hence, Cyl is well-deﬁned
up to natural isomorphism.

Remark 4.11. The cylinder object and functor can be also deﬁned for the model category of dg
algebras. In that case, for a given semifree dg algebra C, C ∐C is the semifree dg algebra (with the
same unique object as C by deﬁnition) whose generating morphisms are doubled, and i1, i2 : C →C∐C
are the obvious inclusions. Cyl(C) is deﬁned as the semifree extension of C ∐C by the morphisms tf,
where f is a generating morphism of C (no tC, t′
C, ˆtC, ˇtC, ¯tC are used for C ∈C). All the formulas
still hold by setting
tC = 1,
t′
C = 1,
ˆtC = 0,
ˇtC = 0,
¯tC = 0.

Given a morphism F : C →D between semifree dg algebras, Cyl(F) is just deﬁned by

Cyl(F)(tf) = tF (f)

for every generating morphism f in C. Hence, this deﬁnes a cylinder functor for the model category
of dg algebras as in Corollary 4.9. The proofs are the similar in the case of semifree dg algebras.

Finally, we want to discuss the cylinder object for the dg localization C[S−1], where C is a semifree
dg category and S is a subset of closed degree zero morphisms in C. We note that since C[S−1] can
be also seen as a semifree dg category, one can apply Theorem 4.5 and 4.8 to C[S−1]. But we would
like to discuss a simpler cylinder object for C[S−1] for later convenience.

16
DOGANCAN KARABAS AND SANGJIN LEE

By the description of dg localization in Proposition 2.8, we know that C[S−1] can be express as a
semifree extension of C by the morphisms

g′ : B →A,
ˆg: A →A,
ˇg : B →B,
¯g : A →B

for every g: A →B in S, whose gradings and diﬀerentials are given as in Proposition 2.8. Hence,
the generating morphisms of C[S−1] are the generating morphisms of C and the morphisms g′, ˆg, ˇg, ¯g
for each morphism g in S.

Recall that i1, i2 : C →C ∐C ֒→Cyl(C) are the inclusions to the ﬁrst and second copies, respectively. Then, it is easy to see that Cyl(C[S−1]) is the semifree extension of Cyl(C)[(i1(S) ⊔i2(S))−1]
by the morphisms

tg′ : i1(B) →i2(A),
tˆg : i1(A) →i2(A),
tˇg : i1(B) →i2(B),
t¯g : i1(A) →i2(B)

for every g: A →B in S.

Note that Cyl(C)[(i1(S)⊔i2(S))−1] is the semifree extension of C[S−1]∐C[S−1] by the morphisms

• tC, t′
C, ˆtC, ˇtC, ¯tC for each C ∈C, and
• tf for each generating morphism f in C.

In [KL21], it is shown that one can choose a cylinder object for C[S−1] that is simpler than
Cyl(C[S−1]) in the sense that it does not need the generating morphisms tg′, tˆg, tˇg, t¯g:

Theorem 4.12 ([KL21]). Let C be a semifree dg category, and S be a subset of closed degree zero
morphisms in C. Then Cyl(C)[(i1(S)⊔i2(S))−1] is a cylinder object for C[S−1]. That is, the semifree
extension
iC[S−1] := i1 ∐i2 : C[S−1] ∐C[S−1] →Cyl(C)[(i1(S) ⊔i2(S))−1]
is a coﬁbration, and the functor

pC[S−1] : Cyl(C)[(i1(S) ⊔i2(S))−1] →C[S−1]

i1(C), i2(C) 7→C,
tC, t′
C 7→1C,
ˆtC, ˇtC, ¯tC 7→0
for each object C ∈C
tf 7→0
for each generating morphism f in C

i1(θ), i2(θ) 7→θ
for each morphism θ in C[S−1]

is an acyclic ﬁbration, and they satisfy pC[S−1] ◦iC[S−1] = ∇C[S−1].

Let C and D be a semifree dg categories, and SC and SD be subsets of closed degree zero morphisms
in C and D, respectively. For a given functor F : C[S−1
C ] →D[S−1
D ], Theorem 4.8 constructs the dg
functor
Cyl(F): Cyl(C[S−1
C ]) →Cyl(D[S−1
D ]).
However, in such cases, i.e., when a dg category is given as a dg localization, we want to work
with its simpler cylinder object given by Theorem 4.12. Hence, we conclude this section with a
construction of a dg functor between the simpler cylinder objects:

Theorem 4.13. Let C and D be a semifree dg categories, and SC and SD be subsets of closed degree
zero morphisms in C and D, respectively. Let F : C[S−1
C ] →D[S−1
D ] be a dg functor.

(1) There is a dg functor

Cyl(F): Cyl(C)[(i1(SC) ⊔i2(SC))−1] →Cyl(D)[(i1(SD) ⊔i2(SD))−1]

between cylinder objects of C[S−1
C ] and D[S−1
D ] given by Theorem 4.12, which is an extension
of the dg functor

F ∐F : C[S−1
C ] ∐C[S−1
C ] →D[S−1
D ] ∐D[S−1
D ]

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
17

by additionally specifying

tC, t′
C, ˆtC, ˇtC, ¯tC 7→tF (C), t′
F (C), ˆtF (C), ˇtF (C), ¯tF (C)
respectively, for each object C ∈C

tf 7→tF (f)
for each generating morphism f in C

where tF (f) is deﬁned according to Deﬁnition 4.6, and for any g: A →B in SD, we deﬁne
the morphisms

tg′ := −i2(g′) ◦tg ◦i1(g′) −i2(ˆg) ◦tA ◦i1(g′) + i2(g′) ◦tB ◦i1(ˇg) + i2(ˆg ◦g′ −g′ ◦ˇg) ◦tB
(4.7)

tˆg := i2(g′) ◦tg ◦i1(ˆg) + i2(ˆg) ◦tA ◦i1(ˆg) + i2(g′) ◦tB ◦i1(¯g) −i2(ˆg ◦ˆg + g′ ◦¯g) ◦tA
−i2(ˆg ◦g′ −g′ ◦ˇg) ◦tg
tˇg := −i2(ˇg) ◦tg ◦i1(g′) + i2(ˇg) ◦tB ◦i1(ˇg) + i2(¯g) ◦tA ◦i1(g′) −i2(ˇg ◦ˇg + ¯g ◦g′) ◦tB
t¯g := −i2(ˇg) ◦tg ◦i1(ˆg) −i2(ˇg) ◦tB ◦i1(¯g) + i2(¯g) ◦tA ◦i1(ˆg) + i2(ˇg ◦¯g −¯g ◦ˆg) ◦tA
−i2(ˇg ◦ˇg + ¯g ◦g′) ◦tg

in Cyl(D)[(i1(SD) ⊔i2(SD))−1].
(2) The dg functor Cyl(F) satisﬁes

Cyl(F)(tθ) = tF (θ)

for any morphism θ in C[S−1
C ].
(3) The diagram

F ∐F

C[S−1
C ] ∐C[S−1
C ]
D[S−1
D ] ∐D[S−1
D ]

iC[S−1
C
]
iD[S−1
D ]

Cyl(F )

Cyl(C)[(i1(SC) ⊔i2(SC))−1]
Cyl(D)[(i1(SD) ⊔i2(SD))−1]

pC[S−1
C
]
pD[S−1
D ]

F

C[S−1
C ]
D[S−1
D ]

commutes.

Proof. First, it is straightforward (although tedious) to check that (4.2) holds for tg′, tˆg, tˇg, t¯g for
every morphism g in SD (and SC). Then, one can similarly prove Proposition 4.7 for the morphisms
in D[S−1
D ] (and C[S−1
C ]). Hence, the ﬁrst two items of Theorem 4.13 can be proven similar to Theorem
4.8. The last item, the commutation of the diagram, is straightforward to check.
□

5. Homotopy colimit functor on the diagrams of semifree dg categories

Our goal in this section is to construct the homotopy colimit functor Ho(dgCatJ) →Ho(dgCat),
where dgCat is the model category of dg categories with Dwyer-Kan, quasi-equiconic, or Morita
model structure (see Theorem 2.2), and J is a category given as follows:

a ←c →b

Note that we equip dgCatJ with the Reedy model structure as in Proposition 3.2.

To achieve our goal, we will explicitly construct the homotopy colimit functor on (dgCats)J in
Theorem 5.3 and Theorem 5.7, where dgCats is the full subcategory of dgCat consisting of semifree

18
DOGANCAN KARABAS AND SANGJIN LEE

dg categories. The category (dgCats)J can be seen as a subcategory of Ho(dgCatJ), and there is a
way to lift the homotopy colimit functor on (dgCats)J to Ho(dgCatJ) via the commuting diagram

j

Q

T

(dgCats)J
dgCatJ
dgCatJ
π(dgCatJ)c

l◦j

l

l ◦colim

hocolim

Ho(dgCatJ)
Ho(dgCat)

where j is the inclusion functor, T is the functor given in Proposition 3.7, and Q is a coﬁbrant
resolution functor, which we will give in Lemma 5.1.
The reason is as follows: hocolim (up to
natural equivalence) is the unique lift of the composition l ◦colim ◦Q ◦T by Corollary 3.8, and each
object of dgCatJ is weakly equivalent to an object of (dgCats)J.

Therefore, we only need to describe a coﬁbrant resolution functor Q on the image of the functor
T ◦j:

Lemma 5.1. There is a coﬁbrant resolution functor Q: (dgCat)J →π(dgCatJ)c satisfying the
following:

(1) Q sends an object of the form

X := (A ∐B
α∐β
←−−−C ∐C
∇C
−−→C)

where A, B, C are semifree dg categories and α, β are dg functors, to the object

Q(X) := (A ∐B
α∐β
←−−−C ∐C
iC
−→Cyl(C))

where Cyl(C) is deﬁned as in Deﬁnition 4.2 (along with the functors iC and pC given in
Theorem 4.5).
(2) Q sends a morphism of the form

α∐β
∇C

X

FA∐FB

FC∐FC
FC

F
:=
A ∐B
C ∐C
C

α′∐β′
∇C′

X′

A′ ∐B′
C′ ∐C′
C′

where FA, FB, FC are dg functors, to the morphism

α∐β
iC

Q(X)

A ∐B
C ∐C
Cyl(C)

FA∐FB

Q(F )
:=

FC∐FC
Cyl(FC)

α′∐β′
iC′

Q(X′)

A′ ∐B′
C′ ∐C′
Cyl(C′)

where Cyl(FC) is deﬁned as in Theorem 4.8.

Proof. For every object X given in the lemma, Q(X) is a coﬁbrant object by Proposition 3.2 and
Theorem 4.5. Moreover, consider the morphism pX : Q(X) →X given by

α∐β
iC

Q(X)

1A∐1B

1C∐1C
pC

pX
:=
A ∐B
C ∐C
Cyl(C)

α∐β
∇C

X

A ∐B
C ∐C
C

A COMPUTATIONAL APPROACH TO THE HOMOTOPY THEORY OF DG CATEGORIES
19

which is indeed a morphism and an acyclic (Reedy) ﬁbration (see [Hir03]) since pC ◦iC = ∇C and
pC is an acyclic ﬁbration by Theorem 4.5. Then, the diagram

Q(F )

Q(X)
Q(X′)

pX
pX′

F

X
X′

commutes for every morphism F : X →X′ given in the lemma since the diagram (4.6) commutes
by Theorem 4.8. Note that Q(F) is indeed a morphisms also because the diagram (4.6) commutes.
Therefore, by Proposition 3.3 and Deﬁnition 3.4, there is a coﬁbration functor Q with the properties
given in the lemma.
□

Remark 5.2. Assume that in Lemma 5.1, we replace C and C′ by C[S−1
C ] and C′[S−1
C′ ] for some
subsets of degree zero closed morphisms SC and SC′ in C and C′, respectively. Then, instead of using
Cyl(FC[S−1
C
]): Cyl
 
C[S−1
C ]

→Cyl(C′[S−1
C′ ]) when deﬁning Q, we can use the dg functor

Cyl(C)[(i1(SC) ⊔i2(SC))−1] →Cyl(C′)[(i1(SC′) ⊔i2(SC′))−1]

given in Theorem 4.13, which gives a simpler coﬁbrant resolution functor Q. The proof will be the
same as the proof of Lemma 5.1 after replacing Theorem 4.8 with Theorem 4.13 in the proof.

Finally, we can state the formula for the homotopy colimit functor on the diagrams of semifree
dg categories:

Theorem 5.3. The homotopy colimit functor hocolim: Ho(dgCatJ) →Ho(dgCat) (up to natural
equivalence) satisﬁes the following:

(1) hocolim sends an object of the form

X := (A
α
←−C
β−→B)

where A, B, C are semifree dg categories and α, β are dg functors, to the object (semifree dg
category)

hocolim(X)

which is the semifree extension of A ∐B by
• closed degree zero morphism tC : α(C) →β(C) for each object C ∈C,
• morphisms t′
C, ˆtC, ˇtC, ¯tC for each object C ∈C as in Deﬁnition 4.2 (after replacing i1
with α and i2 with β),
• degree |f| −1 morphism tf : α(A) →β(B) for each generating morphism f : A →B in
C whose diﬀerential is given as in Deﬁnition 4.2 (after replacing i1 with α and i2 with
β).
(2) hocolim sends a morphism of the form

α
β

X

FA

FC
FB

F
:=
A
C
B

α′
β′

A′
C′
B′

X′

20
DOGANCAN KARABAS AND SANGJIN LEE

where FA, FB, FC are dg functors, to the morphism (dg functor)

hocolim(F): hocolim(X) →hocolim(X′)

A 7→FA(A)
for any object A ∈A

B 7→FB(B)
for any object B ∈B

a 7→FA(a)
for any morphism a in A

b 7→FB(b)
for any morphism b in B

tC, t′
C, ˆtC, ˇtC, ¯tC 7→tFC(C), t′
FC(C), ˆtFC(C), ˇtFC(C), ¯tFC(C)
respectively, for any object C ∈C

tf 7→tFC(f)
for any generating morphism f in C,

where for any generating morphism f : A →B in C, the degree |f| −1 morphism tFC(f) is
deﬁned as

m
X

ni
X

tFC(f) :=

i=1
ci

j=1
(−1)|fi,j−1|+...+|fi,1|β′(fi,ni) ◦. . . ◦β′(fi,j+1) ◦tfi,j ◦α′(fi,j−1) ◦. . . ◦α′(fi,1)

if FC(f) is given by

m
X

FC(f) = c1FC(A) +

i=1
cifi,ni ◦. . . ◦fi,j ◦. . . ◦fi,1 ∈hom∗
C′(FC(A), FC(B))

for some generating morphisms fi,j of C′ and c, ci ∈k (c = 0 if FC(A) ̸= FC(B)).

Proof. First, we note that the description of hocolim(X) is given in [KL21]. We can also see it here
as follows: By Corollary 3.8, we have

hocolim(X) = l ◦colim ◦Q ◦T(X)

where T is a functor as in Proposition 3.7, Q is the coﬁbration functor given in Lemma 5.1. Using
the description of colim given in Proposition 2.9, it is straightforward to check that hocolim(X) is
as described in the ﬁrst item.

For the second item (which does not appear in [KL21]), by Corollary 3.8 again, we have

hocolim(F) = l ◦colim ◦Q ◦T(F).

Then, considering that hocolim(X) is the semifree extension of A ∐B by the morphims given in
the theorem, we see that hocolim(F): hocolim(X) →hocolim(X′) acts like FA on A, FB on B, and
Cyl(FC) (which is given in Theorem 4.8) on the added morphisms.
□

Remark 5.4. In Theorem 5.3, we could have expressed hocolim(X) as the semifree dg category
obtained by ﬁrst taking the semifree extension of A ∐B by the morphisms tC and tf for each
C ∈C and for each generating morphism f in C, and then taking the dg localization of the resulting
category at the morphisms {tC | C ∈C} as in Remark 4.4. Hence, up to natural equivalence, the
images of the morphisms t′
C, ˆtC, ˇtC, ¯tC under the homotopy colimit functor are uniquely determined
for every C ∈C.

Remark 5.5. Theorem 5.3 implies that for any morphism θ in C, we have

hocolim(F)(tθ) = tFC(θ).

This follows from the identity (4.5).

---

*Source: IEEE Transactions on Software Engineering*
