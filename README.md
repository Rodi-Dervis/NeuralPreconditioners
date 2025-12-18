# NeuralPreconditioners
This is a project to develop a neural network which can find preconditioner matrices for solving linear systems of equations via the Preconditioned Conjugate Gradient method (PCG) under certain constraints.

## 1. Theoretical Setup / Motivation
### Disclaimer
As we progress through the setup/motivation, the problem becomes more and more mathematical. Depending on your maths background, you can feel free to stop whenever things start sounding like nonsense.  

**High-level summary:** We aim to make use of a neural network to help us solve linear systems of equations.

### Linear Systems of Equations
Given a matrix $$A \in \mathbb{R}^{\ n \times n}$$ and a vector $$b \in \mathbb{R}^n$$, we wish to find $$x \in \mathbb{R}^n$$ such that $$Ax = b$$.  
This is known as a linear system of equations, and these kinds of equations have unfathomably-many uses and applications across virtually any number of industries you can think of; thus, a lot of research has been done into the most efficient ways to solve these problems. In practice, it is usually nicer to try to approximate the solution (i.e we find $$x$$ so that $$Ax \approx b$$), which can be done via iterative methods. 

### Conjugate Gradient Method
A square matrix $$A \in \mathbb{R}^{\ n \times n}$$ is said to be *symmetric* if $$A = A^T$$, and *positive-definite* if, for all non-zero vectors $$y \in \mathbb{R}^n$$, we have $$y^TAy > 0$$. Matrices satisfying both properties are said to be *symmetric positive-definite* (SPD). 
While we won't go into specifics here, there is an iterative method known as the **Conjugate Gradient** method (CG) for which, if $$A$$ is SPD, we are guaranteed to find a vector $$x$$ such that $$Ax$$ approximates $$b$$ (and we can find such an $$x$$ for *any* positive, desired degree of accuracy). This is, of course, provided we have the patience to perform enough iterations.

Unfortunately, this method can be very slow to converge to a solution if the matrix $$A$$ is not 'nice'. Formally:  
Let $$\lambda_{min}$$ and $$\lambda_{max}$$ be the smallest and largest eigenvalues of $$A$$, and let $$\kappa(A) = \frac{\lambda_{max}}{\lambda_{min}}$$. We call $$\kappa$$ the *condition number* of $$A$$.  
Thus, quantifiably, $$A$$ is 'nice' if the condition number is relatively small. This is by no means guaranteed, and so CG may be slow to yield a solution.

### Preconditioned Conjugate Gradient Method
Suppose we could find a matrix $$M \in \mathbb{R}^{\ n \times n}$$ such that $$M$$ is SPD and $$M \approx A$$. Then $$M^{-1}A \approx A^{-1}A = I$$. Since $$I$$ has only eigenvalues of $$1$$, we have $$\kappa(I) = 1$$, which is the smallest possible condition number. It follows that $$M^{-1}A$$ should have a small condition number too, and now we have a system $$M^{-1}Ax = M^{-1}b$$ to which we can apply CG* and converge at a faster rate!  

*(You may have noticed that $$M^{-1}A$$ is not necessarily SPD in the standard Euclidean inner product, so CG is not guaranteed to converge. In actuality, we apply CG to the operator $$M^{-\frac{1}{2}} A M^{-\frac{1}{2}}$$, which *is* SPD and has the same spectrum as $$M^{-1}A$$, so everything works out fine!)  

This process, whereby we first apply a preconditioner matrix $$M$$ to the system and then perform CG, is (aptly) known as the **Preconditioned Conjugate Gradient** method (PCG). 

### How Does a Neural Network Help?
You came to read about neural networks and cool AI stuff and I just made you sit through a wall of text about linear algebra and iterative methods. I promise, this is only *partly* because of my agenda and sworn duty as a maths guy to practice a form of mathematical evangelism; mostly, I think the problem we are trying to solve should be clear.

Currently, a neural network cannot outperform classical iterative methods end-to-end. In fact, given a total arbitrary matrix $$A$$, it seems unlikely that a neural network will consistently outperform traditional methods. However, neural networks are great at exploiting hidden structure; for example, neural preconditioners have been shown to outperform traditional ones when the problem matrices $$A$$ share similar underlying structures; for example, when these matrices arise from partial differential equations. In particular, we will be training our NN to provide preconditioners when the matrix $$A$$ falls out of the (discretised) variable-coefficient diffusion equation.

### Summary
In summary, if we expend the significant, initial cost of training a neural network to give us suitable preconditioner matrices, we can then call on it as needed prior to performing PCG and gradually save time in the macro sense. **This is the objective of the project.**

Will this particular neural network ultimately live up to these high expectations? Given my resources, both computationally and intellectually, I think I will take the under. Nonetheless, it will be a great learning experience for someone passionate about scentific computation.

## 2. Problem Design / Architecture
### Sample Sizes and Matrix Dimensions
The domain of the PDE from which our matrices arise is a discrete $$N \times N$$ grid. Thus $$A$$ has a total of $$N^2$$ entries - the size of the problem increases quadratically, so we should be careful not to bite off more than we can chew. After careful consideration, we shall start with a nice power of 2: $$N = 32$$. This gives us $$32^2 = 1024 \approx 1000$$ unknowns, which, heuristically, seems appropriate. Later on, we will increase $$N$$ to see if our network can generalise to higher dimensions.

We will train our model on a sample of 500 such matrices. This was based on some rough back-of-the-envelope calculations based on how much computing time and memory seems appropriate for this project. Furthermore, due to locality properties of this PDE, each training matrix encodes a lot of information, so we shouldn't necessarily need a massive training set to see decent results.

### Neural Network Design
Recall that the diffusion equation models the distribution of a diffusing material in a medium. A canonical example of its use is to model the diffusion of a dye in a liquid. Intuitively, then, we can see that this problem has locality and translation-invariance properties (as mentioned above). Thus, it is natural to use a form of neural network tailored exactly to this kind of problem; namely, a convolutional neural network.  




### Schedule
We shall divide this project into several phases.  
#### Phase 1
This phase will consist of training and validating on a sample of 200 $$32 \times 32$$ matrices. Our goal is to outperform a Jacobi preconditioner in terms of iterations with PCG.
#### Phase 2
This phase will consist of training and validating on a sample of 300 $$64 \times 64$$ matrices. Success will be measured based on how well the network generalises to these higher dimensions.
#### Phase 3
**TBD**


