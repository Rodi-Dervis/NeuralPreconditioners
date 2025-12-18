# NeuralPreconditioners
This is a project to develop a neural network which can find preconditioner matrices for solving linear systems of equations via the Preconditioned Conjugate Gradient method (PCG) under certain constraints.

## 1. Theoretical Setup / Motivation
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

### How does a neural network help?
You came to read about neural networks and cool AI stuff and I just made you sit through a wall of text about linear algebra and iterative methods. I promise, this is only *partly* because of my agenda and sworn duty as a maths guy to practice a form of mathematical evangelism; mostly, I think the problem we are trying to solve should be clear.

Currently, a neural network cannot outperform classical iterative methods end-to-end. However, research has shown that neural networks *can* outperform standard methods for finding a preconditioner matrix. Thus, if we expend the significant, initial cost of training a neural network to give us suitable preconditioner matrices, we can then call on it as needed prior to performing PCG and gradually save time in the macro sense. **This is the objective of the project.**

Will this particular neural network ultimately live up to these high expectations? Given my resources, both computationally and intellectually, I think I will take the under. Nonetheless, it will be a great learning experience for someone passionate about scentific computation.

## 2. Problem Design / Architecture
