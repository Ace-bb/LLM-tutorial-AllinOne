在过去几年中，深度学习的进步表明，随着训练时间、数据集大小和模型大小的增加，大型模型的性能持续改善理解这些系统在资源增加时的改进速率对于希望以计算最优的方式扩展模型大小和训练时间的人来说至关重要。经验上，几项研究发现，对于具有𝑡次更新步骤和𝑁个参数的网络，其损失可以相对准确地近似为这些量中的幂律。

$$\mathcal L = C_t \ t^{-\alpha_t} + C_N N^{-\alpha_N} + \mathcal L_{\infty}.$$

常数 $C_t, C_N$ 和指数 $\alpha_t$ 和 $\alpha_N$ 以及渐近线 $\mathcal L_{\infty}$ 取决于学习任务和神经网络架构。下面，我们展示了两个简单的例子，分别是在视觉任务（使用卷积网络的 CIFAR-5M）和语言建模任务（使用Transformer的 Wikitext-103 语言建模）中，其中我们通过增加宽度 $N$ 来改变模型的参数数量。


![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-7.49.23-AM-1024x480.png) ![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-7.49.33-AM-1024x578.png)

我们看到，随着模型规模和训练步数的增加，性能定期提高。尽管我们的 [上一篇博客文章](https://kempnerinstitute.harvard.edu/research/deeper-learning/infinite-limits-of-neural-networks/) 讨论了神经网络的无限参数极限 𝑁→∞，但理解扩展法则需要表征有限模型规模的影响，这些影响在足够的训练后限制了性能。

Additional Observed Phenomena Related to Scaling Laws  
与缩放法则相关的额外观察现象
----------------------------------------------------------------------

In addition to the simple power-law neural scaling laws in training time and model size that are observed in many scaling law studies, we would also like to explain some other observed effects in network training. Some key empirical observations include  
除了许多缩放法研究中观察到的训练时间和模型规模的简单幂律神经缩放法则外，我们还想解释一些在网络训练中观察到的其他效应。一些关键的经验观察包括

1.  **Asymmetric exponents**: the model size and training time exponents $\alpha_N , \alpha_t$ are often different[1-6](#references).  
    **非对称指数**：模型大小和训练时间的指数 $\alpha_N , \alpha_t$ 通常是不同的[1-6](#references)。
2.  **Data reuse effects**: Reusing data leads to a gradual buildup of the gap between train loss and test loss compared to the online training regime where data is never repeated and train and test losses are identical[[10](#footnote_9_8573 "Preetum Nakkiran, Behnam Neyshabur, and Hanie Sedghi. The deep bootstrap framework: Good online learners are good offline generalizers. International Conference on Learning Representations, 2021")][[11](#footnote_10_8573 "Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. Advances in Neural Information Processing systems, 2023")].  
    **数据重用的影响**：与从不重复数据的在线训练方式相比，重用数据会导致训练损失和测试损失之间的差距逐渐加大，训练和测试损失相同[[10](#footnote_9_8573 "Preetum Nakkiran, Behnam Neyshabur, and Hanie Sedghi. The deep bootstrap framework: Good online learners are good offline generalizers. International Conference on Learning Representations, 2021")][[11](#footnote_10_8573 "Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. Advances in Neural Information Processing systems, 2023")]。
3.  **Change in rate of convergence to large width limit**: Very early in training networks converge at a rate $1/\text{width}$ to their infinite width limit [[12](#footnote_11_8573 "Blake Bordelon and Cengiz Pehlevan. Dynamics of finite width kernel and prediction fluctuations in mean field neural networks. Advances in Neural Information Processing Systems, 2023.")][[13](#footnote_12_8573 "Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, and Cengiz Pehlevan. Feature-learning networks are consistent across widths at realistic scales, Advances in Neural Information Processing Systems 2023.")]Late in training, the network has a loss that scales as $\text{width}^{-c}$ where $c$ is task and architecture dependnent[12](#references), 13[[14](#footnote_13_8573 "Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. arXiv preprint arXiv:2102.06701, 2021.")].  
    **收敛速率变化到大宽度极限**: 在训练的早期，网络以 $1/\text{width}$ 的速率收敛到它们的无限宽度极限，在训练的后期，网络的损失呈现为 $\text{width}^{-c}$ 的比例，其中 $c$ 依赖于任务和架构.
4.  **Compute suboptimality of ensembling**: On large datasets or in online training, ensembling over many randomly initialized models often fails to match the performance of training a single larger model.[13](#references)  
    **计算集合的次优性**：在大数据集或在线训练中，许多随机初始化模型的集合通常无法达到训练单个更大模型的性能。[13](#references)

Below we describe a very simple model of network training dynamics that can reproduce these effects.  
下面我们描述一个非常简单的网络训练动态模型，可以重现这些效果。

A Model of Compute Optimal Scaling Laws  
计算最优缩放法则的模型
-----------------------------------------------------

We seek the simplest possible model that captures all of these observed phenomena. In the [previous blog post](https://kempnerinstitute.harvard.edu/research/deeper-learning/infinite-limits-of-neural-networks/), we introduced the kernel limit of neural networks which arises from randomly initializing large width networks in a certain parameterization. This model has serious deficiencies as a model of neural network dynamics since the internal representations in the network are static throughout learning, however it is much more analytically tractable since it is essentially a linear model. Despite the deficiencies of this kernel (linear model) regime of neural network training, we will show that all of the above neural scaling law effects are already observable in the learning dynamics of a linear model. We therefore aim to characterize the test and train loss dynamics of this kind of model.  
我们寻求尽可能简单的模型，以捕捉所有这些观察到的现象。在[之前的博客文章](https://kempnerinstitute.harvard.edu/research/deeper-learning/infinite-limits-of-neural-networks/)中，我们介绍了神经网络的核极限，这一现象源于在某种参数化下随机初始化大宽度网络。作为神经网络动态的模型，这个模型存在严重的缺陷，因为网络中的内部表征在学习过程中是静态的，然而，由于它基本上是一个线性模型，因此解析性更强。尽管这个核（线性模型）模式的神经网络训练存在缺陷，我们将证明上述所有神经扩展定律效应已经在线性模型的学习动态中可观察到。因此，我们的目的是描述这种模型的测试和训练损失动态。

We consider a network with 𝑁 trainable parameters, 𝑃 data points, and 𝑡 timesteps of training. Our goal is to characterize the expected or typical test error as a function of these quantities over random draws of datasets and initial network features.  
我们考虑一个具有 𝑁 个可训练参数、𝑃 个数据点和 𝑡 个训练时间步的网络。我们的目标是将预期或典型测试误差作为这些数量的函数，对数据集和初始网络特征的随机抽样进行特征化。

A Linear Model for Compute-Optimal Scaling  
计算最优扩展的线性模型
--------------------------------------------------------

Neural networks in certain limits can operate as linear models. In this regime, the output prediction $f$ of the neural network is a linear combination of its $N$ features $\left\{\tilde{\psi}_k(x) \right\}_{k=1}^N$, which arise from a rank-$N$ kernel associated with an $N$ parameter model. The target function, $y$, on the other hand, is a linear combination of a complete set of features $\left\{ \psi_k(x) \right\}_{k=1}^\infty$, corresponding to a complete set of square integrable functions. These expansions take the form $$f(x) = \sum_{k} \tilde{\psi}_k(x) w_k \ , \ y(x) = \sum_k \psi_k(x) w^\star_k.$$ We will use the basis of features $\psi_k(x)$ as the infinite width kernel eigenfunctions \.[8,10](#references) The finite model’s $N$ features $\{ \tilde{\psi}_k \}_{k=1}^N$ can be expanded in the basis of the original features with coefficients $A_{k\ell}$, $$\tilde{\psi}_k(x) = \sum_{\ell = 1}^\infty A_{k\ell} \ \psi_\ell(x) .$$  
在某些限制条件下，神经网络可以作为线性模型运作。在这种情况下，神经网络的输出预测 $f$ 是其 $N$ 个特征 $\left\{\tilde{\psi}_k(x) \right\}_{k=1}^N$ 的线性组合，这些特征源自与 $N$ 参数模型相关的秩为 $N$ 的核。另一方面，目标函数 $y$ 是与一组完整的平方可积函数对应的完整特征集 $\left\{ \psi_k(x) \right\}_{k=1}^\infty$ 的线性组合。这些展开的形式为 $$f(x) = \sum_{k} \tilde{\psi}_k(x) w_k \ , \ y(x) = \sum_k \psi_k(x) w^\star_k.$$ 我们将使用特征基 $\psi_k(x)$ 作为无限宽度核的特征函数 \.[8,10](#references) 有限模型的 $N$ 个特征 $\{ \tilde{\psi}_k \}_{k=1}^N$ 可以用原始特征的基展开，系数为 $A_{k\ell}$，即 $$\tilde{\psi}_k(x) = \sum_{\ell = 1}^\infty A_{k\ell} \ \psi_\ell(x) .$$

We will model the matrix $A_{k\ell}$ as random, which reflects the fact that the empirical kernel in a finite parameter model depends on the random initialization of the network weights. The statics of this model were analyzed in prior works [[15](#footnote_14_8573 "Alexander Maloney, Daniel Roberts, James Sully. A solvable model of neural scaling laws. arXiv preprint arXiv:2210.16859. 2022.")][[16](#footnote_15_8573 "Alexander Atanasov, Blake Bordelon, Sabarish Sainathan, Cengiz Pehlevan. Onset of Variance-limited Behavior for networks in the lazy and rich regimes. arXiv preprint arXiv:2212.12147. 2022.")], but in this work we focus on the dynamics of training.  
我们将把矩阵 $A_{k\ell}$ 建模为随机矩阵，这反映了有限参数模型中的经验核依赖于网络权重的随机初始化。该模型的静态在之前的工作中进行了分析 [[15](#footnote_14_8573 "Alexander Maloney, Daniel Roberts, James Sully. A solvable model of neural scaling laws. arXiv preprint arXiv:2210.16859. 2022.")][[16](#footnote_15_8573 "Alexander Atanasov, Blake Bordelon, Sabarish Sainathan, Cengiz Pehlevan. Onset of Variance-limited Behavior for networks in the lazy and rich regimes. arXiv preprint arXiv:2212.12147. 2022.")]，但在本工作中，我们专注于训练的动态。

To train the model parameters $w_k$ with gradient based training, we randomly sample a training set with $P$ data points ${ x_\mu }_{\mu=1}^P$ drawn from the population distribution and train the model with gradient descent/gradient flow on the training loss $\hat{\mathcal{L}} = \frac{1}{P} \sum_{\mu=1}^P [f(x_\mu) – y(x_\mu)]^2$. For gradient flow, we have  
为了使用基于梯度的训练来训练模型参数 $w_k$，我们随机抽取一个包含 $P$ 个数据点的训练集 ${ x_\mu }_{\mu=1}^P$，该数据点来自于总体分布，并使用训练损失 $\hat{\mathcal{L}} = \frac{1}{P} \sum_{\mu=1}^P [f(x_\mu) – y(x_\mu)]^2$ 进行梯度下降/梯度流训练模型。对于梯度流，我们有

$$\frac{d}{dt} \mathbf w(t) = – \eta \nabla \hat{\mathcal L}(\mathbf w(t)) . $$

For simplicity in this post we focus on gradient flow, but discrete time algorithms such as gradient descent or momentum and one pass SGD can also be handled in our framework, see our paper.[7](#references)  
为了简化，我们在这篇文章中专注于梯度流，但离散时间算法，如梯度下降、动量和一次性随机梯度下降（SGD）也可以在我们的框架中处理，详见我们的论文。[7](#references)

Our goal is to track the test error $\mathcal L =\mathbb{E}_{x} [f(x) – y(x)]^2$ over training time. Since $f(x,t)$ depends on the random dataset and random projection, we have to develop a method to average over these sources of disorder.  
我们的目标是跟踪测试误差 $\mathcal L =\mathbb{E}_{x} [f(x) – y(x)]^2$ 随训练时间的变化。由于 $f(x,t)$ 依赖于随机数据集和随机投影，我们必须开发一种方法来对这些无序源进行平均。

Dynamical Mean Field Theory for Learning Curves  
学习曲线的动态均场理论
-------------------------------------------------------------

We develop a theory to track the test and train loss dynamics in this random feature model for 𝑁, 𝑃 large. To analytically calculate these losses, we utilize ideas from statistical physics, specifically dynamical mean field theory (DMFT). This method summarizes all relevant summary statistics of the network in terms of correlation and response functions.  
我们提出了一种理论，以跟踪在这个随机特征模型中，𝑁和𝑃很大的情况下，测试和训练损失的动态变化。为了分析性地计算这些损失，我们利用了统计物理的思想，特别是动态均场理论（DMFT）。该方法通过相关性和响应函数来总结网络的所有相关统计量。

Below, we plot an example of our theoretical predictions of test loss (dashed black lines) against experimental training (solid) for feature maps of varying dimension $N$ with large dataset size $P=1000$. Standard deviations over random realizations of the dataset and projection matrix $A$ are plotted as bands of shaded color. We see that the theory (dashed black lines) accurately captures the deviation of finite models from the $N,P \to \infty$ limiting dynamics (blue). Further, increasing training time $t$ and increasing model size $N$ leads to consistent reductions in test loss.  
下面，我们绘制了我们的理论预测的测试损失（虚线黑色线条）与实验训练（实线）之间的示例，特征图的维度 $N$ 变化，数据集大小为 $P=1000$。数据集和投影矩阵 $A$ 随机实现的标准差被绘制为阴影颜色的带状。我们看到理论（虚线黑色线条）准确地捕捉到了有限模型与 $N,P \to \infty$ 限制动态（蓝色）之间的偏差。此外，增加训练时间 $t$ 和增加模型大小 $N$ 会导致测试损失的一致减少。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.26.50-AM-1024x735.png)

However, if the dataset size is small, the returns to increasing model size eventually diminish as the test loss is bottlenecked by the amount of available data. Below we plot varying model sizes $N$ as we train on a dataset of size $P=128$.  
然而，如果数据集的规模很小，增加模型大小的收益最终会减小，因为测试损失受到可用数据量的瓶颈。下面我们绘制了在数据集大小为 $P=128$ 时，不同模型大小 $N$ 的训练情况。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.26.59-AM-1024x712.png)

Power Law Bottleneck Scalings  
幂律瓶颈缩放
--------------------------------------

From the last section, we saw that the performance of the model can be bottlenecked by one of the three computational/statistical resources: training time $t$, model size $N$, and total available data $P$. By this we mean that even if the other two resources were effectively infinite, the loss can still be nonzero because of the finite value of the third quantity. In this section, we show that the dependence of the loss on these resources can obey power laws when the features themselves have power-law structure. It has been observed that the spectra of neural network kernels on real datasets often follow power-laws[14](#references) [[17](#footnote_16_8573 "Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In International Conference on Machine Learning, pp. 1024–1034. PMLR, 2020.")]  
从上一节，我们看到模型的性能可能会受到三种计算/统计资源之一的瓶颈：训练时间 $t$，模型大小 $N$ 和可用数据总量 $P$。 这意味着即使其他两个资源实际上是无限的，损失仍然可能是非零的，因为第三个量的有限值。在这一节中，我们展示了损失对这些资源的依赖关系在特征本身具有幂律结构时可以遵循幂律。已经观察到，真实数据集上神经网络核的谱通常遵循幂律。  
$$\lambda_k = \mathbb{E}_{x} \psi_k(x)^2 \sim k^{-b} \ , \ [\mathbb{E}_x y(x) \psi_k(x) ]^2 \sim k^{-a} .$$

For this kind of feature structure, our theory gives the following approximate scaling laws when bottlenecked by one of the three resources (time, model size, and dataset size)  
对于这种特征结构，我们的理论在受到三种资源（时间、模型大小和数据集大小）之一的瓶颈时，给出了以下近似缩放法则

\begin{align}  
\mathcal L(t,P,N) \approx  
\begin{cases}  
t^{-(a-1)/b} \ , \ N,P \to \infty  
\\  
N^{-\min\{a-1,2b\}} \ , \ t,P \to \infty  
\\  
P^{-\min\{a-1,2b\}} \ , \ t, N \to \infty  
\end{cases}  
\end{align}

For most cases of interest, the expoents satisfy $a-1 < 2b$[14,17](#references), leading to $\sim N^{-(a-1)}, P^{-(a-1)}$ model and data bottleneck scaling laws. In these cases, our result predicts that in general the training time exponent is smaller than model size or data exponents, depending on the rate of decay of the eigenvalues, set by $b$.  
对于大多数感兴趣的情况，指数满足 $a-1 < 2b$[14,17](#references)，导致 $\sim N^{-(a-1)}, P^{-(a-1)}$ 模型和数据瓶颈缩放定律。在这些情况下，我们的结果预测一般训练时间指数小于模型大小或数据指数，这取决于由 $b$ 设置的特征值的衰减速率。

An intuitive way to interpret this result in the case of interest ($\min{a-1,2b} = a-1$) is that $t$ steps of gradient descent on $N$ features and $P$ data can capture at most  
一种直观的方法来解释这一结果（在关注的情况下，$\min{a-1,2b} = a-1$）是，$t$步的梯度下降在$N$个特征和$P$个数据上最多可以捕获  
$$k_{\star} \approx \min{ t^{1/b}, N, P } . $$  
spectral components of the target function. The loss is determined by the remaining variance that is not captured in these top $k_\star$ components $\mathcal L \approx \sum_{k > k_\star} \mathbb{E}_{x}[y(x) \psi_k(x)]^2$. Thus these bottleneck scaling laws can be viewed low-rank effects in the empirical kernel that limit the performance of the model.  
目标函数的频谱成分。损失由未包含在这 $k_\star$ 个成分中的剩余方差决定 $\mathcal L \approx \sum_{k > k_\star} \mathbb{E}_{x}[y(x) \psi_k(x)]^2$。因此，这些瓶颈缩放法则可以被视为限制模型性能的经验核中的低秩效应。

Compute Optimal Scaling Laws for this Model  
计算此模型的最优缩放法则
----------------------------------------------------------

In this section we consider a regime of training where there is sufficient data, such as the online training regime of large language models. By approximating the test loss as a linear combination of the model size and time bottleneck scalings, we can derive the compute optimal scaling of training time and model size with respect to total compute $C=N t$. This compute budget $C$ is the total number of floating point operations required to train the model. For the optimal choice of training time and model size, we find the loss depends on compute as  
在本节中，我们考虑一个训练模式，其中有足够的数据，例如大语言模型的在线训练模式。通过将测试损失近似为模型规模和时间瓶颈缩放的线性组合，我们可以推导出相对于总计算 $C=N t$ 的训练时间和模型规模的计算最优缩放。这个计算预算 $C$ 是训练模型所需的浮点运算总数。对于训练时间和模型规模的最优选择，我们发现损失依赖于计算，因为

$$\mathcal L_\star(C) \sim C^{-\min{a-1,2b}(a-1) /( b \min{a-1,2b} + a-1)}$$

which in most cases of interest will simply be $\mathcal{L}_\star(C) \sim C^{- (a-1)/(b+1)}$. We show an example of this for $(a,b) = (2,1)$ below. Our theoretical scaling law is compared to the experimental loss curves from training models of varying size $N$ for multiple timesteps.  
在大多数感兴趣的情况下，将简单地表示为 $\mathcal{L}_\star(C) \sim C^{- (a-1)/(b+1)}$。我们在下面展示了 $(a,b) = (2,1)$ 的示例。我们的理论缩放定律与针对多个时间步长训练不同大小 $N$ 的模型的实验损失曲线进行了比较。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.34.51-AM-1024x775.png)

This model shows how the data structure and architecture influence the compute costs of training a highly performant model. Specifically, the decay rate of target coefficients and eigenvalues controls the compute optimal scaling law of the model. For models with fast eigenvalue decay rates, it is preferable to scale up training time much faster than scaling up model size as the optimal scaling rule is $t \sim C^{\frac{b}{1+b}}$ and $N \sim C^{\frac{1}{1+b}}$. As $b \to 1$ the optimal scaling is symmetric.  
该模型展示了数据结构和架构如何影响训练高性能模型的计算成本。具体而言，目标系数和特征值的衰减率控制模型的计算最佳扩展规律。对于特征值衰减率较快的模型，更加希望将训练时间的扩展速度远快于模型规模的扩展，因为最佳扩展规则是 $t \sim C^{\frac{b}{1+b}}$ 和 $N \sim C^{\frac{1}{1+b}}$。当 $b \to 1$ 时，最佳扩展是对称的。

Build up of Finite Width Effects  
有限宽度效应的积累
--------------------------------------------

Many works have observed that the early training-time dynamics of networks with width $N$ deviates from the infinite $N$ limit with a scaling rate of $1/N$[12-14](#references), but that after a long amount of training on sufficient quantities of data the convergence rate exhibits a task-dependent scaling law $N^{-\alpha_N}$[13-14](#references). Our model also exhibits a transition in the convergence rates as training takes place. Below we show the early time loss of our model at $N$ compared to our model in the $N \to \infty$ limit, seeing a $1/N$ convergence rate.  
许多研究观察到，宽度为 $N$ 的网络在早期训练时间的动态行为偏离无穷大 $N$ 极限，其缩放率为 $1/N$[12-14](#references)，但在经过大量数据的长时间训练后，收敛速率表现出依赖于任务的缩放律 $N^{-\alpha_N}$[13-14](#references)。我们的模型在训练过程中也显示出收敛速率的转变。以下我们展示了在 $N$ 时我们模型的早期时间损失，与在 $N \to \infty$ 极限下我们的模型进行比较，观察到 $1/N$ 收敛速率。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.38.20-AM-1024x754.png)

However, after significant training time, the model will eventually depend on the model size $N$ with a scaling exponent that is task-dependent (the bottleneck scaling) as we show below.  
然而，在经过大量训练时间后，该模型最终将依赖于模型大小 $N$，其缩放指数取决于任务（瓶颈缩放），如下所示。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.38.30-AM-1024x757.png)

We see that this scaling law can significantly differ from the $1/N$ rate and indeed becomes task dependent.  
我们看到这一缩放法则与 $1/N$ 速率可以显著不同，实际上它变得依赖于任务。

Data Reuse and Buildup of Test/Train Loss Gaps  
数据重用和测试/训练损失差距的积累
------------------------------------------------------------------

Many works have also observed that the early time training with a finite dataset is well approximated by training with infinite data[10-11](#references), however over time a gap develops between training and test losses. This is also a naturally occuring feature in our model and the DMFT equations exactly describe how the test and train losses diverge over time. Below we plot dynamics for $N=512$ with varying dataset size $P$.  
许多研究还观察到，使用有限数据集的早期训练与无限数据训练的结果非常接近[10-11](#references)，然而随着时间的推移，训练损失和测试损失之间出现了差距。这也是我们模型中自然发生的特征，DMFT 方程准确描述了测试损失和训练损失随时间如何发散。下面我们绘制了 $N=512$ 在不同数据集大小 $P$ 下的动态。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.41.58-AM-1024x758.png)

We note that the test and train losses are close initially but accumulate finite $P$ corrections that drive the separation of test and train. These corrections are larger for small $P$ and vanish as $P \to \infty$.  
我们注意到测试和训练损失最初接近，但积累有限的 $P$ 修正，导致测试和训练的分离。对于小 $P$，这些修正更大，而当 $P \to \infty$ 时消失。

Ensembling Often Outperformed by Increasing Width  
集成通常通过增加宽度获得更好的表现
---------------------------------------------------------------------

Finite sized models with random initial weights can be thought of noisy approximations of infinitely sized neural networks. This extra noise can lead to worse performance and can be eliminated by training multiple models with independent initialization in parallel and averaging their outputs, a procedure known as ensembling. However recent experiments have demonstrated that the benefits to ensembling, while non-negligible, are not as significant as the benefit of increasing model size[12-13](#references).  
具有随机初始权重的有限大小模型可以被视为无限大小神经网络的嘈杂近似。额外的噪声可能导致性能下降，而通过并行训练多个独立初始化的模型并将其输出平均，可以消除这种噪声，这一过程称为集成。然而，最近的实验表明，尽管集成的好处不容忽视，但其好处并不如增加模型大小的好处显著[12-13](#references)。

In our toy model, we can analyze the effect of ensembling on the test loss and ask whether ensembling is compute optimal. Training an ensemble of $E$ networks and averaging their outputs would incur a compute cost of $C = E N t$. Below we plot loss as a function of compute for $E=1$ and $E=4$ ensembles for varying width $N$.  
在我们的玩具模型中，我们可以分析集成对测试损失的影响，并询问集成是否是计算最优的。训练一个由 $E$ 个网络组成的集成并平均它们的输出将产生计算成本 $C = E N t$。下面我们将损失绘制为计算的函数，针对 $E=1$ 和 $E=4$ 的集成，并改变宽度 $N$。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.44.17-AM-1024x768.png)

At each value of compute $C$, it is prefereable to choose the larger model with $E=1$ than to use a smaller model with $E=4$. We argue the reason for this is that doubling $N$ has a similar effect on the variance as doubling $E$. However, doubling $N$ also reduces the _bias_.  
在每个计算值$C$下，选择$E=1$的较大模型比使用$E=4$的小模型更可取。我们认为原因在于，翻倍$N$对方差的影响与翻倍$E$相似。然而，翻倍$N$也减少了_偏差_。

Bias and Mode Errors 偏差和模式错误
----------------------------

To give a flavor of how this theory works, we show how DMFT recovers the bias along the $k$th feature for all $k$. This error is given by: $$ H_k(t) = \frac{\mathbb{E}_{x} [(y(x) – f(x,t)) \psi_k(x)] }{\mathbb{E}_{x} [y(x) \psi_k(x)]}$$  
为了让您了解这个理论是如何工作的，我们展示了 DMFT 如何恢复所有 $k$ 的第 $k$ 个特征的偏差。这个误差由以下公式给出：$$ H_k(t) = \frac{\mathbb{E}_{x} [(y(x) – f(x,t)) \psi_k(x)] }{\mathbb{E}_{x} [y(x) \psi_k(x)]}$$

Our theory explictly calculates the Fourier transform $\mathcal H_k(\omega)$ in closed form. This is given in terms of the eigenvalues $\lambda_k$, the dataset size $P$ and the model size $N$. An example of the closed form solution for the $H_k(t)$ is plotted below with $N = 128$ and varying values for $P$.  
我们的理论明确地以封闭形式计算傅里叶变换 $\mathcal H_k(\omega)$。这是用特征值 $\lambda_k$、数据集大小 $P$ 和模型大小 $N$ 表示的。下面绘制了 $H_k(t)$ 的封闭形式解的示例，$N = 128$，并且 $P$ 的值各不相同。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.45.48-AM-1024x618.png)

The error along the $k$-th eigendirection deviates from the infinite data and infinite model limit (gray lines) and eventually saturates as $t \to \infty$, giving a final loss which depends on $N$ and $P$. Even if $P \to \infty$, the $H_k$ curves saturate in this plot due to the finite value of $N = 128$. We show the losses for $k=1$ (solid) and $k=10$ (dashed). We find that the bias, which is set by $H_k$ decreases as $N,P$ and $t$ increase.  
沿着第 $k$ 个特征方向的误差偏离了无限数据和无限模型极限（灰线），并最终在 $t \to \infty$ 时饱和，产生的最终损失依赖于 $N$ 和 $P$。即使 $P \to \infty$，由于 $N = 128$ 的有限值，$H_k$ 曲线在此图中也饱和。我们展示了 $k=1$（实线）和 $k=10$（虚线）的损失。我们发现，由 $H_k$ 设置的偏差随着 $N$、$P$ 和 $t$ 的增加而减小。

One Pass SGD and Batch Noise  
一轮随机梯度下降和批量噪声
--------------------------------------------

We can also use our methods to analyze stochastic gradient descent (SGD) in discrete time without data reuse. In this setting, the finite model size and finite training time can still limit performance, but the finite batch size $B$ only introduces additional _variance_ in the dynamics as we illustrate below. On the left, we vary the model size $N$ with batchsize set to $B=32$ and see that we still obtain model size bottlenecks which are qualitatively similar to before. We also see additional small fluctuations in the loss from batch to batch. On the right, we show $N=256$ with varying batch size, showing that the expected loss and scale of fluctuations are higher for smaller batches.  
我们还可以使用我们的方法在离散时间内分析随机梯度下降（SGD），而不重用数据。在这种情况下，有限的模型大小和有限的训练时间仍然会限制性能，但有限的批量大小 $B$ 仅会在动态中引入额外的 _方差_，如下面所示。在左侧，我们将模型大小 $N$ 变化，批量大小设置为 $B=32$，可以看到我们仍然获得了模型大小的瓶颈，这与之前的情况 qualitatively 类似。我们还看到损失在每个批次之间有额外的小波动。在右侧，我们展示了 $N=256$，批量大小不同，显示出较小的批量的预期损失和波动规模更高。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.47.38-AM-1024x379.png)

In this setting, a test-train gap is not possible since every fresh batch of data gives an unbiased estimate of the population loss. As a consequence, online learning does not experience a data bottleneck in the bias, but only additional variance from the fluctuations in the SGD updates. These updates disappear in the continuous time (infinitesimal learning rate) limit which recovers the infinite data $P \to \infty$ limit of the previously discussed gradient flow equations.  
在这种情况下，测试-训练差距是不可能的，因为每一批新数据都能提供对总体损失的无偏估计。因此，在线学习不会在偏差上出现数据瓶颈，而只是由于 SGD 更新中的波动而产生额外的方差。这些更新在连续时间（无穷小学习率）限制中消失，从而恢复了之前讨论的梯度流方程的无限数据 $P \to \infty$ 限制。

What is Missing? Feature Learning Scaling Laws  
缺失的是什么？特征学习的扩展定律
-----------------------------------------------------------------

Our model is based on a kernel approximation of neural network training which fails to capture the benefits to performance due to feature learning. Below we plot neural networks trained in the kernel regime (solid) and the predicted compute scaling exponent (blue), obtained from fitting the exponents $a$ and $b$ to the measured initial kernel spectra. We also plot the loss curves for networks in the feature learning regime (dotted lines).  
我们的模型基于神经网络训练的核逼近，但未能捕捉到特征学习带来的性能提升。下面我们绘制了在核领域训练的神经网络（实线）和通过拟合测得的初始核谱得到的预测计算缩放指数（蓝色），该指数为$a$和$b$。我们还绘制了特征学习领域网络的损失曲线（虚线）。

![](https://kempnerinstitute.harvard.edu/app/uploads/2024/06/Screenshot-2024-06-11-at-9.47.46-AM-1024x711.png)

While the networks operating in the kernel regime (solid) are well described by our theoretical prediction for the compute scaling law, the networks in the rich, feature learning regime have a much better dependence on compute $C$. This illustrates that quantitatively capturing the compute optimal scaling exponents observed in practice will require a theory of how feature learning accelerates convergence during training.  
虽然在内核机制（固态）下运行的网络很好地符合我们对计算规模法则的理论预测，但在丰富的特征学习机制下的网络对计算 $C$ 有更好的依赖性。这表明，要定量捕捉实践中观察到的计算最优缩放指数，将需要一种理论来解释特征学习如何加速训练过程中的收敛。

Conclusion 结论
-------------

We proposed a simple linear model to analyze dynamical neural scaling laws. This model captures many of the observed phenomena related to network training and test loss dynamics. Looking forward, theories which incorporate feature learning into the network training dynamics will improve our understanding of scaling laws. The fact that infinite sized models perform the best suggests that starting with theories of feature learning at infinite width are a good place to start.  
我们提出了一个简单的线性模型来分析动态神经缩放规律。该模型捕捉了与网络训练和测试损失动态相关的许多观察现象。展望未来，将特征学习纳入网络训练动态的理论将有助于我们更好地理解缩放规律。无限大小的模型表现最佳这一事实表明，从无限宽度的特征学习理论开始是一个良好的起点。

References 参考文献
---------------

1.  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.[[Return to text ↩](#identifier_0_8573)]  
    贾里德·卡普兰，萨姆·麦肯德利什，汤姆·赫尼根，汤姆·B·布朗，本杰明·切斯，瑞温·查尔德，斯科特·格雷，亚历克·拉德福德，杰弗里·Wu，以及达里奥·阿莫代。神经语言模型的规模法则。arXiv 预印本 arXiv:2001.08361，2020。[[返回文本 ↩](#identifier_0_8573)]
2.  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.[[Return to text ↩](#identifier_1_8573)]  
    汤姆·布朗，本杰明·曼，尼克·莱德，梅拉妮·萨比亚，贾里德·D·卡普兰，普拉法拉·达里瓦尔，阿尔文·尼拉坎坦，普拉纳夫·夏姆，吉里什·萨斯特里，阿曼达·阿斯克尔等人。语言模型是少量样本学习者。《神经信息处理系统进展》，33:1877–1901，2020 年。[[返回文本 ↩](#identifier_1_8573)]
3.  Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.[[Return to text ↩](#identifier_2_8573)]
4.  Gemini Team Google. Gemini: A Family of Highly Capable Multimodal Models, arXiv preprint arXiv:2203.15556, 2024.[[Return to text ↩](#identifier_3_8573)]
5.  Tamay Besiroglu, Ege Erdil, Matthew Barnett and Josh You. Chinchilla Scaling: A replication attempt. arXiv preprint arXiv:2404.10102, 2024.[[Return to text ↩](#identifier_4_8573)]
6.  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.[[Return to text ↩](#identifier_5_8573)]
7.  Blake Bordelon, Alex Atanasov, Cengiz Pehlevan. Dynamical Model of Neural Scaling Laws. To appear International Conference of Machine Learning 2024.[[Return to text ↩](#identifier_6_8573)]
8.  Elliot Paquette, Courtney Paquette, Lechao Xiao, Jeffrey Pennington. 4+3 Phases of Compute-Optimal Neural Sclaing Laws. arXiv preprint arXiv:2405.15074, 2024.[[Return to text ↩](#identifier_7_8573)]
9.  Licong Lin, Jingfeng Wu, Sham M. Kakade, Peter L. Bartlett, Jason D. Lee. Scaling Laws in Linear Regression: Compute, Parameters, and Data. arXiv preprint arXiv:2406:08466, 2024.[[Return to text ↩](#identifier_8_8573)]
10.  Preetum Nakkiran, Behnam Neyshabur, and Hanie Sedghi. The deep bootstrap framework: Good online learners are good offline generalizers. International Conference on Learning Representations, 2021[[Return to text ↩](#identifier_9_8573)]
11.  Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. Advances in Neural Information Processing systems, 2023[[Return to text ↩](#identifier_10_8573)]
12.  Blake Bordelon and Cengiz Pehlevan. Dynamics of finite width kernel and prediction fluctuations in mean field neural networks. Advances in Neural Information Processing Systems, 2023.[[Return to text ↩](#identifier_11_8573)]
13.  Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, and Cengiz Pehlevan. Feature-learning networks are consistent across widths at realistic scales, Advances in Neural Information Processing Systems 2023.[[Return to text ↩](#identifier_12_8573)]
14.  Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. arXiv preprint arXiv:2102.06701, 2021.[[Return to text ↩](#identifier_13_8573)]
15.  Alexander Maloney, Daniel Roberts, James Sully. A solvable model of neural scaling laws. arXiv preprint arXiv:2210.16859. 2022.[[Return to text ↩](#identifier_14_8573)]
16.  Alexander Atanasov, Blake Bordelon, Sabarish Sainathan, Cengiz Pehlevan. Onset of Variance-limited Behavior for networks in the lazy and rich regimes. arXiv preprint arXiv:2212.12147. 2022.[[Return to text ↩](#identifier_15_8573)]
17.  Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In International Conference on Machine Learning, pp. 1024–1034. PMLR, 2020.[[Return to text ↩](#identifier_16_8573)]