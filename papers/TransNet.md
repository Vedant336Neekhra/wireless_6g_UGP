---
title: TransNet

---

## The Linear System Model and the Channel Matrix **H**

In any wireless link, the relationship between the transmitted signal vector  
$\mathbf{x} \in \mathbb{C}^{N_t}$ (where $N_t$ is the number of transmit antennas)  
and the received signal vector  
$\mathbf{y} \in \mathbb{C}^{N_r}$ (where $N_r$ is the number of receive antennas)  
is modeled as a linear system:  
$\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}$

Here, $\mathbf{n}$ represents **additive white Gaussian noise (AWGN)**, a fundamental
impairment in all communication systems.  
The matrix $\mathbf{H} \in \mathbb{C}^{N_r \times N_t}$ is the **Channel Matrix**.  
It contains the **Channel State Information (CSI)**.
Unlike the weight matrices in a neural network, which are learned parameters,
$\mathbf{H}$ is a stochastic, time-varying operator imposed by nature.
It represents the aggregate effect of the environment on the electromagnetic
wave as it travels from transmitter to receiver. Each element $h_{ij}$ in this
matrix is a complex number $a + bi$ (or $Ae^{j\phi}$), where:

- **Amplitude ($A$):** Represents the attenuation or fading of the signal power
  due to distance and obstacles.

- **Phase ($\phi$):** Represents the delay and phase rotation caused by the
  wave's travel time and interactions with surfaces.

For the specific case of **Massive MIMO** discussed in TransNet, the Base Station
(BS) has a massive number of antennas (e.g., $N_t = 32$ or $64$), while the User
Equipment (UE) often has a single antenna ($N_r = 1$) or a small number.
The TransNet paper assumes a single-antenna user for simplicity, making the
channel at a single frequency a row vector
$\mathbf{h} \in \mathbb{C}^{1 \times N_t}$.

## 2.2 Orthogonal Frequency Division Multiplexing (OFDM): The Third Dimension

Modern cellular systems (4G LTE, 5G NR) do not transmit on a single frequency.
They use **OFDM**, a modulation scheme that divides the available bandwidth into
hundreds or thousands of parallel, orthogonal sub-carriers.

This is critical for the data structure. The channel is **frequency-selective**,
meaning the complex value $h$ changes depending on the frequency of the
sub-carrier. A deep fade might obliterate the signal at $2.401$ GHz while the
signal at $2.405$ GHz remains strong.

Consequently, the CSI is not a vector, but a **matrix (or tensor)** with dimensions
defined by space and frequency:
$\tilde{\mathbf{H}} \in \mathbb{C}^{N_c \times N_t}$

- $N_c$ (**Sub-carriers**): The frequency dimension (e.g., $1024$ sub-carriers).
- $N_t$ (**Antennas**): The spatial dimension (e.g., $32$ antennas).

To a Computer Vision engineer, this data structure is analogous to a
single-channel **grayscale image** with resolution $1024 \times 32$.
However, unlike a natural image where pixels are real-valued integers,
these "pixels" are complex floating-point numbers.
TransNet treats this complex-valued matrix $\tilde{\mathbf{H}}$ as the raw
input "image" that must be compressed.

## 2.3 The Physics of Multipath: Why the Matrix is Structured

If the entries of $\tilde{\mathbf{H}}$ were independent and identically distributed
(i.i.d.) random variables, compression would be impossible
(Shannon’s Source Coding Theorem). However, wireless channels exhibit extreme
structure and correlation due to **multipath propagation**.

When a signal is transmitted, it does not follow a single path. It radiates in
all directions, reflecting off buildings, diffracting around corners, and
scattering off foliage. The receiver observes the superposition of these multiple
copies (rays) of the transmitted signal.

Key physical phenomena responsible for structure in $\tilde{\mathbf{H}}$ are:

- **Scattering Clusters:** In realistic environments, there are only a finite
  number of dominant reflectors (e.g., a few high-rise buildings). These create
  clusters of signal energy rather than uniformly random paths.

- **Angle of Arrival (AoA):** Since signals reflect from specific objects, they
  arrive at the antenna array from specific angles. This induces strong
  correlation across the spatial dimension $N_t$.

- **Delay Spread:** Because different paths have different lengths, signals
  arrive at different times. This induces correlation across the frequency
  dimension $N_c$ through the properties of the Fourier Transform.

As a result, although the matrix $\tilde{\mathbf{H}}$ appears dense (non-zero
everywhere), it is in fact a **low-rank representation** of a sparse physical
reality (few reflectors, few dominant delays). This underlying sparsity is the
fundamental property that makes **deep learning–based CSI compression and
feedback** feasible.


## 3. Massive MIMO: The System Architecture

The **Massive MIMO** component of the paper’s title refers to the technological
context that exacerbates the data problem to the point where AI-based solutions
become necessary.

### 3.1 Definition and the Beamforming Imperative

Massive MIMO (Multiple-Input Multiple-Output) involves equipping the Base Station
(BS) with a very large array of antennas—orders of magnitude more than previous
systems. While traditional MIMO systems might use $2$ or $4$ antennas, Massive
MIMO systems commonly employ $32$, $64$, $128$, or more antennas.

The primary capability enabled by this hardware is **digital beamforming**.
By carefully manipulating the amplitude and phase of the signal transmitted
from each of the $N_t$ antennas, the BS exploits interference patterns to focus
energy into a narrow beam directed precisely at a target user.

- **Constructive interference:** Signals add coherently at the user’s location,
  resulting in high signal strength.
- **Destructive interference:** Signals cancel out at other spatial locations,
  reducing interference to unintended users.

This spatial focusing enables **spatial multiplexing**, allowing the BS to reuse
the same time and frequency resources for multiple users simultaneously, as long
as they are separated in angle.

### 3.2 The Dependency on Accurate CSI

Here lies the core challenge: beamforming is mathematically impossible without
accurate knowledge of the channel matrix $\mathbf{H}$.

To steer a beam toward a user, the BS computes a **precoding matrix**
$\mathbf{W}$ based on $\mathbf{H}$. For instance, in **Zero-Forcing (ZF)**
precoding, the BS effectively inverts the channel to pre-cancel the distortion
introduced by the propagation environment.

- If the BS has perfect knowledge of $\mathbf{H}$, the resulting beam is
  sharply focused and achievable data rates are maximized.
- If the BS relies on outdated, noisy, or excessively compressed CSI, the beam
  becomes misaligned. Transmit energy is wasted, and interference leaks to other
  users, severely degrading system performance.

As a result, the BS must acquire up-to-date **Channel State Information (CSI)**
for every user and every sub-carrier, and refresh it every few milliseconds—
within the channel’s **coherence time**—before user mobility causes the channel
to change.

### 3.3 The Dimensionality Curse of Massive MIMO

In the era of two-antenna systems, feeding channel values back to the Base Station
(BS) was trivial. In Massive MIMO, however, the data volume grows explosively.

- **Dimensions:** $N_t = 32$ antennas $\times$ $1024$ sub-carriers
  $= 32{,}768$ complex channel coefficients.
- **Precision:** $2$ floating-point values per coefficient (real and imaginary).
- **Total payload:** Approximately $65{,}000$ floating-point values per user.
- **Update rate:** Every $\approx 5$–$10$ ms.

This results in a raw data throughput requirement on the order of **megabits per
second per user** solely for control signaling. This **feedback overhead**
consumes valuable uplink bandwidth that would otherwise be used for user data
(e.g., uploading images or videos).

As the number of antennas scales to the hundreds, this overhead becomes
prohibitively large. Consequently, practical deployment of Massive MIMO is
impossible without **drastic compression** of CSI feedback.
**TransNet** is explicitly designed to provide this level of compression.

## 4. The Duplexing Dilemma: FDD vs. TDD

The paper explicitly considers an **FDD Massive MIMO system**. This distinction
is fundamental rather than incidental—if the system were TDD, TransNet would be
largely unnecessary.

### 4.1 Time Division Duplex (TDD) and Channel Reciprocity

In **Time Division Duplex (TDD)** systems, the uplink (UE → BS) and downlink
(BS → UE) operate on the same frequency band but are separated in time.

Because electromagnetic wave propagation obeys the principle of **reciprocity**,
the physical channel from point A to B is identical to the channel from B to A
when the frequency is the same.

- **TDD advantage:** The BS estimates the uplink channel
  $\mathbf{H}_{UL}$ by receiving pilot signals transmitted by the UE.
- **Inference:** By reciprocity, the BS assumes
  $\mathbf{H}_{DL} \approx \mathbf{H}_{UL}$.
- **Result:** The BS acquires downlink CSI without explicit feedback from the UE,
  resulting in minimal overhead.

### 4.2 Frequency Division Duplex (FDD) and the Blind Transmitter

In **Frequency Division Duplex (FDD)** systems, the uplink and downlink operate
on separate frequency bands, often separated by tens or hundreds of MHz, enabling
simultaneous transmission.

In this case, reciprocity breaks down. Small-scale fading effects are highly
frequency-dependent. For example, a carrier at $2.1$ GHz may experience
constructive interference at a given location, while a carrier at $2.2$ GHz
experiences destructive interference due to the change in wavelength
($\lambda = c / f$).

- **The gap:** $\mathbf{H}_{DL} \neq \mathbf{H}_{UL}$.
- **Implication:** Knowledge of the uplink channel provides little information
  about the instantaneous downlink channel phases.

As a result, the BS is effectively **blind** to the downlink channel and cannot
estimate it directly.

### 4.3 The Explicit Feedback Loop

In FDD systems, a closed-loop feedback mechanism is unavoidable:

- **Pilot transmission:** The BS transmits reference signals (pilots) on the
  downlink frequency band.
- **Estimation:** The UE receives these pilots and estimates the dense downlink
  channel matrix $\mathbf{H}_{DL}$.
- **Compression (TransNet):** The UE compresses this high-dimensional matrix
  into a compact bitstream.
- **Feedback:** The UE sends this bitstream back to the BS over the uplink.
- **Reconstruction:** The BS decodes the bitstream to obtain an approximation
  $\hat{\mathbf{H}}_{DL}$ for beamforming.

Most cellular spectrum worldwide is allocated using FDD (paired frequency
bands). Consequently, enabling Massive MIMO in FDD systems is considered a
**“holy grail” problem** in wireless communications. Traditional codebook-based
methods fail to scale to large antenna arrays, motivating **deep learning–based
CSI feedback approaches** such as **TransNet**.

## 5. Signal Processing Transformations: The Angular-Delay Domain

Before the CSI matrix is fed into the neural network, the TransNet paper (and the
standard literature it builds upon) applies a critical domain transformation.
Understanding this preprocessing is essential because it defines what features
the neural network actually sees.

### 5.1 The 2D-DFT Transformation

The raw channel matrix $\tilde{\mathbf{H}}$ in the spatial–frequency domain is
dense, with energy spread across all antennas and sub-carriers. From the
perspective of a compression algorithm, this resembles high-entropy noise.

To expose the underlying structure, the system applies a two-dimensional
Discrete Fourier Transform (DFT):

$\mathbf{H} = \mathbf{F}_c \tilde{\mathbf{H}} \mathbf{F}_t^H$

where:

- $\tilde{\mathbf{H}}$ is the raw $N_c \times N_t$ channel matrix.
- $\mathbf{F}_c$ is the $N_c \times N_c$ DFT matrix applied along the frequency
  dimension.
- $\mathbf{F}_t$ is the $N_t \times N_t$ DFT matrix applied along the spatial
  (antenna) dimension.
- $\mathbf{H}$ is the transformed matrix in the **angular-delay domain**.

### 5.2 Deciphering the New Axes

This transformation rotates the channel into a basis where the signal becomes
sparse and highly structured.

#### 5.2.1 The Angular Domain (Virtual Angles)

The spatial DFT $\mathbf{F}_t$ transforms the antenna domain into the **angular
domain**, often referred to as beamspace.

- **Physical meaning:** Each column index corresponds to a specific Angle of
  Arrival (AoA) or Angle of Departure (AoD).
- **Sparsity intuition:** Signals do not arrive uniformly from all directions.
  Instead, they originate from a small number of discrete angles associated with
  dominant scatterers (e.g., a building at $45^\circ$, a tree at $-10^\circ$).
- **Result:** Most columns are nearly zero, while only a few columns contain
  significant energy. This phenomenon is known as **angular sparsity**.

#### 5.2.2 The Delay Domain (Time Taps)

The frequency DFT $\mathbf{F}_c$ transforms the frequency domain into the
**delay (time) domain**.

- **Physical meaning:** Each row index corresponds to a specific propagation
  delay. Row $0$ represents the shortest-delay path (typically line-of-sight),
  while higher rows correspond to reflected paths arriving later.
- **Sparsity intuition:** Wireless channels exhibit a finite maximum delay
  spread. Multipath echoes decay rapidly and do not persist indefinitely.
- **Result:** Most of the channel energy is concentrated in the first few rows,
  while rows corresponding to large delays contain mostly noise. This is known
  as **delay sparsity**.

### 5.3 Truncation ($N_a$) and Dimensionality Reduction

The TransNet framework explicitly exploits delay sparsity to reduce the input
dimensionality before neural network processing.

- **Observation:** Almost all significant non-zero components are concentrated
  within the first $N_a$ rows of the angular-delay matrix.
- **Action:** The matrix is truncated by retaining only the first $N_a$ rows and
  discarding rows $N_a + 1$ through $N_c$.
- **Data size reduction:** The dimensionality decreases from
  $N_c \times N_t$ (e.g., $1024 \times 32$) to
  $N_a \times N_t$ (e.g., $32 \times 32$).

The resulting truncated matrix $\mathbf{H}_a$ serves as the input to TransNet.
It represents a zoomed-in view of the dominant energy clusters in the
angular-delay domain. From a machine learning perspective, this corresponds to a
$32 \times 32$ image with two channels (real and imaginary components).


# Transformer architecture for transnet

![image](https://hackmd.io/_uploads/B1RhDoPBZg.png)


![image](https://hackmd.io/_uploads/By5B8sDBZg.png)

### Detailed Analysis of the Other Layers in Fig. 2

We now provide a detailed analysis of the remaining layers in Fig. 2.
Note that the shape of $\mathbf{H}_a$ is $N_a \times N_t$, and $\mathbf{H}_a$ is a
complex-valued matrix. In TransNet, the real and imaginary parts of
$\mathbf{H}_a$ are separated and then reformatted into a real-valued matrix of
size $2N_a \times N_t$.

Suppose the input of the TransNet encoder is a $2N_a \times N_t$ real-valued
matrix $\mathbf{X}$. The matrix $\mathbf{X}$ is first fed into the multi-head
attention layer of encoder\#1, where it is multiplied by weight matrices
$\mathbf{W}_n^Q$, $\mathbf{W}_n^K$, and $\mathbf{W}_n^V$ as

- $\mathbf{X}\mathbf{W}_n^Q = \mathbf{Q}_n$
- $\mathbf{X}\mathbf{W}_n^K = \mathbf{K}_n$
- $\mathbf{X}\mathbf{W}_n^V = \mathbf{V}_n$

where $\mathbf{W}_n^Q$, $\mathbf{W}_n^K$, and $\mathbf{W}_n^V \in
\mathbb{R}^{N_t \times (d_{\text{model}}/2)}$, and $n = 1, 2$. This means that
two sets of corresponding matrices $\mathbf{Q}_n$, $\mathbf{K}_n$, and
$\mathbf{V}_n$ are generated. In Transformer terminology, this corresponds to
using two attention heads, and $d_{\text{model}}$ denotes the model dimension
of TransNet.

The multi-head attention layer uses $\mathbf{Q}_n$, $\mathbf{K}_n$, and
$\mathbf{V}_n$ together with the softmax function to compute attention matrices
as

$\mathbf{A}_n = \mathbf{Q}_n \mathbf{K}_n^T / \sqrt{d_{\text{model}}/2}
= (a_{i,j|n}) \in \mathbb{R}^{2N_a \times 2N_a}$,

where $a_{i,j|n}$ denotes the element of $\mathbf{A}_n$ in the $i$-th row and
$j$-th column. The softmax-normalized attention matrix is defined as

$\text{Atten}(\mathbf{A}_n) = (b_{i,j|n})
= \left( \frac{e^{a_{i,j|n}}}{\sum_{m=1}^{2N_a} e^{a_{i,m|n}}} \right)
\in \mathbb{R}^{2N_a \times 2N_a}$,

where $b_{i,j|n}$ denotes the element of $\text{Atten}(\mathbf{A}_n)$ in the
$i$-th row and $j$-th column.

Then $\text{Atten}(\mathbf{A}_n)$ and $\mathbf{V}_n$ are used to compute
$\mathbf{Z}_n$ as

$\mathbf{Z}_n = \text{Atten}(\mathbf{A}_n)\mathbf{V}_n
\in \mathbb{R}^{2N_a \times (d_{\text{model}}/2)}$.

Since $n = 1, 2$, the matrices $\mathbf{Z}_1$ and $\mathbf{Z}_2$ are combined
into a new block matrix, which is then multiplied by a weight matrix
$\mathbf{W}^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ to yield

$\mathbf{Z} = [\mathbf{Z}_1 : \mathbf{Z}_2]\mathbf{W}^O
\in \mathbb{R}^{2N_a \times d_{\text{model}}}$.

transnet uses $d_{\text{model}} = 32$.

Intuitively, this process means that for each row $\mathbf{x}_i$ (with
$i = 1, 2, \ldots, 2N_a$) of the input $\mathbf{X}$, the multi-head attention
layer learns normalized attention weights indicating how much attention should
be paid to every other row $\mathbf{x}_j$ (with $j = 1, 2, \ldots, 2N_a$) of
$\mathbf{X}$.

The input $\mathbf{X}$ and output $\mathbf{Z}$ of the multi-head attention layer
are combined via skip connections in the add-and-normalize layer shown in
Fig. 2. This add-and-norm operation improves convergence and facilitates gradient
flow during training. The output of the first add-and-norm layer is then passed
to the feed-forward layer.

Similarly, the input and output of the feed-forward layer are combined through
skip connections in the second add-and-norm layer.

The decoder layers follow a similar structure to the encoder layers, except
that the masked multi-head attention layers in the decoder prevent information
leakage by employing a masking mechanism. The compressed information $\mathbf{Y}$
is reshaped into a $64 \times 32$ real-valued matrix and sent to the FC layer of
the TransNet decoder.

As shown in Fig. 2, $\mathbf{Y}$ serves as the input to the first decoder layer
and is used to construct $\mathbf{K}_n$ and $\mathbf{V}_n$ (for $n = 1, 2$) of
the multi-head attention layers in decoder\#1 and decoder\#2.

During reconstruction, $\mathbf{Y}$ is first fed into the masked multi-head
attention layer of decoder\#1 to construct $\mathbf{Q}_n$, $\mathbf{K}_n$, and
$\mathbf{V}_n$ (for $n = 1, 2$). The output of this layer is combined with
$\mathbf{Y}$ via the add-and-norm layer, and the result is then passed to the
multi-head attention layer to generate $\mathbf{Q}_n$ (for $n = 1, 2$).

The output of the multi-head attention layer in decoder\#1 undergoes the same
processing steps as in the encoder. The output of decoder\#1 is then used as the
input to the masked multi-head attention layer of decoder\#2. Finally, the
output of decoder\#2 is passed through the linear layer and softmax layer of the
TransNet decoder, yielding the reconstructed channel matrix $\hat{\mathbf{H}}_a$.


## TransNet Decoder Mechanism Explanation

### 1. Role of $\mathbf{Y}$ in Constructing Attention Matrices

The matrix $\mathbf{Y}$, obtained by reshaping the compressed vector $\mathbf{v}$,
plays two distinct roles within the TransNet decoder, depending on the specific
attention sub-layer.

#### Cross-Attention (Multihead Attention Layer)

In both **Decoder\#1** and **Decoder\#2**, $\mathbf{Y}$ serves as a stable
**global context**. It is used to construct the **Key** and **Value** matrices
for the multihead attention layers. This design ensures that, even as the signal
propagates deeper into the decoder, it continues to reference the original
compressed information produced by the encoder.

Formally, following standard Transformer notation:

- $\mathbf{K}_n = \mathbf{Y}\mathbf{W}_n^K$
- $\mathbf{V}_n = \mathbf{Y}\mathbf{W}_n^V$

for $n = 1, 2$ attention heads.

#### Self-Attention (Masked Multihead Attention Layer)

In **Decoder\#1**, $\mathbf{Y}$ also serves as the **initial input** to the
masked multihead attention layer. In this sub-layer, $\mathbf{Y}$ is used to
construct all three attention components:

- $\mathbf{Q}_n = \mathbf{Y}\mathbf{W}_n^Q$
- $\mathbf{K}_n = \mathbf{Y}\mathbf{W}_n^K$
- $\mathbf{V}_n = \mathbf{Y}\mathbf{W}_n^V$

This self-attention step allows the decoder to learn **internal correlations**
within the received compressed features before interacting with the global
context again.

---

### 2. Data Flow Between Decoder Sub-Layers

The transition between the masked self-attention layer and the cross-attention
layer follows the standard Transformer **Query–Key–Value interaction logic**.

#### Refining the Query ($\mathbf{Q}$)

The output of the masked multihead attention layer, denoted as $\mathbf{Z}$, is
combined with its input via an **Add & Norm** operation. The normalized result
represents a refined internal state of the decoder.

This refined representation is then fed into the subsequent multihead attention
layer to construct the **Query** matrices:

- $\mathbf{Q}_n = \text{AddNorm}(\mathbf{Z}) \mathbf{W}_n^Q$

#### Querying the Global Context

The refined Query $\mathbf{Q}_n$ interacts with the Key $\mathbf{K}_n$ and Value
$\mathbf{V}_n$ matrices derived from the original $\mathbf{Y}$:

- $\mathbf{K}_n = \mathbf{Y}\mathbf{W}_n^K$
- $\mathbf{V}_n = \mathbf{Y}\mathbf{W}_n^V$

This mechanism enables the decoder’s current state to **query the original
compressed representation** $\mathbf{Y}$ and selectively retrieve the
information required for accurate channel reconstruction.

In essence, the decoder alternates between:
- **self-refinement** (via masked self-attention), and
- **contextual retrieval** (via cross-attention to $\mathbf{Y}$),

which together drive the progressive reconstruction of the CSI matrix.

----

## 1. The NMSE Metric Used

The paper defines the **Normalized Mean Squared Error (NMSE)** metric in
Equation (12). NMSE measures how accurately the recovered Channel State
Information (CSI) matrix matches the original one.

The metric is defined as  
$\text{NMSE} = E\left\{ \frac{\lVert \mathbf{H}_a - \hat{\mathbf{H}}_a(\Theta_{\mathcal{C}}, \Theta_{\mathcal{R}}) \rVert_2^2}{\lVert \mathbf{H}_a \rVert_2^2} \right\}$.

Here:

- $\mathbf{H}_a$: The original truncated CSI matrix in the angular-delay domain.
- $\hat{\mathbf{H}}_a$: The reconstructed CSI matrix after encoder compression
  and decoder recovery.
- $\lVert \cdot \rVert_2^2$: The squared Euclidean (Frobenius) norm, representing
  the total energy of a matrix.
- Numerator $\lVert \mathbf{H}_a - \hat{\mathbf{H}}_a \rVert_2^2$: The **error
  power**, i.e., the energy of the reconstruction error.
- Denominator $\lVert \mathbf{H}_a \rVert_2^2$: The **signal power** of the
  original channel.
- $E\{\cdot\}$: Expectation over the dataset (average over all test samples).

In simple terms, NMSE computes the ratio of **error energy** to **signal energy**,
averaged over the entire dataset.

---

## 2. Why NMSE Is Reported in Decibels (dB)

While the NMSE formula produces a linear value (e.g., $0.01$ or $0.001$), the
results in Table I are reported in decibels (dB), such as $-29.22$ dB or
$-32.38$ dB.

The conversion is given by  
$\text{NMSE}_{\text{dB}} = 10 \log_{10}(\text{NMSE}_{\text{linear}})$.

This logarithmic representation is used for several standard reasons in wireless
communications:

- **Wide dynamic range:** CSI reconstruction errors can span several orders of
  magnitude (e.g., from $10^{-1}$ to $10^{-4}$). Expressing NMSE in dB allows
  meaningful comparison across this wide range.
- **Engineering convention:** Metrics such as SNR, path loss, and fading gains
  are traditionally expressed in dB. Reporting NMSE in dB aligns it with common
  communication-system performance metrics.
- **Interpretability:**
  - $0$ dB means the error power equals the signal power (poor reconstruction).
  - Negative values indicate the error power is smaller than the signal power.
  - More negative values imply better reconstruction accuracy.

For example, an NMSE of $-30$ dB indicates that the reconstruction error energy
is approximately $1/1000$ of the original signal energy, which corresponds to
very high recovery fidelity.

In the indoor scenario with compression ratio $\eta = 1/4$, TransNet achieves an
NMSE of $-29.22$ dB, demonstrating significa
