# 🤖 Autonomous Robot for Inventory Monitoring within SMEs

This project pairs a custom **YOLOv8** object-detection model trained on warehouse inventory data with a real-time **computer-vision perception → control** pipeline that extracts a navigation signal from raw camera frames and feeds it into a **Ziegler–Nichols-tuned PID controller**. The full stack runs as a set of **ROS nodes** on a custom TurtleBot3-style robot, achieving sub-5% overshoot and a settling time of under 1 second.

📹 **Full project walkthrough & live demo:** [https://www.youtube.com/watch?v=drLWpMeTv8o](https://www.youtube.com/watch?v=drLWpMeTv8o)

---

## 1. Perception as a Cascaded Signal-Extraction Pipeline

The robot navigates by following a colored guide line on the warehouse floor. Treat each camera frame as a discrete 2D signal

$$I_0 : \Omega \rightarrow \mathbb{Z}^3, \qquad \Omega = \{0,\dots,W-1\}\times\{0,\dots,H-1\}, \qquad I_0(x,y) = [B,G,R]^\top$$

which we model as a clean line signal $S$ corrupted by additive and impulsive noise (motion blur, uneven lighting, specular highlights, sensor grain):

$$I_0(x,y) = S(x,y) + N(x,y)$$

The goal of perception is to **demodulate** this noisy, high-dimensional field down to a single robust scalar control signal $e(t)$. This is done as a composition of operators, where the output of each stage is the conditioned input of the next:

$$e \;=\; \big(\mathcal{T}_M \circ \mathcal{T}_W \circ \mathcal{T}_\circ \circ \mathcal{T}_\Theta \circ \mathcal{T}_G \circ \mathcal{T}_{\text{HSV}}\big)\,(I_0)$$

### Stage 1 — Color-space transform $\mathcal{T}_{\text{HSV}}$

With normalized channels $R',G',B' \in [0,1]$, $C_{\max}=\max(R',G',B')$, $C_{\min}=\min(R',G',B')$, $\Delta = C_{\max}-C_{\min}$:

$$V = C_{\max}, \qquad S = \begin{cases}\Delta / C_{\max} & C_{\max}\neq 0\\ 0 & \text{otherwise}\end{cases}, \qquad H = 60^{\circ}\cdot \text{hue}(R',G',B')$$

**Why feed this forward:** projecting BGR onto HSV decorrelates *chroma* (H, S) from *luminance* (V). The line now occupies a compact, illumination-invariant region of the signal space, so the downstream threshold can be a fixed band that survives shadows and glare. This conditions the signal so Stage 3 needs only a static decision boundary.

### Stage 2 — Gaussian low-pass filter $\mathcal{T}_G$

$$I_2 = I_1 * G_\sigma, \qquad G_\sigma(x,y) = \frac{1}{2\pi\sigma^2}\,e^{-\frac{x^2+y^2}{2\sigma^2}}$$

**Why feed this forward:** convolution with $G_\sigma$ is a linear low-pass filter that attenuates the high-frequency content where sensor noise $N$ concentrates, raising the signal-to-noise ratio *before* the nonlinear thresholding step. Smoothing first prevents noise spikes from being frozen into the binary mask as isolated false pixels — i.e. it stops noise from propagating into a stage that cannot later remove it linearly.

### Stage 3 — Threshold / masking $\mathcal{T}_\Theta$

$$B(x,y) = \prod_{c \in \{H,S,V\}} \mathbb{1}\!\left[\theta_c^{-} \le I_2^{c}(x,y) \le \theta_c^{+}\right]$$

This is the `cv2.inRange` mask using the HSV bounds from the design document, acting as a **matched filter / segmenter** that returns a binary signal $B(x,y)\in\{0,1\}$.

**Why feed this forward:** because Stages 1–2 already made the line band compact and noise-suppressed, the indicator collapses the 3-channel field to a clean 1-bit-per-pixel signal with most energy belonging to the true line. Dimensionality drops from $\mathbb{Z}^3$ to $\{0,1\}$, which is exactly the representation the morphological and moment operators require.

### Stage 4 — Morphological opening $\mathcal{T}_\circ$

With structuring element $K$, opening is an erosion followed by a dilation:

$$(B \ominus K)(x,y) = \min_{(i,j)\in K} B(x-i,\,y-j), \qquad (B \oplus K)(x,y) = \max_{(i,j)\in K} B(x-i,\,y-j)$$

$$B_4 = (B \ominus K)\oplus K$$

**Why feed this forward:** opening is a nonlinear (rank) filter that deletes any connected component smaller than $K$ — precisely the residual impulsive noise that survived thresholding — while restoring the line to its original width and closing pinholes. It guarantees the next stage integrates over a single coherent blob rather than fragmented speckle, which is what makes the centroid estimate stable frame-to-frame.

### Stage 5 — ROI windowing $\mathcal{T}_W$

$$B_5(x,y) = B_4(x,y)\cdot w(x,y), \qquad w(x,y) = \mathbb{1}\!\left[y \ge y_0\right]$$

**Why feed this forward:** multiplying by a spatial window gates the signal to the lookahead band directly in front of the robot, rejecting far-field and off-path energy. This minimizes phase lag in the control loop — the error reflects where the robot is *about to be*, not distant track — which directly improves settling time and stability.

### Stage 6 — Moment demodulation $\mathcal{T}_M$

$$M_{pq} = \sum_{x}\sum_{y} x^{p} y^{q}\, B_5(x,y), \qquad c_x = \frac{M_{10}}{M_{00}}, \qquad c_y = \frac{M_{01}}{M_{00}}$$

**Why feed this forward:** the spatial summation is an integrator/estimator. Averaging over the $M_{00}$ line pixels suppresses any remaining zero-mean residual noise by a factor of roughly $\sqrt{M_{00}}$ (law of large numbers), collapsing the entire 2D field to one high-SNR scalar — the line centroid $c_x$.

### Output — the control error

$$e(t) = c_x - \frac{W}{2}$$

The **cross-track error** is the deviation of the extracted centroid from the optical center, and it is the single scalar handed to the controller.

### Why the cascade yields a high-performing system

Each operator is chosen so its output is the *ideal input domain* of the next, producing two monotonic trends along the chain:

- **Monotonically increasing SNR** — HSV isolates the band, Gaussian filtering removes additive noise, opening removes impulsive noise, and moment integration averages out the rest. Noise is attenuated by linear, nonlinear, and statistical means at successive stages, so it can never accumulate.
- **Monotonically decreasing dimensionality** — $\mathbb{Z}^3 \to \mathbb{Z}^3_{\text{HSV}} \to \{0,1\} \to \{0,1\}_{\text{clean}} \to \{0,1\}_{\text{ROI}} \to \mathbb{R}$. Information irrelevant to steering is discarded early and cheaply, keeping per-frame latency low.

Ordering matters: smoothing *before* thresholding prevents noise from being quantized into the mask; opening *before* moments guarantees a single coherent region to integrate; windowing *before* the centroid removes lag-inducing far-field data. The result is a clean, low-latency, high-SNR error signal — a well-conditioned input that lets the PID controller (Section 2) achieve <5% overshoot and <1 s settling time.

---

## 2. Control: Ziegler–Nichols-Tuned PID Controller

The cross-track error $e(t)$ drives a **PID controller** that commands the robot's angular velocity, steering the centroid back toward the image center.

### Control law

$$u(t) = K_p\, e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d \frac{de(t)}{dt}$$

Implemented in discrete time at the camera frame rate $\Delta t$:

$$u_k = K_p\, e_k + K_i \sum_{j=0}^{k} e_j\,\Delta t + K_d \frac{e_k - e_{k-1}}{\Delta t}$$

The output $u_k$ is published as the angular component (`angular.z`) of a ROS `Twist` message while a constant forward linear velocity (`linear.x`) is maintained.

### Ziegler–Nichols tuning

The gains were tuned using the **Ziegler–Nichols ultimate-gain method**:

1. Set $K_i = K_d = 0$ and increase the proportional gain until the closed loop exhibits **sustained, stable oscillation** at the ultimate gain $K_u$ with oscillation period $T_u$.
2. Derive the classic PID gains from $K_u$ and $T_u$:

| Gain | Formula |
|------|---------|
| $K_p$ | $0.6\,K_u$ |
| $K_i$ | $1.2\,K_u / T_u$ |
| $K_d$ | $0.075\,K_u\,T_u$ |

This tuning yields a **critically-damped response with <5% overshoot and a settling time under 1 second**, allowing the robot to track the line and correct its $x,y$ pose with high accuracy. Step-response and trajectory behavior were modelled and validated in **Matplotlib** to confirm the overshoot and settling-time targets before deployment.

---

## 3. System Architecture: ROS Nodes

The system is built on the **Robot Operating System (ROS)**, exploiting its modular publish/subscribe architecture for interoperability and reuse:

- **Camera node** — publishes raw frames to `/image_raw`.
- **Vision node** — subscribes to `/image_raw`, runs the six-stage signal-extraction cascade (Section 1), and publishes the cross-track error $e(t)$.
- **Control node** — subscribes to the error topic, runs the Ziegler–Nichols PID loop (Section 2), and publishes velocity commands to `/cmd_vel`.
- **Inventory / detection node** — runs the custom YOLOv8 model at defined stop positions (Section 4).

Decoupling perception, control, and detection into independent nodes makes the pipeline modular, testable, and portable across robot platforms.

---

## 4. Inventory Detection: Custom YOLOv8

At defined waypoints — triggered by visual cues such as a change in line color — the robot halts and runs a **custom YOLOv8 model** trained on warehouse inventory data to capture and report stock levels.

YOLOv8 uses a single-pass **convolutional neural network** that simultaneously regresses bounding boxes and class probabilities, refined with **Intersection-over-Union (IoU)** scoring and **Non-Maximum Suppression (NMS)**. The nano variant delivers real-time inference with a mAP roughly 33% higher than YOLOv5n, making it ideal for low-cost, on-robot deployment.

This gives SMEs accurate, consistent inventory insight **without expensive detection hardware** such as RFID or barcode scanners, and without the line-of-sight and printing limitations of QR-based approaches.

---

## Key Results

- **< 5%** system overshoot
- **< 1 second** settling time
- Real-time line-following control derived purely from a noisy monocular camera feed
- Low-cost inventory monitoring with no dedicated scanning hardware
