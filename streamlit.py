import streamlit as st

st.set_page_config(page_title="Rohit Ganti – Research Showcase", layout="wide")

# ---- Header ----
st.title("Research Showcase")
st.write("Submission for **Applied Math Software Engineer – Windborne Systems**")
st.markdown("---")

# ---- Project 1: Angle-Proxy Heuristic for ACK ----
st.header("1️⃣ Angle-Proxy Method for Adaptive Circuit Knitting (ACK)")

st.markdown("""
**Problem Context**  
Distributed quantum circuit execution suffers from cross-talk and entanglement bottlenecks.  
The *Angle-Proxy* method introduces a heuristic estimator for entanglement cost, replacing
entropy-based metrics to identify optimal circuit cut points in large 2D systems.

**Mathematical Formulation**  
Given a Hamiltonian $H = H_x + g H_z + h H_y$, the heuristic defines  
$\\theta_i = \\arccos(|\\langle \\psi_i | \\psi_{i+1} \\rangle|)$ as a proxy for local entanglement.
Low-$\\theta$ bonds indicate candidate cuts minimizing the communication cost in ACK.

**Toolchain & Implementation**  
- Python 3, PennyLane + Qiskit + MPI4Py  
- qTEBD / DMRG back-end for ground-state references  
- Adaptive Circuit Knitting (ACK) framework with angle-threshold selection

**Computational Bottleneck**  
Evaluating mutual information scales as $O(N^2)$; Angle-Proxy reduces this to $O(N)$  
while maintaining ≥ 95 % fidelity to DMRG ground-state benchmarks.

**Results**
""")

col1, col2 = st.columns(2)
with col1:
    st.image("https://raw.githubusercontent.com/rohitganti/assets/main/angle_proxy_cutpoints.png",
             caption="Angle-Proxy cut-point visualization", use_container_width=True)
with col2:
    st.image("https://raw.githubusercontent.com/rohitganti/assets/main/energy_vs_fidelity.png",
             caption="Energy vs Fidelity convergence", use_container_width=True)

st.markdown("---")

# ---- Project 2: Diffusion Adapters for LLMs ----
st.header("2️⃣ Diffusion Adapters for Large Language Models (LLMs)")

st.markdown("""
**Problem Context**  
Fine-tuning large LLMs for domain alignment (e.g. Rego policies, CUDA-Q code) is expensive.
We introduce *Diffusion Adapters* — lightweight stochastic adapters trained via a denoising diffusion
objective, enabling controllable alignment with minimal compute overhead.

**Mathematical Formulation**  
Let $\\theta_t$ denote the adapter parameters at diffusion step $t$.
Training minimizes  
$\\mathcal{L} = \\mathbb{E}_{t,x,\\epsilon}\\big[\\|\\epsilon - \\epsilon_\\theta(x_t,t)\\|^2\\big]$  
where $x_t$ is the noised hidden representation of the LLM.

**Toolchain & Implementation**  
- PyTorch + Transformers + Accelerate  
- Adapter Fusion / LoRA layers trained under diffusion noise schedule  
- Synthetic instruction data generated via Nemotron-4 340B Instruct + Reward models

**Computational Bottleneck**  
Baseline full fine-tuning (340B parameters) exceeds 2× A100 80 GB nodes.  
Diffusion Adapters achieve 98 % performance using < 5 % of compute.

**Results**
""")

col3, col4 = st.columns(2)
with col3:
    st.image("https://raw.githubusercontent.com/rohitganti/assets/main/diffusion_loss_curve.png",
             caption="Diffusion Adapter training curve", use_container_width=True)
with col4:
    st.image("https://raw.githubusercontent.com/rohitganti/assets/main/alignment_comparison.png",
             caption="Adapter vs Full Fine-Tuning performance", use_container_width=True)

st.markdown("---")

# ---- Contact ----
st.header("📬 Contact & Links")
st.write("**Rohit Ganti**  ·  soundwave.rohit@gmail.com")
st.markdown(
    "[GitHub](https://github.com/soundwaverohit) | "
    "[LinkedIn](https://www.linkedin.com/in/rohit-ganti-64280a19b/) | "
    "[Resume (Google Drive)](https://drive.google.com/file/d/18jlHGEJ3TX_e53n6tCwPEX7BEhiS5En5/view?usp=sharing)"
)
