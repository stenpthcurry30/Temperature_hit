# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:03:29 2025

@author: Drake
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:03:29 2025

@author: Drake
"""

# app.py (V2.1 - English Version)
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import time
from pathlib import Path
import sys  # <--- 新增导入
import os   # <--- 新增导入

# ++++++++++ 新增的函数 ++++++++++
def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和 PyInstaller 打包的 EXE """
    try:
        # PyInstaller 会创建一个临时文件夹并将其路径存储在 _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # 未打包状态，使用常规路径
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# ==============================================================================
# ===== 1. Model Definition and Constants (Unchanged) =====
# ==============================================================================

# --- Constants ---
L = 0.2; N_z = 401;
z_coords_numpy = np.linspace(0.0, L, N_z, dtype=np.float32)
T_MIN, T_MAX = 25.0, 1200.0

# --- Model Classes ---
class SinCosFourierFeatures(nn.Module):
    def __init__(self, n_frequencies: int):
        super().__init__(); self.register_buffer("w", torch.arange(1, n_frequencies+1, dtype=torch.float32).view(1,-1))
    def forward(self, x):
        wx = math.pi * x.float() @ self.w; return torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1)

class TrunkNet(nn.Module):
    def __init__(self, z_coords: np.ndarray, n_frequencies: int, trunk_width: int):
        super().__init__(); z = torch.from_numpy(z_coords).view(-1,1); self.register_buffer("z01", z / float(L))
        self.fourier = SinCosFourierFeatures(n_frequencies)
        self.net = nn.Sequential(nn.Linear(2*n_frequencies, trunk_width), nn.GELU(), nn.Linear(trunk_width, trunk_width), nn.GELU())
    def forward(self): return self.net(self.fourier(self.z01))

class BranchNet(nn.Module):
    def __init__(self, width: int, out_dim: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_dim))
    def forward(self, it): return self.net(it)

class DeepONetTemp(nn.Module):
    def __init__(self, z_coords: np.ndarray, n_frequencies=32, trunk_width=128, branch_width=128):
        super().__init__(); self.trunk = TrunkNet(z_coords, n_frequencies, trunk_width)
        self.branch = BranchNet(branch_width, trunk_width); self.bias = nn.Parameter(torch.zeros(len(z_coords), dtype=torch.float32))
    def forward(self, I_in, t_in):
        I01 = (I_in.float() - 1100.0) / (3000.0 - 1100.0); t01 = (t_in.float() / 250.0)
        it = torch.stack([I01, t01], dim=-1); w = self.branch(it); Phi = self.trunk()
        Y = (Phi @ w.T).T + self.bias; return torch.clamp(Y, 0.0, 1.0)

# ==============================================================================
# ===== 2. Core Functions (Unchanged) =====
# ==============================================================================

@st.cache_resource
def load_model_from_checkpoint(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_dir / "best_model.pth"
    config_path = model_dir / "config_final.json"
    if not model_path.exists() or not config_path.exists():
        st.error(f"Error: Could not find 'best_model.pth' or 'config_final.json' in '{model_dir}'.")
        return None, None
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    model = DeepONetTemp(z_coords_numpy, n_frequencies=cfg.get('n_frequencies', 32), trunk_width=cfg.get('trunk_width', 128), branch_width=cfg.get('branch_width', 128)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@torch.no_grad()
def get_predictions(model, device, current, max_time, time_steps):
    t_array = np.linspace(0, max_time, time_steps)
    I_tensor = torch.full((time_steps,), float(current), dtype=torch.float32, device=device)
    t_tensor = torch.from_numpy(t_array).float().to(device)
    all_preds_normalized = model(I_tensor, t_tensor)
    all_preds_real = all_preds_normalized.cpu().numpy() * (T_MAX - T_MIN) + T_MIN
    max_temps_curve = np.max(all_preds_real, axis=1)
    return t_array, all_preds_real, max_temps_curve

# ==============================================================================
# ===== 3. Streamlit UI Logic (Translated to English) =====
# ==============================================================================

st.set_page_config(page_title="DeepONet Temperature Prediction", layout="wide")
st.title("⚡️ DeepONet Transient Temperature Field Predictor V2.1")

model_directory = Path("./results")
model, device = load_model_from_checkpoint(model_directory)

if model:
    # --- Create two tabs, one for static prediction and one for animation ---
    tab1, tab2 = st.tabs(["📊 Static Chart Prediction", "🎬 Dynamic Evolution Animation"])

    # --- Tab 1: Static Prediction ---
    with tab1:
        st.header("Static Operating Point Prediction")
        col1_static, col2_static = st.columns([1, 3]) # Sidebar and main area

        with col1_static:
            st.subheader("⚙️ Parameter Input")
            current_static = st.slider("Current (A)", 1100, 3000, 1700, 10, key="current_static")
            time_static = st.slider("Time (s)", 0, 250, 20, 1, key="time_static")
            if st.button("🚀 Generate Static Charts", use_container_width=True):
                with st.spinner("Calculating..."):
                    t_curve, all_fields, max_curve = get_predictions(model, device, current_static, 250.0, 100)
                    time_index = np.abs(t_curve - time_static).argmin()
                    specific_field = all_fields[time_index, :]

                    fig1, ax1 = plt.subplots(figsize=(7, 4))
                    ax1.plot(z_coords_numpy, specific_field, 'b-', linewidth=2)
                    ax1.set_xlabel("Location z (m)"); ax1.set_ylabel("Temperature T (°C)")
                    ax1.set_title(f"Temperature Field Distribution @ I={current_static}A, t={time_static}s")
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.plot(t_curve, max_curve, 'r-', linewidth=2)
                    ax2.plot(t_curve[time_index], max_curve[time_index], 'bo', markersize=8, label=f'Current Point (t={time_static}s)')
                    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Maximum Temperature T_max (°C)")
                    ax2.set_title(f"Hottest Point Heating Curve @ I={current_static}A")
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    ax2.legend()
                    
                    with col2_static:
                        st.pyplot(fig1)
                        st.pyplot(fig2)

    # --- Tab 2: Dynamic Evolution Animation ---
    with tab2:
        st.header("Heating Pipe Temperature Field Animation")
        col1_anim, col2_anim = st.columns([1, 3])

        with col1_anim:
            st.subheader("⚙️ Animation Parameters")
            current_anim = st.slider("Current (A)", 1100, 3000, 2000, 10, key="current_anim")
            duration_anim = st.slider("Animation Duration (s)", 10, 250, 50, 10, key="duration_anim")
            speed_anim = st.select_slider("Animation Speed", options=['Slow', 'Medium', 'Fast', 'Real-time'], value='Medium')
            
            # Map speed to delay
            speed_map = {'Slow': 0.1, 'Medium': 0.05, 'Fast': 0.01, 'Real-time': 0}

            if st.button("🎬 Generate Animation", use_container_width=True):
                with st.spinner("Preparing animation data..."):
                    t_anim, all_fields_anim, _ = get_predictions(model, device, current_anim, duration_anim, 100)

                with col2_anim:
                    st.subheader("▶️ Real-time Evolution")
                    # Create placeholders to update content
                    plot_placeholder = st.empty()
                    stats_placeholder = st.empty()

                    # Prepare the plot
                    fig_anim, ax_anim = plt.subplots(figsize=(10, 2))
                    
                    # Broadcast 1D data to 2D for the heatmap
                    dummy_tube = np.broadcast_to(all_fields_anim[0, :], (5, N_z))
                    
                    # Fix color bar range based on the max temperature in the animation
                    vmin = T_MIN
                    vmax = np.max(all_fields_anim)

                    im = ax_anim.imshow(dummy_tube, aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
                    ax_anim.set_xlabel("Heating Pipe Location z (m)")
                    ax_anim.set_yticks([]) # Hide y-axis ticks
                    
                    # Add color bar
                    cbar = fig_anim.colorbar(im, ax=ax_anim, orientation='vertical')
                    cbar.set_label('Temperature (°C)')

                    # Loop to update each frame of the animation
                    for i, t in enumerate(t_anim):
                        frame_data = all_fields_anim[i, :]
                        
                        # Update stats
                        max_temp_frame = np.max(frame_data)
                        stats_placeholder.markdown(f"### Real-time Data: "
                                                   f"<span style='color:blue;'>Time = {t:.2f} s</span> | "
                                                   f"<span style='color:red;'>Max Temperature = {max_temp_frame:.1f} °C</span>",
                                                   unsafe_allow_html=True)
                        
                        # Update heatmap data
                        im.set_data(np.broadcast_to(frame_data, (5, N_z)))
                        
                        # Redraw the plot in the placeholder
                        plot_placeholder.pyplot(fig_anim)
                        
                        # Control animation speed
                        time.sleep(speed_map[speed_anim])
                    
                    st.success("Animation finished!")
                    plt.close(fig_anim) # Close the figure to free up memory

else:
    st.warning("Model could not be loaded. Please check if the 'results' directory path is correct.")
