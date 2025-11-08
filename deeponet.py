# -*- coding: utf-8 -*-
"""
DeepONet: I(1100~3000A), t(0~250s) -> T(z in 0..L)

V6 (Final Version with Fair Visualization):
- Visualization logic now correctly compares the model against the FEM ground truth from the validation set.
- Model loading in visualization is robust to hyperparameter changes.
- Added separate epoch controls for optimization vs. final training for efficiency.
- N_z is correctly set based on FEM data.
"""

import os, sys, json, csv, math, random, time, argparse, contextlib, bisect
from pathlib import Path

try:
    import optuna
except ImportError:
    print("Optuna not found. Please install it with 'pip install optuna' to use the --mode optimize feature.")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import scienceplots
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
except Exception:
    import matplotlib
    import matplotlib.pyplot as plt
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")

# ===================== Utils & Constants =====================
def human_time(sec: float) -> str:
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

@contextlib.contextmanager
def timer(msg: str):
    t0 = time.time(); print(f"[{msg}] ..."); yield; print(f"[{msg}] done in {human_time(time.time()-t0)}")

def set_global_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

# --- NOTE: N_z is now 401 to match your FEM data ---
L = 0.2; N_z = 401; rho_material = 4500.0; T_env_C = 25.0
r_inner = 0.018; r_outer = 0.020; A_cross = np.pi * (r_outer**2 - r_inner**2)
epsilon = 0.7; sigma_SB = 5.670374419e-8; END_ZONE_LEN = 0.03
H_END_ZONE    = 1500.0; dt_const = 0.05006
torch_dtype_pde = torch.float64
z_coords_numpy = np.linspace(0.0, L, N_z, dtype=np.float32)
T_MIN, T_MAX = 25.0, 1200.0 # Define temperature range globally

k_data = np.array([[100,11.7],[200,13.0],[300,13.4],[400,14.2],[500,14.7],[600,16.3],[700,17.6],[800,19.3],[900,20.9]], dtype=np.float64)
cp_data = np.array([[25,532],[100,560],[200,595],[300,640],[400,670],[500,684],[600,692],[700,715],[800,880]], dtype=np.float64)
resistivity_data = np.array([[20,1.25],[100,1.31],[200,1.42],[300,1.52],[400,1.60],[500,1.66],[600,1.71],[700,1.74],[800,1.74],[900,1.75]], dtype=np.float64)
resistivity_data[:,1] /= 1e6
k_torch = torch.from_numpy(k_data).to(torch_dtype_pde)
cp_torch = torch.from_numpy(cp_data).to(torch_dtype_pde)
resistivity_torch = torch.from_numpy(resistivity_data).to(torch_dtype_pde)

# ===================== PDE Core (FDM - Kept for data generation if needed) =====================
def interp1d_tensor(x, xp, fp):
    xp = xp.contiguous(); xp_col = xp[:,0].contiguous(); x0, x1 = xp_col[0], xp_col[-1]; x = torch.clamp(x, x0, x1)
    idx = torch.searchsorted(xp_col, x); idx = torch.clamp(idx, 1, len(xp)-1)
    x_lo, x_hi = xp[idx-1,0], xp[idx,0]; y_lo, y_hi = fp[idx-1], fp[idx]
    slope = (y_hi - y_lo) / (x_hi - x_lo).clamp_min(1e-9); return y_lo + slope * (x - x_lo)
def get_k(T_c):  return interp1d_tensor(T_c, k_torch, k_torch[:,1])
def get_cp(T_c): return interp1d_tensor(T_c, cp_torch, cp_torch[:,1])
def get_resistivity(T_c): return interp1d_tensor(T_c, resistivity_torch, resistivity_torch[:,1])
def heat_transfer_coefficient(z, L, h0_val):
    h0_val = float(h0_val); h = torch.full_like(z, h0_val, dtype=torch_dtype_pde)
    if END_ZONE_LEN > 0.0:
        left_mask  = (z <= float(END_ZONE_LEN)); right_mask = (z >= float(L - END_ZONE_LEN))
        h[left_mask | right_mask] = float(H_END_ZONE)
    return h
def thomas_solve(a,b,c,d):
    n = d.shape[-1]; ac, bc, cc, dc = a.clone(), b.clone(), c.clone(), d.clone()
    for i in range(1,n): w = ac[i-1] / bc[i-1]; bc[i] -= w * cc[i-1]; dc[i] -= w * dc[i-1]
    x = torch.empty_like(dc); x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1): x[i] = (dc[i] - cc[i]*x[i+1]) / bc[i]
    return x
@torch.no_grad()
def simulate_to_time_return_snapshot(t_target_s: float, I_amp: float, h0: float = 82.0, T0_C: float = T_env_C, h_end: float = 500.0, pde_device="cpu"):
    z = torch.linspace(0.0, L, N_z, dtype=torch_dtype_pde, device=pde_device); dz = L / (N_z - 1); steps = max(1, int(round(t_target_s / dt_const))); dt = dt_const
    Tn_C = torch.full((N_z,), float(T0_C), dtype=torch_dtype_pde, device=pde_device); S_over_V = (2.0 * r_outer) / (r_outer**2 - r_inner**2); J = float(I_amp) / A_cross
    T_env_K = float(T_env_C + 273.15); a = torch.zeros(N_z-1, dtype=torch_dtype_pde, device=pde_device); b = torch.zeros(N_z, dtype=torch_dtype_pde, device=pde_device); c = torch.zeros(N_z-1, dtype=torch_dtype_pde, device=pde_device)
    efficiency_factor = 0.90
    for _ in range(steps):
        Tn_K = Tn_C + 273.15; k_dyn  = get_k(Tn_C); cp_dyn = get_cp(Tn_C); rho_e  = get_resistivity(Tn_C); alpha  = k_dyn / (rho_material * cp_dyn)
        q_joule = (J**2) * rho_e * efficiency_factor; h_side  = heat_transfer_coefficient(z, L, h0); q_conv  = h_side * (Tn_K - T_env_K) * S_over_V
        q_rad   = epsilon * sigma_SB * (Tn_K**4 - T_env_K**4) * S_over_V; q_net   = q_joule - (q_conv + q_rad); r = alpha * (dt / (dz**2))
        b.fill_(1.0); b[1:-1] = 1.0 + 2.0 * r[1:-1]; a[:] = -r[1:]; c[:] = -r[:-1]; rhs = Tn_C + (dt / (rho_material * cp_dyn)) * q_net
        Bi_L = float(h_end * dz / k_dyn[0].item()); b[0]   = 1.0 + 2.0 * r[0] * (1.0 + Bi_L); c[0]   = -2.0 * r[0]; rhs[0] += 2.0 * r[0] * Bi_L * T_env_C
        Bi_R = float(h_end * dz / k_dyn[-1].item()); b[-1]   = 1.0 + 2.0 * r[-1] * (1.0 + Bi_R); a[-1]   = -2.0 * r[-1]; rhs[-1] += 2.0 * r[-1] * Bi_R * T_env_C
        Tn_C = thomas_solve(a,b,c,rhs); Tn_C = torch.clamp(Tn_C, T_MIN, T_MAX)
    return Tn_C

# ===================== Model & Dataset =====================
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
class ITSnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        super().__init__(); self.data_dir = Path(data_dir); self.shards = sorted(self.data_dir.glob("shard_*.pt"))
        if not self.shards: raise ValueError(f"No shard_*.pt in {self.data_dir}")
        self.cum_sizes = []; total = 0
        for f in self.shards: total += torch.load(f, map_location="cpu")["I"].shape[0]; self.cum_sizes.append(total)
        self.total = total; self.cache = {}
    def __len__(self): return self.total
    def _locate(self, idx):
        shard_id = bisect.bisect_right(self.cum_sizes, idx); start_base = 0 if shard_id == 0 else self.cum_sizes[shard_id-1]
        offset = idx - start_base; return shard_id, offset
    def __getitem__(self, idx):
        shard_id, offset = self._locate(idx)
        if shard_id not in self.cache: self.cache[shard_id] = torch.load(self.shards[shard_id], map_location="cpu")
        m = self.cache[shard_id]; I_amp = m["I"][offset].float(); t_tar = m["t"][offset].float(); Tz = m["Tz"][offset].float()
        Tz_normalized = (Tz - T_MIN) / (T_MAX - T_MIN); return I_amp, t_tar, Tz_normalized

# ===================== Data Generation (FDM-based) =====================
def generate_and_save_data(num_samples, data_dir, h0, seed, shard_size=2000):
    data_path = Path(data_dir); data_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating {num_samples} samples into '{data_path}' (shard_size={shard_size}) ...")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    I_samples = np.random.uniform(1100.0, 3000.0, size=num_samples).astype(np.float32)
    t_samples = np.random.uniform(0.0,   250.0, size=num_samples).astype(np.float32)
    shard_idx = 0
    for start in tqdm(range(0, num_samples, shard_size), desc=f"Generating shards for '{data_path.name}'"):
        end = min(start + shard_size, num_samples); I_batch = I_samples[start:end]; t_batch = t_samples[start:end]
        T_list = [simulate_to_time_return_snapshot(float(t), float(I), h0=h0, pde_device="cpu").unsqueeze(0) for I, t in zip(I_batch, t_batch)]
        shard = {"I": torch.from_numpy(I_batch), "t": torch.from_numpy(t_batch), "Tz": torch.cat(T_list, dim=0).to(torch.float32)}
        torch.save(shard, data_path / f"shard_{shard_idx:05d}.pt"); shard_idx += 1
    print(f"Saved {shard_idx} shards in {data_path}")

# ===================== Evaluation =====================
def evaluate(model, loader, criterion, device, desc="val"):
    model.eval(); mse_sum, cnt = 0.0, 0
    with torch.no_grad():
        for I_b, t_b, T_b in tqdm(loader, desc=f"[Eval:{desc}]", leave=False):
            I_b, t_b, T_b = I_b.to(device), t_b.to(device), T_b.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                loss = criterion(model(I_b, t_b), T_b)
            mse_sum += loss.item() * I_b.shape[0]; cnt += I_b.shape[0]
    return mse_sum / max(cnt, 1)

# ===================== Training & Visualization =====================
def train_model(cfg):
    set_global_seed(42, deterministic=False); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}");
    if torch.cuda.is_available(): print(f"[CUDA] {torch.cuda.get_device_name(0)}")
    
    effective_batch = cfg['batch_size'] * cfg['accum_steps']
    # Use a default for base_lr if it's not found (e.g., when not running from optimize)
    lr = cfg.get('base_lr', 2e-3) * (effective_batch / 32)
    
    train_ds = ITSnapshotDataset(data_dir=Path(cfg['data_dir']) / "train")
    val_ds = ITSnapshotDataset(data_dir=Path(cfg['data_dir']) / "val")
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=(device.type=="cuda"), persistent_workers=(cfg['num_workers']>0))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=(device.type=="cuda"), persistent_workers=(cfg['num_workers']>0))
    print(f"[Data] train={len(train_ds)}, val={len(val_ds)}, batch={cfg['batch_size']}, accum={cfg['accum_steps']}, eff_batch={effective_batch}, LR={lr:.2e}")
    
    model = DeepONetTemp(
        z_coords_numpy,
        n_frequencies=cfg.get('n_frequencies', 32),
        trunk_width=cfg.get('trunk_width', 128),
        branch_width=cfg.get('branch_width', 128)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    last_ckpt, best_ckpt = output_dir / "last.pth", output_dir / "best_model.pth"
    
    # Save a copy of the final config used for this run
    cfg_to_save = cfg.copy()
    cfg_to_save['data_dir'] = str(cfg_to_save['data_dir'])
    cfg_to_save['output_dir'] = str(cfg_to_save['output_dir'])
    with open(output_dir / "config_final.json", "w") as f:
        json.dump(cfg_to_save, f, indent=2)

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_mse", "lr", "epoch_time_sec"])
            
    start_ep, best_val = 1, float("inf")
    if last_ckpt.exists():
        print(f"[Resume] Loading {last_ckpt}"); state = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optim"]); scheduler.load_state_dict(state["sched"])
        start_ep = state["epoch"] + 1; best_val = state.get("best_val", best_val)
        
    print(f"[Model] parameter count = {sum(p.numel() for p in model.parameters())}"); t_total0 = time.time()
    patience, bad_epochs = cfg.get('early_stop_patience', 20), 0
    
    for ep in range(start_ep, cfg['epochs'] + 1):
        t_ep0 = time.time(); model.train(); running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        
        lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
        pbar = tqdm(train_loader, desc=f"[Train] Epoch {ep}/{cfg['epochs']} | lr={lr_str}", leave=False)
        
        for step, (I_b, t_b, T_b) in enumerate(pbar, 1):
            I_b, t_b, T_b = I_b.to(device), t_b.to(device), T_b.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                loss = criterion(model(I_b, t_b), T_b)
            scaler.scale(loss / cfg['accum_steps']).backward()
            
            if step % cfg['accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            running_loss += loss.item() * I_b.shape[0]; seen += I_b.shape[0]
            pbar.set_postfix({"avg_loss": f"{(running_loss/max(seen,1)):.4f}"})
            
        scheduler.step()
        ep_time = time.time() - t_ep0
        train_avg = running_loss / max(seen, 1)
        
        should_eval = (ep % cfg['eval_every'] == 0) or (ep == 1) or (ep == cfg['epochs'])
        val_mse_str = ""
        if should_eval:
            val_mse = evaluate(model, val_loader, criterion, device, desc=f"epoch {ep}")
            val_mse_str = f"{val_mse:.6f}"
            if val_mse < best_val:
                best_val = val_mse; torch.save(model.state_dict(), best_ckpt)
                print(f"⭐⭐⭐ New best model saved with val_MSE={best_val:.6f} ⭐⭐⭐"); bad_epochs = 0
            else:
                bad_epochs += 1
                
        print(f"[Epoch {ep:4d}] train_loss={train_avg:.6f} | val_MSE={val_mse_str or 'N/A'} | best_val={best_val:.6f} | time={human_time(ep_time)}")
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{train_avg:.6f}", val_mse_str, lr_str, f"{ep_time:.2f}"])
            
        torch.save({
            "epoch": ep, "model": model.state_dict(), "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(), "best_val": best_val
        }, last_ckpt)
        
        if patience > 0 and bad_epochs >= patience:
            print(f"[EarlyStop] No improvement for {patience} evals. Stop.")
            break
            
    print(f"\n[Train] Total time: {human_time(time.time()-t_total0)}, best val MSE={best_val:.6f}")
    
    # --- MODIFICATION: Pass val_ds to visualization for a fair comparison ---
    if best_ckpt.exists():
        visualize_results(best_ckpt, device, Path(cfg['output_dir']), cfg, val_ds)

# ===================== NEW FAIR VISUALIZATION FUNCTION =====================
def visualize_results(model_path, device, output_dir, cfg, val_dataset):
    print("\nLoading best model for a FAIR visualization against FEM data...")
    
    # Load the model that was trained with the optimal hyperparameters
    model = DeepONetTemp(
        z_coords_numpy, 
        n_frequencies=cfg.get('n_frequencies', 32), 
        trunk_width=cfg.get('trunk_width', 128), 
        branch_width=cfg.get('branch_width', 128)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Randomly select a sample from the validation set for comparison
    sample_idx = random.randint(0, len(val_dataset) - 1)
    I_test_val, t_test_val, T_true_normalized = val_dataset[sample_idx]
    
    # Move the single sample data to the device and add a batch dimension (B=1)
    I_in = I_test_val.unsqueeze(0).to(device)
    t_in = t_test_val.unsqueeze(0).to(device)

    print(f"Visualizing for a random sample from validation set: I={I_in.item():.1f}A, t={t_in.item():.1f}s")

    # Perform inference with the model
    with torch.no_grad():
        pred_T_normalized = model(I_in, t_in).detach().cpu()

    # --- Un-normalize to get the real temperature values ---
    # Model's prediction
    pred_T_real = pred_T_normalized.numpy() * (T_MAX - T_MIN) + T_MIN
    # The ground truth from the FEM data
    T_true_real = T_true_normalized.numpy() * (T_MAX - T_MIN) + T_MIN
    
    # Plotting
    try:
        plt.style.use(["science", "ieee"])
    except Exception:
        pass
        
    plt.figure(figsize=(8, 5))
    plt.plot(z_coords_numpy, pred_T_real[0], 'k-', linewidth=2, label='DeepONet Predict (Best)')
    plt.plot(z_coords_numpy, T_true_real, 'r--', linewidth=2, label='FEM Reference (Ground Truth)')
    
    plt.xlabel("Location z (m)", fontsize=12)
    plt.ylabel("Temperature T (°C)", fontsize=12)
    plt.title(f"I={I_in.item():.0f}A, t={t_in.item():.0f}s", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the figure with a new, more descriptive name
    fig_path = output_dir / "prediction_vs_FEM_truth.png"
    plt.savefig(fig_path, dpi=300)
    
    if os.environ.get("DISPLAY"):
        plt.show()
        
    print(f"Saved FAIR comparison plot to {fig_path}")

# ===================== Hyperparameter Optimization =====================
def objective(trial, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        'n_frequencies': trial.suggest_categorical('n_frequencies', [16, 32, 64]),
        'trunk_width': trial.suggest_categorical('trunk_width', [64, 128, 256]),
        'branch_width': trial.suggest_categorical('branch_width', [64, 128, 256]),
        'base_lr': trial.suggest_float('base_lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    }
    model = DeepONetTemp(
        z_coords_numpy,
        n_frequencies=params['n_frequencies'],
        trunk_width=params['trunk_width'],
        branch_width=params['branch_width']
    ).to(device)
    
    effective_batch = args.batch_size * args.accum_steps
    lr = params['base_lr'] * (effective_batch / 32)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_epochs)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    
    train_ds = ITSnapshotDataset(data_dir=args.data_dir / "train")
    val_ds   = ITSnapshotDataset(data_dir=args.data_dir / "val")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for ep in range(1, args.opt_epochs + 1):
        model.train()
        for I_b, t_b, T_b in train_loader:
            I_b, t_b, T_b = I_b.to(device), t_b.to(device), T_b.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                loss = criterion(model(I_b, t_b), T_b)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        if ep % args.eval_every == 0:
            val_mse = evaluate(model, val_loader, criterion, device, desc=f"trial {trial.number} ep {ep}")
            trial.report(val_mse, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
    return evaluate(model, val_loader, criterion, device, "final")

def run_optimization(args):
    if 'optuna' not in sys.modules:
        print("Cannot run optimization, Optuna is not installed.")
        return
        
    study_name = "deeponet-study"
    storage_path = Path(args.output_dir) / "optuna_study.db"
    storage = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"Starting Optuna study. Storage: {storage}")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print(f"Best trial value: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    
    best_params_path = Path(args.output_dir) / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"Best parameters saved to {best_params_path}")

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(description="DeepONet for Temperature Prediction - V6")
    parser.add_argument('--mode', type=str, required=True, choices=['generate', 'train', 'optimize'], help="Script mode.")
    # Data generation
    parser.add_argument('--num_train_samples', type=int, default=2000)
    parser.add_argument('--num_val_samples', type=int, default=200)
    parser.add_argument('--h0', type=float, default=10.0)
    parser.add_argument('--shard_size', type=int, default=1000)
    # Training & Optimization
    parser.add_argument('--epochs', type=int, default=1500, help="Epochs for FINAL training.")
    parser.add_argument('--opt_epochs', type=int, default=100, help="Epochs for EACH optimization trial.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials.")
    # Paths
    parser.add_argument('--data_dir', type=Path, default=Path("./pde_data"))
    parser.add_argument('--output_dir', type=Path, default=Path("./results"))
    args = parser.parse_args()

    if args.mode == 'generate':
        set_global_seed(123)
        generate_and_save_data(args.num_train_samples, args.data_dir / "train", args.h0, 123, args.shard_size)
        generate_and_save_data(args.num_val_samples, args.data_dir / "val", args.h0, 999, args.shard_size)
        
    elif args.mode == 'optimize':
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_optimization(args)
        
    elif args.mode == 'train':
        cfg = vars(args)
        best_params_path = args.output_dir / "best_params.json"
        if best_params_path.exists():
            print(f"Found best parameters file: {best_params_path}. Overriding CLI args.")
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
            cfg.update(best_params)
        else:
            print("No 'best_params.json' found. Using parameters from command line.")
        train_model(cfg)

if __name__ == "__main__":
    main()