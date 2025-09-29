import time, sys, importlib, optuna, torch, pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from train_runner import train_model
from graph_utils import GraphWithCond

#optuna settings
DATASET_PATH = "graph_dataset_MAY_25.pt"
BATCH_SIZE   = 32
N_TRIALS     = 40
COND_DIM     = 217
IN_CHANNELS  = 3
OUT_CHANNELS = 3
HIDDEN_DIM   = 64
LATENT_DIM   = 64
RANDOM_STATE = 42

# the classes of modules are in models.py 
MODELS_FILE  = "models"   # 
ARCH_MAP = {
    "GCN": "GCN",                 # class GCN in models.py
    "GAT": "GAT",                 # class GAT in models.py
    "Transformer": "Transformer"  # class Transformer in models.py
}

def load_data():
    dataset = torch.load(DATASET_PATH)
    print(f"✅ Loaded dataset with {len(dataset)} graphs")
    tr, va = train_test_split(dataset, test_size=0.2, random_state=RANDOM_STATE)
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(va, batch_size=BATCH_SIZE)
    )

def make_model(arch_class, device, hidden_dim, latent_dim):
    return arch_class(
        in_channels=IN_CHANNELS,
        cond_dim=COND_DIM,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        out_channels=OUT_CHANNELS
    ).to(device)

def build_objective(arch_class, device, train_loader, val_loader):
    def objective(trial):
        # tune training/loss + (optionally) model size
        num_epochs    = trial.suggest_int("num_epochs", 60, 160)
        bond_w        = trial.suggest_float("bond_loss_weight", 1.0, 1e4, log=True)
        angle_w       = trial.suggest_float("angle_weight", 1.0, 1e4, log=True)
        kl_weight_max = trial.suggest_float("kl_weight_max", 1e-3, 1.0)
        hidden_dim    = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        latent_dim    = trial.suggest_categorical("latent_dim", [16, 32, 64])

        model = make_model(arch_class, device, hidden_dim, latent_dim)

        val_loss, bond_err, angle_err = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            bond_loss_weight=bond_w,
            angle_weight=angle_w,
            kl_weight_max=kl_weight_max,
        )

        trial.set_user_attr("bond_error", bond_err)
        trial.set_user_attr("angle_error", angle_err)
        return val_loss
    return objective

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Device:", device)

    # import the single module that contains all classes
    models_mod = importlib.import_module(MODELS_FILE)

    train_loader, val_loader = load_data()
    summary_rows = []

    for name, class_name in ARCH_MAP.items():
        print(f"\n🔄 Optimizing: {name}")
        arch_class = getattr(models_mod, class_name)

        study = optuna.create_study(direction="minimize")
        t0 = time.time()
        study.optimize(
            build_objective(arch_class, device, train_loader, val_loader),
            n_trials=N_TRIALS
        )
        elapsed = time.time() - t0
        print(f"✅ {name} done in {elapsed:.1f}s")

        best = study.best_trial
        print(f"🏆 {name} best val loss: {best.value:.6f}")
        print("   params:", best.params)
        print("   bond_err:", best.user_attrs.get("bond_error"))
        print("   angle_err:", best.user_attrs.get("angle_error"))

        # per-arch CSV
        df = study.trials_dataframe()
        csv_path = f"optuna_trials_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved {csv_path}")

        summary_rows.append({
            "model": name,
            "best_val_loss": best.value,
            "best_params": best.params,
            "bond_error": best.user_attrs.get("bond_error"),
            "angle_error": best.user_attrs.get("angle_error"),
            "elapsed_sec": round(elapsed, 1),
        })

    pd.DataFrame(summary_rows).to_csv("optuna_summary_all.csv", index=False)
    print("\n📈 Wrote optuna_summary_all.csv")

if __name__ == "__main__":
    main()
