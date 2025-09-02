
from soccer_event.train import train_loop
from soccer_event.config import DataCfg, ModelCfg, TrainCfg

if __name__ == "__main__":
    home = "/mnt/data/tracking_home.csv"
    away = "/mnt/data/tracking_away.csv"
    events = "/mnt/data/events.csv"

    data_cfg = DataCfg(window_s=3.2, stride_s=0.8, det_tol_s=0.5)
    model_cfg = ModelCfg(d_in=11, d_model=128, heads=4, layers=2, temp_layers=4)
    train_cfg = TrainCfg(batch_size=2, lr=1e-3, weight_decay=1e-2, max_steps=40)

    train_loop(home, away, events, data_cfg, model_cfg, train_cfg)
