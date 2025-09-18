import numpy as np, collections as C
import pandas as pd
from dataset import prepare_features_and_labels, make_windows_from_frames
from train import train_with_tqdm
import torch, os, json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score

def load_id2name(event_type_map: dict | None = None, path="./artifacts/event_type_map.json"):
    if event_type_map is None:
        with open(path, "r") as f:
            name2id = json.load(f)
    else:
        name2id = event_type_map
    return {v: k for k, v in name2id.items()}

@torch.no_grad()
def predict_on_test(model, X_test, batch_size=256, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    y_pred_type, y_prob_type, y_pred_actor = [], [], []
    for s in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[s:s+batch_size]).float().to(device)
        pt, pa = model(xb)                                  # (B,n_types),(B,n_actors)
        p = torch.softmax(pt, dim=1).cpu().numpy()
        y_prob_type.append(p)
        y_pred_type.append(p.argmax(axis=1))
        y_pred_actor.append(pa.argmax(dim=1).cpu().numpy())
    return (np.concatenate(y_pred_type), 
            np.concatenate(y_prob_type), 
            np.concatenate(y_pred_actor))

def visualize_and_save(Y_type_test, Y_actor_test, y_pred_type, y_prob_type, y_pred_actor,
                       id2name: dict, no_event_idx: int = 0, out_dir="./artifacts", topk=50):
    os.makedirs(out_dir, exist_ok=True)
    n_types = len(id2name)
    # CSV 저장
    p_event = 1.0 - y_prob_type[:, no_event_idx]
    conf_pred = y_prob_type[np.arange(len(y_pred_type)), y_pred_type]
    df_pred = pd.DataFrame({
        "idx": np.arange(len(y_pred_type)),
        "y_true_id": Y_type_test,
        "y_true": [id2name[i] for i in Y_type_test],
        "y_pred_id": y_pred_type,
        "y_pred": [id2name[i] for i in y_pred_type],
        "p_event": p_event,
        "p_pred_class": conf_pred,
        "pred_actor": y_pred_actor,
        "true_actor": Y_actor_test,
    })
    df_pred.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # 혼동행렬 이미지
    cm = confusion_matrix(Y_type_test, y_pred_type, labels=np.arange(n_types))
    plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Type)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(np.arange(n_types), [id2name[i] for i in range(n_types)], rotation=45, ha="right")
    plt.yticks(np.arange(n_types), [id2name[i] for i in range(n_types)])
    plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cm_type.png"), dpi=160); plt.close()

    # Event vs NoEvent 지표
    y_true_bin = (Y_type_test != no_event_idx).astype(int)
    pr_auc = average_precision_score(y_true_bin, p_event)
    y_pred_bin = (p_event >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    print(f"[Event v NoEvent] PR-AUC={pr_auc:.4f} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # 타임라인(앞 3,000 포인트)
    show_len = min(3000, len(p_event))
    plt.figure(figsize=(10,3))
    plt.plot(np.arange(show_len), p_event[:show_len])
    plt.title("Event probability timeline (first segment)")
    plt.xlabel("Test index (window end)"); plt.ylabel("P(Event)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timeline_event_prob.png"), dpi=160); plt.close()

    # Top-K 이벤트 후보 프레임
    top_idx = np.argsort(-p_event)[:topk]
    (df_pred.loc[top_idx, ["idx","y_true","y_pred","p_event","p_pred_class","true_actor","pred_actor"]]
            .sort_values("p_event", ascending=False)
            .to_csv(os.path.join(out_dir, f"top{topk}_event_frames.csv"), index=False))

if __name__ == '__main__':
    data_file_path = "./artifacts/final_model_data.parquet"
    df = pd.read_parquet(data_file_path)
    print("Data loaded successfully.")

    frames_ordered, event_type_map, ytype_by_frame, yactor_by_frame, feat_cols = prepare_features_and_labels(df, include_is_actor=True)
    # 윈도우 만들기 (125프레임=5초 @25fps)
    (train_set, val_set, test_set) = make_windows_from_frames(
        frames_ordered, ytype_by_frame, yactor_by_frame,
        seq_len=125, include_is_actor=True, feature_cols=feat_cols, mask_last_is_actor=True,
        train_ratio=0.7, val_ratio=0.15
    )

    (X_train, Y_type_train, Y_actor_train) = train_set
    (X_val,   Y_type_val,   Y_actor_val)   = val_set
    (X_test,  Y_type_test,  Y_actor_test)  = test_set

    model = train_with_tqdm(
        X_train, Y_type_train, Y_actor_train,
        X_val,   Y_type_val,   Y_actor_val,
        X_test,  Y_type_test,  Y_actor_test,
        event_type_map
    )
     # 체크포인트 로드(안전)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = "./checkpoints/best_model.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # id->name 매핑 준비 (메모리의 event_type_map을 그대로 써도 OK)
    id2name = load_id2name(event_type_map)   # 또는 load_id2name(None, "./artifacts/event_type_map.json")

    # 테스트셋 예측
    y_pred_type, y_prob_type, y_pred_actor = predict_on_test(model, X_test, batch_size=256, device=device)

    # 시각화/저장
    visualize_and_save(
        Y_type_test, Y_actor_test,
        y_pred_type, y_prob_type, y_pred_actor,
        id2name=id2name, no_event_idx=0, out_dir="./artifacts", topk=50
    )
