import torch

bundle = torch.load("artifacts/dataset.pt", map_location="cpu")

# 어떤 키들이 들어있는지 확인
print(bundle.keys())

# type_map (이벤트 타입 라벨 매핑)
print("type_map:", bundle["type_map"])

# 홈/어웨이 등번호 확인
print("home_ids:", bundle["home_ids"])
print("away_ids:", bundle["away_ids"])

# 토큰 텐서 하나 확인
x0 = bundle["tokens"][0]
print("tokens[0].shape:", x0.shape)   # [T, N, F]
print("예: T(시간), N(엔티티 수), F(피쳐 수)")
print("det[0].shape:", bundle["det"][0].shape)
print("type[0]:", bundle["type"][0])
print("from[0]:", bundle["from"][0])
print("to[0]:", bundle["to"][0])
