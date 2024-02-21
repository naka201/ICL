import torch

# 与えられた角度（度数）
theta_radians = torch.tensor([0.5913, 0.5024, 0.5519])

# 角度をラジアンに変換
theta_degrees = torch.rad2deg(theta_radians)

# 結果を表示
print("角度（度数）:", theta_degrees)
print("角度（ラジアン）:", theta_radians)
