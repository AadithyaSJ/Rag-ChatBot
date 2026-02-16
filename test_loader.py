from data_loader import load_data

details, summary, points, batting, bowling = load_data()

print("Details shape:", details.shape)
print("Summary shape:", summary.shape)
print("Points shape:", points.shape)
print("Batting shape:", batting.shape)
print("Bowling shape:", bowling.shape)
