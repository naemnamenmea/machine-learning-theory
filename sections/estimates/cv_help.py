# counting CV for alpha
alpha_range = np.arange(0.2, 4., 0.2)
alpha_scores = []
for k in alpha_range:
    model.alpha = k
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    alpha_scores.append(scores.mean())

plt.figure()
plt.plot(alpha_range, alpha_scores)
plt.xlabel('Value of alpha for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()