# Predict the suitability of each candidate
predictions = clf.predict_proba(X_test_tfidf)

# Rank the candidates based on their predicted suitability scores
ranked_candidates = sorted(zip(predictions, data['candidate_id']), key=lambda x: x[0][1], reverse=True)

# Print the ranked candidates
for candidate in ranked_candidates:
    print('Candidate ID:', candidate[1], 'Suitability Score:', candidate[0][1])
