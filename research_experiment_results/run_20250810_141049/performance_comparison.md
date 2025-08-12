
# Algorithm Performance Comparison

| Algorithm | Win Rate | Key Characteristics | Robustness Score |
|-----------|----------|-------------------|------------------|
| FSPPPO | 64.1% | High opponent diversity, historical sampling | High |
| scripted_random | 49.0% | Unpredictable, exploration-based | Medium |
| scripted_seek | 35.4% | Aggressive, goal-directed | Medium |
| IPPO | 22.4% | Minimal diversity, independent learning | Low |
| SPPPO | 14.4% | Zero diversity, pure self-play | Very Low |

## Key Insights:
- **186% performance improvement** with opponent diversity (FSPPPO vs SPPPO)
- **Simple scripted behaviors often outperform learned policies**
- **Excessive conservatism** in learned algorithms (high draw rates)
- **Generalization gap** when facing unseen opponents
