__all__ = [
    'rank_dataframe'
]


def rank_dataframe(df, group='group_id', score_col='score', sort_group_rank=True, from_zero=True):
    assert group in df
    assert score_col in df

    # pd.Series.rank orders from 1; this variable allows the ranks to be 0-indexed
    sub = 1 if from_zero else 0

    ranked_df = df.assign(
        score_rank=df.groupby(group)[score_col].apply(
            lambda score: score.rank(
                method='first',  # How to manage ties in scores
                ascending=False  # Larger is better
            ).astype(int) - sub
        ))

    if sort_group_rank:
        return ranked_df.sort_values(
            by=['group_id', 'score_rank'],
        )

    return ranked_df
