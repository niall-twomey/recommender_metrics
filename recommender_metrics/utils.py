__all__ = [
    'rank_dataframe'
]


def rank_dataframe(df, group_col='group_id', score_col='score', ascending=False, sort_group_rank=True, from_zero=True):
    assert group_col in df
    assert score_col in df

    ranked_col = f'{score_col}_ranked'

    # pd.Series.rank orders from 1; this variable allows the ranks to be 0-indexed
    sub = 1 if from_zero else 0

    ranked_df = df.assign(**{
        ranked_col: df.groupby(group_col)[score_col].apply(
            lambda score: score.rank(
                method='first',  # How to manage ties in scores
                ascending=ascending  # Whether larger scores are better
            ).astype(int) - sub
        )
    })

    if sort_group_rank:
        ranked_df = ranked_df.sort_values(
            by=[group_col, ranked_col],
        )

    return ranked_df, ranked_col
