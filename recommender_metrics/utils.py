__all__ = [
    'rank_dataframe'
]


def rank_dataframe(df, group='group_id', score_col='score', from_zero=True):
    sub = 1 if from_zero else 0
    return df.assign(
        score_rank=df.groupby(group)[score_col].apply(
            lambda score: score.rank(
                ascending=False,
                method='first'
            ).astype(int) - sub
        ))
