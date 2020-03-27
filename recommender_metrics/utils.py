from tqdm import tqdm

__all__ = [
    'rank_dataframe',
    'verbose_iterator',
]


def rank_dataframe(
        df,
        group_col='group_id',
        score_col='score',
        ascending=False,
        sort_group_rank=True,
        from_zero=True
):
    """
    This function takes a dataframe, groups it according to the `group_col` parameter,
    and within each group ranks it according to the `score_col` parameter. The rank order
    is populated into a new field of the dataframe based on string formatting of
    f'{score_col}_ranked'. If this column is already in `df` it will be overwritten.

    Parameters
    ----------
    df : Pandas Dataframe
        This dataframe is required to be made from the two columns specified in the next
        two arguments.

    group_col : str, option (default='group_id')
        The column name of `df` that is used in groupby

    score_col : str, optional (default='score')
        The column name of `df` that is used for ranking

    ascending : bool, optional (default=False)
        Whether to sort each group in ascending or descending order. In search result data,
        set `ascending=True` since lower search positions are better. With ranking scores
        typically larger scores define a better afinity between a user-item pair.

    sort_group_rank : bool, optional (default=True)
        Whether the dataframe should be re-ordered by the `group_col` and `ranked_col`. If
        this function is being used for evaluating performance (as in from `./metrics.py`)
        this parameter should be `True`

    from_zero : bool, optional (default=True)
        This parameter specifies whether the rank order column will start from 0 or 1. When
        this parameter is True, rank order goes from 0, when it's False it goes from 1.

    Returns
    -------
    ranked_dataframe : pandas Dataframe
        The original dataframe with the new rank order column

    ranked_col : str
        The name of the added column
    """

    assert group_col in df
    assert score_col in df

    ranked_col = f'{score_col}_ranked'

    # pd.Series.rank orders from 1; this variable allows the ranks to be 0-indexed
    from_zero = 1 if from_zero else 0

    ranked_df = df.assign(**{
        ranked_col: df.groupby(group_col)[score_col].apply(
            lambda score: score.rank(
                method='first',  # How to manage ties in scores
                ascending=ascending  # Whether larger scores are better
            ).astype(int) - from_zero
        )
    })

    if sort_group_rank:
        ranked_df = ranked_df.sort_values(
            by=[group_col, ranked_col],
        )

    return ranked_df, ranked_col


def verbose_iterator(iterator, total=None, desc=None, verbose=True):
    if verbose:
        return tqdm(iterator, total=total, desc=desc)
    return iterator
