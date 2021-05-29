import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .StatResult import StatResult
from .utils import assert_is_in, raise_expected_got


def stat_test(
    box_data1,
    box_data2,
    test,
    comparisons_correction=None,
    num_comparisons=1,
    **stats_params
    ):
    """Get formatted result of two sample statistical test.

    Arguments
    ---------
    bbox_data1, bbox_data2
    test: str
        Statistical test to run. Must be one of:
        - `Levene`
        - `Mann-Whitney`
        - `Mann-Whitney-gt`
        - `Mann-Whitney-ls`
        - `t-test_ind`
        - `t-test_welch`
        - `t-test_paired`
        - 'bootstrap'
        - 'paired_bootstrap'
        - `Wilcoxon`
        - `Kruskal`
        comparisons_correction: str or None, default None
        Method to use for multiple comparisons correction. Implements
        statsmodels mutliple comparisons methods and bootstrapped comparisons.
    num_comparisons: int, default 1
        Number of comparisons to use for multiple comparisons correction.
    stats_params
        Additional keyword arguments to pass to scipy stats functions.

    Returns
    -------
    StatResult object with formatted result of test.

    """
    # Check arguments.
    assert_is_in(
        comparisons_correction,
        ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky', None],
        label='argument `comparisons_correction`',
    )

    # Switch to run scipy.stats hypothesis test.
    if test == 'Levene':
        stat, pval = stats.levene(box_data1, box_data2, **stats_params)
        result = StatResult(
            'Levene test of variance', 'levene', 'stat', stat, pval
        )
    elif test == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='two-sided', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test two-sided',
            'M-W',
            'U_stat',
            u_stat,
            pval,
        )
    elif test == 'Mann-Whitney-gt':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='greater', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test greater',
            'Mann-Whitney',
            'U_stat',
            u_stat,
            pval,
        )
    elif test == 'Mann-Whitney-ls':
        u_stat, pval = stats.mannwhitneyu(
            box_data1, box_data2, alternative='less', **stats_params
        )
        result = StatResult(
            'Mann-Whitney-Wilcoxon test smaller',
            'Mann-Whitney',
            'U_stat',
            u_stat,
            pval,
        )
    elif test == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test independent samples', 't-test_ind', 'stat', stat, pval
        )
    elif test == 't-test_welch':
        stat, pval = stats.ttest_ind(
            a=box_data1, b=box_data2, equal_var=False, **stats_params
        )
        result = StatResult(
            'Welch\'s t-test independent samples',
            't-test_welch',
            'stat',
            stat,
            pval,
        )
    elif test == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2, **stats_params)
        result = StatResult(
            't-test paired samples', 't-test_rel', 'stat', stat, pval
        )
    elif test == 'Wilcoxon':
        zero_method_default = len(box_data1) <= 20 and "pratt" or "wilcox"
        zero_method = stats_params.get('zero_method', zero_method_default)
        print("Using zero_method ", zero_method)
        stat, pval = stats.wilcoxon(
            box_data1, box_data2, zero_method=zero_method, **stats_params
        )
        result = StatResult(
            'Wilcoxon test (paired samples)', 'Wilcoxon', 'stat', stat, pval
        )
    elif test == 'Kruskal':
        stat, pval = stats.kruskal(box_data1, box_data2, **stats_params)
        test_short_name = 'Kruskal'
        result = StatResult(
            'Kruskal-Wallis paired samples', 'Kruskal', 'stat', stat, pval
        )
    elif test == 'bootstrap':
        n = 100000
        print('bootstrapping...')
        pval = bootstrap_diff(box_data1, box_data2, n)
        test_short_name = 'Bootstrap'
        result = StatResult(
            'Non-parametric bootstrapped two-sided comparison', test_short_name, 'stat', n, pval
        )
    elif test == 'paired_bootstrap':
        n = 100000
        print('bootstrapping...')
        pval = paired_bootstrap(box_data1, box_data2, n)
        test_short_name = 'paired_bootstrap'
        result = StatResult(
            'Non-parametric paired bootstrap two-sided comparison', test_short_name, 'stat', n, pval
        )
    else:
        result = StatResult(None, '', None, None, np.nan)
    return result

def bootstrap_diff(dataset_a, dataset_b, n, stat='mean'):
    '''determine via bootstrap if mean of dataset a is different than mean of dataset b'''
    boot_a = np.array([np.mean(np.random.choice(dataset_a, size=len(dataset_a), replace=True)) for _ in range(n)])
    boot_b = np.array([np.mean(np.random.choice(dataset_b, size=len(dataset_b), replace=True)) for _ in range(n)])

    return min(np.mean(boot_a > boot_b), np.mean(boot_a < boot_b))*2  # two-sided

def paired_bootstrap(dataset_a, dataset_b, n):
    '''determine via bootstrap of size n if items from one dataset are higher or lower
    than items from another paired dataset a significantly distinct proportion of the time.'''
    if len(dataset_a) != len(dataset_b):
        print ('Paired bootstrap tests can only be performed between two variables of equal size.')
        raise ValueError

    diffs = dataset_a.values - dataset_b.values
    samples = np.array([np.sum(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n)])
    return min(np.mean(samples > 0), np.mean(samples < 0))*2  # two-sided

def multiple_comparisons(p_values, method):
    '''
    Given an array of p-values and one of the specified methods of adjusting for mulitple comparisons,
    this function applies statsmodels' multiple tests module to those p-values.
    Returns boolean array of p-values that were accepted (True) or rejected (False), and
    array of p-values adjusted for multiple comparisons'''
    # Input checks.
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )
    p_values = np.atleast_1d(p_values)

    reject, corrected_p_values, _, _ = multipletests(p_values, method=method)
    if 'fdr' in method:
        corrected_p_values = corrected_fdr_pvals(p_values, reject)

    return reject, corrected_p_values


def corrected_fdr_pvals(pvals, rejects, alpha=0.05):
    """
    Statsmodels' fdr adjustments do not alter the p-values given, but only determine
    if a p-value should be rejected or not.

    Given an array of pvalues and a boolean array of which values were rejected,
    this calculates the corrected pvalue (pvalue/highest_valid_pvalue_threshold)*alpha,
    but in a slightly more efficient manner (pvalue*((1-based index of highest valid pvalue)*1/# of comparisons))
    """
    sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, sortind)
    ecdfactor = 1/len(pvals) * np.arange(1, len(pvals)+1)

    if len(np.nonzero(rejects[sortind])[0]) != 0:
        rejectmax = max(np.nonzero(rejects[sortind])[0])
        pvals_corrected_raw = pvals_sorted / ecdfactor[rejectmax]

        return pvals_corrected_raw[np.argsort(sortind)]
    else:
        return np.ones(len(pvals))