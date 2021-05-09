#Taken from https://github.com/webermarcolivier/statannot/blob/master/statannot/statannot.py

from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib import transforms, lines
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
from seaborn.utils import remove_na


from scipy import stats
from statsmodels.stats.multitest import multipletests

DEFAULT = object()

def raise_expected_got(expected, for_, got, error_type=ValueError):
    """Raise a standardized error message.
    Raise an `error_type` error with the message
        Expected `expected` for `for_`; got `got` instead.
    Or, if `for_` is `None`,
        Expected `expected`; got `got` instead.
    """
    if for_ is not None:
        raise error_type(
            'Expected {} for {}; got {} instead.'.format(expected, for_, got)
        )
    else:
        raise error_type(
            'Expected {}; got {} instead.'.format(expected, got)
        )

class StatResult:
    def __init__(self, test_str, test_short_name, stat_str, stat, pval):
        self.test_str = test_str
        self.test_short_name = test_short_name
        self.stat_str = stat_str
        self.stat = stat
        self.pval = pval
        self.pval_formatted = None
        self._format_pval()

    def _format_pval(self):
        if self.pval < 0.001:
            self.pval_formatted = '{:.2e}'.format(self.pval)
            if 'strap' in self.test_short_name:
                self.pval_formatted = '{:.1e}'.format(self.pval)
        else:
            self.pval_formatted = '{:.2g}'.format(self.pval)
        if 'strap' in self.test_short_name and self.pval == 0:
            if self.stat < 0.001:
                self.pval_formatted = '< {:.2e}'.format(1/self.stat)
            else:
                self.pval_formatted = '< {:.1g}'.format(1/self.stat)

    def set_pval(self, pval):
        self.pval = pval
        self._format_pval()

    @property
    def formatted_output(self):
        if self.stat_str is None and self.stat is None:
            stat_summary = '{}, P_val:{}'.format(self.test_str, self.pval_formatted)
        else:
            stat_summary = '{}, P_val={} {}={:.2e}'.format(
                self.test_str, self.pval_formatted, self.stat_str, self.stat
            )
        return stat_summary

    def __str__(self):
        return self.formatted_output

def assert_is_in(x, valid_values, error_type=ValueError, label=None):
    """Raise an error if x is not in valid_values."""
    if x not in valid_values:
            raise_expected_got('one of {}'.format(valid_values), label, x)

def stat_test(
    box_data1,
    box_data2,
    test,
    mult_comp_correction=None,
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
    mult_comp_correction: str or None, default None
        Method to use for multiple comparisons correction.
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
        mult_comp_correction,
        ['bonferroni', 'sidak','holm-sidak','holm','simes-hochberg','hommel','fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky',None],
        label='argument `mult_comp_correction`',
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
    # Optionally, run multiple comparisons correction.
    return result

def bootstrap_diff(dataset_a, dataset_b, n, stat='mean'):
    '''determine via bootstrap if mean of dataset a is different than mean of dataset b'''
    boot_a = np.array([np.mean(np.random.choice(dataset_a, size=len(dataset_a), replace=True)) for _ in range(n)])
    boot_b = np.array([np.mean(np.random.choice(dataset_b, size=len(dataset_b), replace=True)) for _ in range(n)])
    # if np.mean(boot_a) > np.mean(boot_b):
    # print (len(dataset_a), len(dataset_b))
    return min(np.mean(boot_a > boot_b), np.mean(boot_a < boot_b))*2 #two-sided

def paired_bootstrap(dataset_a, dataset_b, n):
    '''determine via bootstrap if items from one dataset are higher or lower
    than items from another paired dataset a significantly distinct proportion of the time.'''
    if len(dataset_a) != len(dataset_b):
        print ('Paired bootstrap tests can only be done between two variables of equal size.')
        raise ValueError
    # sample_idxs = np.random.choice(0, len(dataset_a), size=n, replace=True)
    diffs = dataset_a.values-dataset_b.values
    # print(dataset_b.v)
    # diff_sum = sum(dataset_a-dataset_b)
    samples = np.array([np.sum(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n)])
    return min(np.mean(samples > 0), np.mean(0 > samples))*2 #two-sided

def corrected_fdr_pvals(pvals, rejects, alpha=0.05):
    sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, sortind)
    ecdfactor = 1/len(pvals)*np.arange(1, len(pvals)+1)
    print(len(np.nonzero(rejects[sortind])))
    if len(np.nonzero(rejects[sortind])[0]) != 0:
        rejectmax = max(np.nonzero(rejects[sortind])[0])
        pvals_corrected_raw = pvals_sorted / ecdfactor[rejectmax]

        return pvals_corrected_raw[np.argsort(sortind)]
    else:
        return np.ones(len(pvals))

def multiple_comparisons(p_values, method):
    # Input checks.
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )

    p_values_array = np.atleast_1d(p_values)

    reject, corrected_p_values, _, _ = multipletests(p_values, method=method)
    if 'fdr' in method:
        corrected_p_values = corrected_fdr_pvals(p_values, reject)

    return reject, corrected_p_values

def bonferroni(p_values, num_comparisons='auto'):
    """Apply Bonferroni correction for multiple comparisons.
    The Bonferroni correction is defined as
        p_corrected = min(num_comparisons * p, 1.0).
    Arguments
    ---------
    p_values: scalar or list-like
        One or more p_values to correct.
    num_comparisons: int or `auto`
        Number of comparisons. Use `auto` to infer the number of comparisons
        from the length of the `p_values` list.
    Returns
    -------
    Scalar or numpy array of corrected p-values.
    """
    # Input checks.
    if np.ndim(p_values) > 1:
        raise_expected_got(
            'Scalar or list-like', 'argument `p_values`', p_values
        )
    if num_comparisons != 'auto':
        try:
            # Raise a TypeError if num_comparisons is not numeric, and raise
            # an AssertionError if it isn't int-like.
            assert np.ceil(num_comparisons) == num_comparisons
        except (AssertionError, TypeError) as e:
            raise_expected_got(
                'Int or `auto`', 'argument `num_comparisons`', num_comparisons
            )

    # Coerce p_values to numpy array.
    p_values_array = np.atleast_1d(p_values)

    if num_comparisons == 'auto':
        # Infer number of comparisons
        num_comparisons = len(p_values_array)
    elif len(p_values_array) > 1 and num_comparisons != len(p_values_array):
        # Warn if multiple p_values have been passed and num_comparisons is
        # set manually.
        warnings.warn(
            'Manually-specified `num_comparisons={}` differs from number of '
            'p_values to correct ({}).'.format(
                num_comparisons, len(p_values_array)
            )
        )

    # Apply correction by multiplying p_values and thresholding at p=1.0
    p_values_array *= num_comparisons
    p_values_array = np.min(
        [p_values_array, np.ones_like(p_values_array)], axis=0
    )

    if len(p_values_array) == 1:
        # Return a scalar if input was a scalar.
        return p_values_array[0]
    else:
        return p_values_array

def add_stat_annotation(ax, plot='boxplot',
                        data=None, x=None, y=None, hue=None, units=None, order=None,
                        hue_order=None, box_pairs=None, width=0.8, yerr=None,
                        perform_stat_test=True,
                        pvalues=None, test_short_name=None,
                        test='Mann-Whitney', text_format='standard', pvalue_format_string=DEFAULT,
                        text_annot_custom=None,
                        loc='inside', show_test_name=True,
                        pvalue_thresholds=DEFAULT, stats_params=dict(),
                        mult_comp_correction=None,
                        use_fixed_offset=False, line_offset_to_box=None,
                        use_fixed_offset_from_top=False,
                        line_offset=0.1, line_height=0.03, text_offset=1,
                        color='0.2', linewidth=1.5,
                        fontsize=15, verbose=1):
    """
    Optionally computes statistical test between pairs of data series, and add statistical annotation on top
    of the boxes/bars. The same exact arguments `data`, `x`, `y`, `hue`, `order`, `width`,
    `hue_order` (and `units`) as in the seaborn boxplot/barplot function must be passed to this function.
    This function works in one of the two following modes:
    a) `perform_stat_test` is True: statistical test as given by argument `test` is performed.
    b) `perform_stat_test` is False: no statistical test is performed, list of custom p-values `pvalues` are
       used for each pair of boxes. The `test_short_name` argument is then used as the name of the
       custom statistical test.
    :param plot: type of the plot, one of 'boxplot' or 'barplot'.
    :param line_height: in axes fraction coordinates
    :param text_offset: in points
    :param box_pairs: can be of either form: For non-grouped boxplot: `[(cat1, cat2), (cat3, cat4)]`. For boxplot grouped by hue: `[((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]`
    :param pvalue_format_string: defaults to `"{.3e}"`
    :param pvalue_thresholds: list of lists, or tuples. Default is: For "star" text_format: `[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]`. For "simple" text_format : `[[1e-5, "1e-5"], [1e-4, "1e-4"], [1e-3, "0.001"], [1e-2, "0.01"]]`
    :param pvalues: list or array of p-values for each box pair comparison.
    :param mult_comp_correction: Method for multiple comparisons correction. `bonferroni` or None.
    """

    if type(fontsize) != str:
        text_offset = fontsize*.275

    def find_x_position_box(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if box_plotter.plot_hues is None:
            cat = boxName
            hue_offset = 0
        else:
            cat = boxName[0]
            hue = boxName[1]
            hue_offset = box_plotter.hue_offsets[
                box_plotter.hue_names.index(hue)]

        group_pos = box_plotter.group_names.index(cat)
        box_pos = group_pos + hue_offset
        return box_pos

    def get_xpos_location(pos, xranges):
        '''given a value and a dictionary of range:id, return middle of range'''
        for xrange, label in xranges.items():
            if (pos >= xrange[0]) & (pos <= xrange[1]):
                return xrange[2]

    def generate_ymaxes(box_plotter, boxNames, data_to_ax):
        '''given box plotter and box name, return highest y point drawn in that box'''
        xpositions = {np.round(find_x_position_box(box_plotter, boxName),1):boxName for boxName in boxNames}
        ymaxes = {name:0 for name in boxNames}

        for child in ax.get_children():
            if (type(child) == PathCollection) and (len(child.properties()['offsets'])!=0):

                ymax = child.properties()['offsets'][:,1].max()
                xpos = float(np.round(np.nanmean(child.properties()['offsets'][:,0]),1))
                try:
                    xname = xpositions[xpos]
                except:
                    print (xpositions)
                    print (child.properties()['offsets'])
                    print (xpos)
                    raise
                ypos = data_to_ax.transform((0,ymax))[1]
                if ypos > ymaxes[xname]:
                    ymaxes[xname] = ypos

            elif (type(child) == lines.Line2D) or (type(child) == Rectangle):
                xunits = (max(list(xpositions.keys()))+1)/len(xpositions)
                xranges = {(pos-xunits/2, pos+xunits/2, pos):boxName for pos, boxName in xpositions.items()}
                box = ax.transData.inverted().transform(child.get_window_extent(fig.canvas.get_renderer()))

                if (box[:,0].max()-box[:,0].min())>1.1*xunits:
                    continue
                raw_xpos = np.round(box[:,0].mean(),1)
                xpos = get_xpos_location(raw_xpos, xranges)
                if xpos not in xpositions:
                    continue
                xname = xpositions[xpos]
                ypos = box[:,1].max()
                ypos = data_to_ax.transform((0,ypos))[1]
                if ypos > ymaxes[xname]:
                    ymaxes[xname] = ypos
        return ymaxes

    def get_box_data(box_plotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        Here we really have to duplicate seaborn code, because there is not
        direct access to the box_data in the BoxPlotter class.
        """
        #if boxName isn't a string, then boxName[0] raises an IndexError. This fixes that.
        try:
            cat = box_plotter.plot_hues is None and boxName or boxName[0]
        except IndexError:
            cat = box_plotter.plot_hues is None and boxName

        index = box_plotter.group_names.index(cat)
        group_data = box_plotter.plot_data[index]

        if box_plotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)
        else:
            hue_level = boxName[1]
            hue_mask = box_plotter.plot_hues[index] == hue_level
            box_data = remove_na(group_data[hue_mask])

        return box_data

    # Set default values if necessary
    if pvalue_format_string is DEFAULT:
        pvalue_format_string = '{:.3e}'
        simple_format_string = '{:.2f}'
    else:
        simple_format_string = pvalue_format_string

    if pvalue_thresholds is DEFAULT:
        if text_format == "star":
            pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        else:
            pvalue_thresholds = [[1e-5, "1e-5"], [1e-4, "1e-4"],
                                 [1e-3, "0.001"], [1e-2, "0.01"]]

    fig = plt.gcf()

    # Validate arguments
    if perform_stat_test:
        if test is None:
            raise ValueError("If `perform_stat_test` is True, `test` must be specified.")
        if pvalues is not None or test_short_name is not None:
            raise ValueError("If `perform_stat_test` is True, custom `pvalues` "
                             "or `test_short_name` must be `None`.")
        valid_list = ['t-test_ind', 't-test_welch', 't-test_paired',
                      'Mann-Whitney', 'Mann-Whitney-gt', 'Mann-Whitney-ls',
                      'Levene', 'Wilcoxon', 'Kruskal', 'bootstrap', 'paired_bootstrap']
        if test not in valid_list:
            raise ValueError("test value should be one of the following: {}."
                             .format(', '.join(valid_list)))
    else:
        if pvalues is None:
            raise ValueError("If `perform_stat_test` is False, custom `pvalues` must be specified.")
        if test is not None:
            raise ValueError("If `perform_stat_test` is False, `test` must be None.")
        if len(pvalues) != len(box_pairs):
            raise ValueError("`pvalues` should be of the same length as `box_pairs`.")

    if text_annot_custom is not None and len(text_annot_custom) != len(box_pairs):
        raise ValueError("`text_annot_custom` should be of same length as `box_pairs`.")

    assert_is_in(
        loc, ['inside', 'outside'], label='argument `loc`'
    )
    assert_is_in(
        text_format,
        ['standard','full', 'simple', 'star'],
        label='argument `text_format`'
    )
    assert_is_in(
        mult_comp_correction,
        ['bonferroni', 'sidak','holm-sidak','holm','simes-hochberg','hommel','fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky','fdr_gbs',None],
        label='argument `mult_comp_correction`'
    )

    if verbose >= 1 and text_format == 'star':
        print("p-value annotation legend:")
        pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalue_thresholds)):
            if i < len(pvalue_thresholds)-1:
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalue_thresholds[i][1],
                                                        pvalue_thresholds[i+1][0],
                                                        pvalue_thresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalue_thresholds[i][1], pvalue_thresholds[i][0]))
        print()

    orig_ylim = ax.get_ylim()
    yrange = orig_ylim[1] - orig_ylim[0]
    trans = ax.get_xaxis_transform()
    data_to_ax = ax.transData+ax.get_xaxis_transform().inverted() #Will work in data coordinates on x axis, and axis coordinates on y axis
    ax_to_data = data_to_ax.inverted()
    pix_to_ax = ax.transAxes.inverted()

    ylim = (0,1)
    yrange = 1

    if line_offset is None:
        if loc == 'inside':
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = 0.1
        # 'outside', see valid_list
        else:
            line_offset = 0.05
            if line_offset_to_box is None:
                line_offset_to_box = line_offset
    else:
        if loc == 'inside':
            if line_offset_to_box is None:
                line_offset_to_box = 0.1
        elif loc == 'outside':
            line_offset_to_box = line_offset

    y_offset = line_offset*yrange
    y_offset_to_box = line_offset_to_box*yrange

    if plot == 'boxplot':
        # Create the same plotter object as seaborn's boxplot
        box_plotter = sns.categorical._BoxPlotter(
            x, y, hue, data, order, hue_order, orient=None, width=width, color=None,
            palette=None, saturation=.75, dodge=True, fliersize=5, linewidth=None)
    elif plot == 'barplot':
        # Create the same plotter object as seaborn's barplot
        if yerr:
            data[y] += np.array(yerr)
        box_plotter = sns.categorical._BarPlotter(
            x, y, hue, data, order, hue_order, seed=None,
            estimator=np.mean, ci=95, n_boot=1000, units=None,
            orient=None, color=None, palette=None, saturation=.75,
            errcolor=".26", errwidth=None, capsize=None, dodge=True, yerr=[10,4,2,8,1])

    # Build the list of box data structures with the x and ymax positions
    group_names = box_plotter.group_names
    hue_names = box_plotter.hue_names

    if box_plotter.plot_hues is None:
        box_names = group_names
        labels = box_names
    else:
        box_names = [(group_name, hue_name) for group_name in group_names for hue_name in hue_names]
        labels = ['{}_{}'.format(group_name, hue_name) for (group_name, hue_name) in box_names]

    ymaxes = generate_ymaxes(box_plotter, box_names, data_to_ax)

    box_structs = [{'box':box_names[i],
                    'label':labels[i],
                    'x':find_x_position_box(box_plotter, box_names[i]),
                    'box_data':get_box_data(box_plotter, box_names[i]),
                    'ymax':ymaxes[box_names[i]]}
                   for i in range(len(box_names))]

    # Sort the box data structures by position along the x axis
    box_structs = sorted(box_structs, key=lambda x: x['x'])
    # Add the index position in the list of boxes along the x axis
    box_structs = [dict(box_struct, xi=i) for i, box_struct in enumerate(box_structs)]
    # Same data structure list with access key by box name
    box_structs_dic = {box_struct['box']:box_struct for box_struct in box_structs}

    # Build the list of box data structure pairs
    box_struct_pairs = []
    test_result_list = []

    for i_box_pair, (box1, box2) in enumerate(box_pairs):
        valid = box1 in box_names and box2 in box_names
        if not valid:
            raise ValueError("box_pairs contains an invalid box pair.")
            pass
        # i_box_pair will keep track of the original order of the box pairs.
        box_struct1 = dict(box_structs_dic[box1], i_box_pair=i_box_pair)
        box_struct2 = dict(box_structs_dic[box2], i_box_pair=i_box_pair)
        if box_struct1['x'] <= box_struct2['x']:
            pair = (box_struct1, box_struct2)
        else:
            pair = (box_struct2, box_struct1)

        if perform_stat_test:
            result = stat_test(
                box_struct1['box_data'],
                box_struct2['box_data'],
                test,
                **stats_params
            )
        else:
            test_short_name = test_short_name if test_short_name is not None else ''
            result = StatResult(
                'Custom statistical test',
                test_short_name,
                None,
                None,
                pvalues[i_box_pair]
            )

        result.box1 = box1
        result.box2 = box2
        test_result_list.append(result)
        box_struct_pairs.append(pair)

        if verbose >= 1:
            greater_or_less_than = 'equal to'
            print(f"{box_struct1['label']}: {np.median(box_struct1['box_data'])}, {box_struct2['label']}: {np.median(box_struct2['box_data'])}")
            if np.median(box_struct1['box_data']) > np.median(box_struct2['box_data']):
                greater_or_less_than = 'greater than'
            elif np.median(box_struct1['box_data']) < np.median(box_struct2['box_data']):
                greater_or_less_than = 'less than'
            print("{} is {} {}: {}".format(box_struct1['label'], greater_or_less_than, box_struct2['label'], result.formatted_output))
            if box_struct1['box_data'].mean() > box_struct2['box_data'].mean():
                greater_or_less_than = 'greater than'
            elif box_struct1['box_data'].mean() < box_struct2['box_data'].mean():
                greater_or_less_than = 'less than'
            print("(means: {} is {} {}: {} vs {})".format(box_struct1['label'], greater_or_less_than, box_struct2['label'], box_struct1['box_data'].mean(), box_struct2['box_data'].mean()))

    if mult_comp_correction:
        pvals = [result.pval for result in test_result_list]
        reject_null, corrected_pvals = multiple_comparisons(pvals, method= mult_comp_correction)
        print (pvals)
        print (corrected_pvals)
        print (reject_null)
        for result, reject_null, pval in zip(test_result_list, reject_null, corrected_pvals):
            result.set_pval(pval)
            if not reject_null:
                result.test_str = 'n.s.'
                result.set_pval(1)
            else:
                result.test_str = result.test_str + f'{result.test_str} with {mult_comp_correction} correction'
    box_struct_pairs = [pair+(result,) for pair, result in zip(box_struct_pairs, test_result_list)]

    # Draw first the annotations with the shortest between-boxes distance, in order to reduce
    # overlapping between annotations.
    box_struct_pairs = sorted(box_struct_pairs, key=lambda x: abs(x[1]['x'] - x[0]['x']))

    # Build array that contains the x and y_max position of the highest annotation or box data at
    # a given x position, and also keeps track of the number of stacked annotations.
    # This array will be updated when a new annotation is drawn.
    y_stack_arr = np.array([[box_struct['x'] for box_struct in box_structs],
                            [box_struct['ymax'] for box_struct in box_structs],
                            [0 for i in range(len(box_structs))]])

    highestDataDrawn = y_stack_arr[1,:].max()

    if loc == 'outside':
        y_stack_arr[1, :] = ylim[1]
    ann_list = []
    ymaxs = []
    y_stack = []
    items_to_draw = []

    for box_struct1, box_struct2, result in box_struct_pairs:
        box1 = box_struct1['box']
        box2 = box_struct2['box']
        label1 = box_struct1['label']
        label2 = box_struct2['label']
        box_data1 = box_struct1['box_data']
        box_data2 = box_struct2['box_data']
        x1 = box_struct1['x']
        x2 = box_struct2['x']
        xi1 = box_struct1['xi']
        xi2 = box_struct2['xi']
        ymax1 = box_struct1['ymax']
        ymax2 = box_struct2['ymax']
        i_box_pair = box_struct1['i_box_pair']

        # Find y maximum for all the y_stacks *in between* the box1 and the box2
        i_ymax_in_range_x1_x2 = xi1 + np.nanargmax(y_stack_arr[1, np.where((x1 <= y_stack_arr[0, :]) &
                                                                           (y_stack_arr[0, :] <= x2))])
        ymax_in_range_x1_x2 = y_stack_arr[1, i_ymax_in_range_x1_x2]

        if use_fixed_offset_from_top:
            #if allowing overlap, simply find the highest y_stack
            ymax_in_range_x1_x2 = highestDataDrawn

        # if perform_stat_test:
        #     result = stat_test(
        #         box_data1,
        #         box_data2,
        #         test,
        #         mult_comp_correction,
        #         len(box_struct_pairs),
        #         **stats_params
        #     )
        # else:
        #     test_short_name = test_short_name if test_short_name is not None else ''
        #     result = StatResult(
        #         'Custom statistical test',
        #         test_short_name,
        #         None,
        #         None,
        #         pvalues[i_box_pair]
        #     )

        # result.box1 = box1
        # result.box2 = box2
        # test_result_list.append(result)

        # if verbose >= 1:
        #     print("{} v.s. {}: {}".format(label1, label2, result.formatted_output))

        if text_annot_custom is not None:
            text = text_annot_custom[i_box_pair]
        else:
            if text_format == 'standard':
                if result.pval > 0.05:
                    text = 'n.s.'
                else:
                    if '<' in result.pval_formatted:
                        text = f'p {result.pval_formatted}'
                    else:
                        text = f'p = {result.pval_formatted}'
            if text_format == 'full':
                text = "{} p = {}".format(result.test_short_name, result.formatted_output)
            elif text_format is None:
                text = None
            elif text_format is 'star':
                text = pval_annotation_text(result.pval, pvalue_thresholds)
            elif text_format is 'simple':
                test_short_name = show_test_name and test_short_name or ""
                text = simple_text(result.pval, simple_format_string, pvalue_thresholds, test_short_name)

        yref = ymax_in_range_x1_x2
        yref2 = yref

        # Choose the best offset depending on wether there is an annotation below
        # at the x position in the range [x1, x2] where the stack is the highest

        if (y_stack_arr[2, i_ymax_in_range_x1_x2] == 0) or use_fixed_offset_from_top:
            # there is only a box below
            offset = y_offset_to_box
        else:
            # there is an annotation below
            offset = y_offset

        y = yref2 + offset
        h = line_height*yrange
        ax_line_x, ax_line_y = [x1, x1, x2, x2], [y, y + h, y + h, y]
        points = [ax_to_data.transform((x,y)) for x,y in zip(ax_line_x, ax_line_y)]
        line_x, line_y = [x for x,y in points], [y for x,y in points]

        if loc == 'inside':
            ax.plot(line_x, line_y, lw=linewidth, c=color)
        elif loc == 'outside':
            line = lines.Line2D(line_x, line_y, lw=linewidth, c=color)
            line.set_clip_on(False)
            ax.add_line(line)

        if text is not None:
            ann = ax.annotate(
                text, xy=(np.mean([x1, x2]), line_y[2]),
                xytext=(0, text_offset),
                textcoords='offset points',
                xycoords='data', ha='center', va='bottom',
                fontsize=fontsize, clip_on=False, annotation_clip=False)

            ann_list.append(ann)

            plt.draw()
            ax.set_ylim(orig_ylim)
            y_top_annot = None
            got_mpl_error = False
            if not use_fixed_offset:
                try:
                    bbox = ann.get_window_extent()
                    bbox_ax = bbox.transformed(pix_to_ax)
                    y_top_annot = bbox_ax.ymax

                except RuntimeError:
                    got_mpl_error = True

            if use_fixed_offset or got_mpl_error:
                if verbose >= 1:
                    print("Warning: cannot get the text bounding box. Falling back to a fixed"
                          " y offset. Layout may be not optimal.")
                # We will apply a fixed offset in points,
                # based on the font size of the annotation.
                fontsize_points = FontProperties(size='medium').get_size_in_points()
                offset_trans = mtransforms.offset_copy(
                    ax.transData, fig=fig, x=0,
                    y=1.0*fontsize_points + text_offset, units='points')
                y_top_display = offset_trans.transform((0, y + h))
                y_top_annot = ax.transData.inverted().transform(y_top_display)[1]
                
        else:
            y_top_annot = y + h

        y_stack.append(y_top_annot)	# remark: y_stack is not really necessary if we have the stack_array
        
        ymaxs.append(max(y_stack))
        # Fill the highest y position of the annotation into the y_stack array
        # for all positions in the range x1 to x2
        y_stack_arr[1, (x1 <= y_stack_arr[0, :]) & (y_stack_arr[0, :] <= x2)] = y_top_annot
        y_stack_arr[2, xi1:xi2 + 1] = y_stack_arr[2, xi1:xi2 + 1] + 1

    y_stack_max = max(ymaxs)
    
    
    #reset transformation
    data_to_ax = ax.transData+ax.get_xaxis_transform().inverted() #Will work in data coordinates on x axis, and axis coordinates on y axis
    ax_to_data = data_to_ax.inverted()
    
    if loc == 'inside':
        ax.set_ylim(ax_to_data.transform([(0,ylim[0]),(0,max(1.05*y_stack_max, ylim[1]))])[:,1])
    elif loc == 'outside':
        ax.set_ylim(ax_to_data.transform([(0,ylim[0]),(0,y_stack_max, ylim[1])])[:,1])

    return ax, test_result_list