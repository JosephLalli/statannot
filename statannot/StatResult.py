class StatResult:
    def __init__(self, test_str, test_short_name, stat_str, stat, pval):
        self.test_str = test_str
        self.test_short_name = test_short_name
        self.stat_str = stat_str
        self.stat = stat
        self.pval = pval
        self.pval_formatted = None
        self.box1 = None
        self.box2 = None
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


def pval_annotation_text(x, pvalue_thresholds):
    single_value = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        single_value = True
    # Sort the threshold array
    pvalue_thresholds = pd.DataFrame(pvalue_thresholds).sort_values(by=0, ascending=False).values
    x_annot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalue_thresholds)):
        if i < len(pvalue_thresholds)-1:
            condition = (x1 <= pvalue_thresholds[i][0]) & (pvalue_thresholds[i+1][0] < x1)
            x_annot[condition] = pvalue_thresholds[i][1]
        else:
            condition = x1 < pvalue_thresholds[i][0]
            x_annot[condition] = pvalue_thresholds[i][1]

    return x_annot if not single_value else x_annot.iloc[0]


def simple_text(pval, pvalue_format, pvalue_thresholds, test_short_name=None):
    """
    Generates simple text for test name and pvalue
    :param pval: pvalue
    :param pvalue_format: format string for pvalue
    :param test_short_name: Short name of test to show
    :param pvalue_thresholds: String to display per pvalue range
    :return: simple annotation
    """
    # Sort thresholds
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])

    # Test name if passed
    text = test_short_name and test_short_name + " " or ""

    for threshold in thresholds:
        if pval < threshold[0]:
            pval_text = "p ≤ {}".format(threshold[1])
            break
    else:
        pval_text = "p = {}".format(pvalue_format).format(pval)

    return text + pval_text
