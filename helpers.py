import collections
import pathlib

import igraph
import pandas
import scipy.stats
import statsmodels.api as sm

CURRENT_PATH = pathlib.Path(__file__).absolute().parent

DATA_PATH = CURRENT_PATH / 'data'
GRAPH_PATH = CURRENT_PATH / 'graphs'
FIGURE_PATH = CURRENT_PATH / 'figures'

LIBRARIES_IO_PATH = CURRENT_PATH / 'libraries.io'
LIBRARIES_IO_KIND = {
    'npm': ['runtime'],  # Development Optional runtime
    'rubygems': ['runtime'],  # Development runtime
    'cran': ['imports', 'depends'],  # depends enhances imports suggests
    'packagist': ['runtime'],  # Development runtime
    'cpan': ['runtime'],  # build configure develop runtime test x_benchmarks x_examples
    'nuget': ['runtime'],  # runtime
    'cargo': ['normal', 'runtime'],  # build dev normal runtime
}

ECOSYSTEMS = ['cargo', 'cpan', 'cran', 'npm', 'nuget', 'packagist', 'rubygems']
DATE_RANGE = pandas.date_range('2012-01-01', '2017-01-01', freq='MS')

RE_SEMVER = r'^(?:v|V)?(?P<v_major>\d+)\.(?P<v_minor>\d+)\.(?P<v_patch>\d+)(?P<v_misc>.*)$'


def convert_from_libraries_io(ecosystem, target=None):
    """
    Convert data from libraries.io format to our format.
    """
    target = ecosystem if target is None else target
    
    pkg_filepath = LIBRARIES_IO_PATH / ecosystem / '{}-versions.csv'.format(ecosystem)
    deps_filepath = LIBRARIES_IO_PATH / ecosystem / '{}-dependencies.csv'.format(ecosystem)

    try:
        (DATA_PATH / target).mkdir()
    except FileExistsError as e:
        pass

    (
        pandas.read_csv(pkg_filepath.as_posix())
        .rename(columns={
            'Project name': 'package',
            'Version number': 'version',
            'Version date': 'time',
        })
        .to_csv(
            (DATA_PATH / target / 'packages.csv.gz').as_posix(),
            columns=['package', 'version', 'time'],
            index=False,
            compression='gzip'
        )
    )
    
    (
        pandas.read_csv(deps_filepath.as_posix())
        .rename(columns={
            'Project name': 'package',
            'Version number': 'version',
            'Dependency name': 'dependency',
            'Dependency requirements': 'constraint',
            'Dependency kind': 'kind',
        })
        .query(' or '.join(['kind == "{}"'.format(kind) for kind in LIBRARIES_IO_KIND[ecosystem]]))
        .to_csv(
            (DATA_PATH / target / 'dependencies.csv.gz').as_posix(),
            columns=['package', 'version', 'dependency', 'constraint'],
            index=False,
            compression='gzip'
        )
    )


def clean_data(packages, dependencies, ecosystem=None):
    """
    Remove invalid or unknown packages and dependencies.
    """
    # Filter releases with invalide date
    packages = (
        packages[packages['time'] >= pandas.to_datetime('1980-01-01')]
        .dropna()
    )

    # For npm, remove packages starting with:
    # - all-packages-
    # - cool-
    # - neat-
    # - wowdude-
    # This represents around 245 packages with have a very high number
    # of dependencies, and are just "fun packages", as explained here:
    # https://libraries.io/npm/wowdude-119
    if ecosystem == 'npm':
        filtered = ('all-packages-', 'cool-', 'neat-', '-wowdue-',)
        packages = packages[~packages['package'].str.startswith(filtered)]

    # Filter unknown package/version
    dependencies = dependencies.merge(
        packages[['package', 'version']],
        how='inner'
    )
    
    # Filter unknown dependencies
    dependencies = dependencies[dependencies['dependency'].isin(packages['package'])]
    
    return packages, dependencies
    
    
def load_data(ecosystem):
    """
    Return a pair (packages, dependencies) of dataframes for the given ecosystem.
    """
    # Load data files
    packages = pandas.read_csv(
        (DATA_PATH / ecosystem / 'packages.csv.gz').as_posix(),
        usecols=['package', 'version', 'time'],
        parse_dates=['time'],
        infer_datetime_format=True,
        engine='c',
    )
    
    dependencies = pandas.read_csv(
        (DATA_PATH / ecosystem / 'dependencies.csv.gz').as_posix(),
        usecols=['package', 'version', 'dependency', 'constraint'],
        engine='c',
    )
    
    return clean_data(packages, dependencies, ecosystem)


def create_snapshot(packages, dependencies, date):
    """
    Return a pair (packages, dependencies) that corresponds to the state of the ecosystem at given date.
    """
    packages = (
        packages[packages['time'] <= pandas.to_datetime(date)]
        .sort_values('time')
        .groupby('package', sort=False)
        .tail(1)
    )
    
    dependencies = dependencies.merge(
        packages[['package', 'version']],
        how='inner',
    )
    
    return clean_data(packages, dependencies)


def create_graph(packages, dependencies):
    """
    Create and enrich a dependency graph.
    
    The enrichment adds 'version', 'in', 'out', 'tr-in', 'tr-out' to the nodes, and 'constraint' to the edges.
    """
    graph = igraph.Graph(directed=True)
    
    graph.add_vertices(str(v) for v in packages['package'])
    # graph.vs['time'] = [v for v in packages['time']]
    graph.vs['version'] = [v for v in packages['version']]
    
    graph.add_edges(
        [(row.package, row.dependency) for row in dependencies[['package', 'dependency']].itertuples()]
    )
    graph.es['constraint'] = [v for v in dependencies['constraint']]
    
    graph.vs['in'] = [n - 1 for n in graph.neighborhood_size(order=1, mode=igraph.IN)]
    graph.vs['out'] = [n - 1 for n in graph.neighborhood_size(order=1, mode=igraph.OUT)]
    graph.vs['tr-in'] = [n - 1 for n in graph.neighborhood_size(order=len(graph.vs), mode=igraph.IN)]
    graph.vs['tr-out'] = [n - 1 for n in graph.neighborhood_size(order=len(graph.vs), mode=igraph.OUT)]
    
    return graph


def load_graph(ecosystem, date, force=False):
    """
    Load or construct a dependency graph for the ecosystem at given date.
    
    Set 'force' to True to bypass caching.
    """
    filename = pandas.to_datetime(date).strftime('%Y-%m-%d.graphml.gz')
    filepath = (GRAPH_PATH / ecosystem / filename)
    
    if force or not filepath.exists():
        packages, dependencies = load_data(ecosystem)
        packages, dependencies = create_snapshot(packages, dependencies, date)
        graph = create_graph(packages, dependencies)
        
        # Save graph
        try:
            (GRAPH_PATH / ecosystem).mkdir()
        except FileExistsError as e:
            pass
        graph.write_graphmlz(filepath.as_posix())
        
        return graph
    else:
        return igraph.read(filepath.as_posix(), format='graphmlz')


def evolution_regression(df, xlog=False, ylog=False, return_raw=False):
    """
    Return R² value for df.index] ~ a.df[x] + b for all column x in given dataframe.
    
    If 'return_raw' is True, return the resulting OLS object instead of its R².
    The results are returned through a dict which associates to each column the result of the regression.
    """
    results = collections.OrderedDict()
    
    for column in df.columns:
        time = pandas.Series(df[column].dropna().index)
        X = 1 + (time - time.min()).dt.days
        X = pandas.np.log10(X) if xlog else X
        X = sm.add_constant(X, prepend=False)
        
        if not ylog:
            y = df[column].dropna()
        else:
            y = pandas.np.log10(df[column].dropna().apply(lambda v: max(v, 10e-5)))
        y = y.reset_index(drop=True)
        
        result = sm.OLS(y, X).fit()
        
        results[column] = result if return_raw else result.rsquared
        
    return results


def evolution_linlog_regressions(df, return_raw=False):
    """
    Return the results of multiple regressions (lin/log).
    """
    data = pandas.DataFrame(columns=df.columns)
    
    data.loc['lin-lin', :] = evolution_regression(df, return_raw=return_raw)
    data.loc['lin-log', :] = evolution_regression(df, ylog=True, return_raw=return_raw)
    data.loc['log-lin', :] = evolution_regression(df, xlog=True, return_raw=return_raw)
    data.loc['log-log', :] = evolution_regression(df, xlog=True, ylog=True, return_raw=return_raw)
    
    return data


def CohenEffectSize(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    
    effect sizes as "small, d = .2," "medium, d = .5," and "large, d = .8"
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / pandas.np.sqrt(pooled_var)
    return d
    

def cliffsDelta(lst1, lst2):
    """
    0.147 / 0.33 / 0.474 (negligible/small/medium/large).
    """
    def runs(lst):
        "Iterator, chunks repeated values"
        for j, two in enumerate(lst):
            if j == 0:
                one, i = two, 0
            if one != two:
                yield j - i, one
                i = j
            one = two
        yield j - i + 1, two
        
    m, n = len(lst1), len(lst2)
    lst2 = sorted(lst2)
    j = more = less = 0
    for repeats, x in runs(sorted(lst1)):
        while j <= (n - 1) and lst2[j] < x:
            j += 1
        more += j*repeats
        while j <= (n - 1) and lst2[j] == x:
            j += 1
        less += (n - j)*repeats
    d = (more - less) / (m*n)
    d = abs(d)
    if d < 0.147:
        label = 'negligible'
    elif d < 0.33: 
        label = 'small'
    elif d < 0.474:
        label = 'medium'
    else:
        label = 'large'
    
    return d, label


def compare_distributions(a, b):
    """
    Test for a < b using Mann Whitney U and Cliff's delta. 
    Return score, p-value, Cliff's delta, label.
    """
    score, pvalue = scipy.stats.mannwhitneyu(a, b, alternative='less')
    d, label = cliffsDelta(a, b)
    return score, pvalue, d, label
    

def savefig(fig, name):
    fig.savefig(
        (FIGURE_PATH / '{}.pdf'.format(name)).as_posix(),
        bbox_inches='tight'
    )
    
    
if __name__ == '__main__':
    for ecosystem in ECOSYSTEMS:
        print(ecosystem)
        packages, dependencies = load_data(ecosystem)
        
        for date in DATE_RANGE:
            print('-', date, end=': ')
            
            filepath = (GRAPH_PATH / ecosystem / date.strftime('%Y-%m-%d.graphml.gz'))
            graph = create_graph(*create_snapshot(packages, dependencies, date))
            print('{} vertices and {} edges'.format(len(graph.vs), len(graph.es)))
            try:
                (GRAPH_PATH / ecosystem).mkdir()
            except FileExistsError as e: 
                pass
            graph.write_graphmlz(filepath.as_posix())
            
        print()
