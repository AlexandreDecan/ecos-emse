import collections
import pathlib

import igraph
import pandas
import statsmodels.api as sm


CURRENT_PATH = pathlib.Path(__file__).absolute().parent

DATA_PATH = CURRENT_PATH / 'data'
GRAPH_PATH = CURRENT_PATH / 'graphs'

ECOSYSTEMS = sorted([p.parts[-1] for p in DATA_PATH.iterdir() if p.is_dir()])
DATE_RANGE = pandas.date_range('2011-01-01', '2016-06-01', freq='MS')

# RE_SEMVER = r'(\d+)\.(\d+)(?:\.(\d+))?'
RE_SEMVER = r'^(?P<v_major>\d+)\.(?P<v_minor>\d+)\.(?P<v_patch>.+)$'


def clean_data(packages, dependencies):
    """
    Remove invalid or unknown packages and dependencies.
    """
    # Filter releases with invalide date
    packages = packages[packages['time'] >= pandas.to_datetime('1980-01-01')]

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
    )
    dependencies = pandas.read_csv(
        (DATA_PATH / ecosystem / 'dependencies.csv.gz').as_posix(),
        usecols=['package', 'version', 'dependency', 'constraint'],
    )
    
    return clean_data(packages, dependencies)


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
    
    The enrichment adds 'time', 'version', 'in', 'out', 'tr-in', 'tr-out' to the nodes, and 'constraint' to the edges.
    """
    graph = igraph.Graph(directed=True)
    
    graph.add_vertices(str(v) for v in packages['package'])
    graph.vs['time'] = (v for v in packages['time'])
    graph.vs['version'] = (v for v in packages['version'])
    
    graph.add_edges(
        [(row.package, row.dependency) for row in dependencies[['package', 'dependency']].itertuples()]
    )
    graph.es['constraint'] = (v for v in dependencies['constraint'])
    
    graph.vs['in'] = graph.indegree()
    graph.vs['out'] = graph.outdegree()
    graph.vs['tr-in'] = graph.neighborhood_size(order=len(graph.vs), mode=igraph.IN)
    graph.vs['tr-out'] = graph.neighborhood_size(order=len(graph.vs), mode=igraph.OUT)
    
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
    The results are returned through a dict which associates to each column  the result of the regression.
    """
    results = collections.OrderedDict()
    
    time = pandas.Series(df.index)
    X = 1 + (time - time.min()).dt.days
    X = pandas.np.log10(X) if xlog else X
    X = sm.add_constant(X, prepend=False)
    
    for column in df.columns:
        y = df[column] if not ylog else pandas.np.log10(df[column])
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
