import pathlib
import pandas
import igraph

CURRENT_PATH = pathlib.Path(__file__).absolute().parent

DATA_PATH = CURRENT_PATH / 'data'
GRAPH_PATH = CURRENT_PATH / 'graphs'

ECOSYSTEMS = [p.parts[-1] for p in DATA_PATH.iterdir() if p.is_dir()]
DATE_RANGE = pandas.date_range('2011-01-01', '2016-09-01', freq='6MS')


def clean_data(packages, dependencies):
    # Filter unknown package/version
    dependencies = dependencies.merge(
        packages[['package', 'version']],
        how='inner'
    )
    
    # Filter unknown dependencies
    dependencies = dependencies.merge(
        packages[['package']],
        how='inner',
        left_on='dependency',
        right_on='package',
        suffixes=('', '_y')
    ).drop('package_y', axis=1)
    
    return packages, dependencies
    
    
def load_data(ecosystem):
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
    graph = igraph.Graph(directed=True)
    graph.add_vertices(v for v in packages['package'])
    graph.vs['time'] = (v for v in packages['time'])
    graph.vs['version'] = (v for v in packages['version'])
    graph.add_edges(
        [(row.package, row.dependency) for row in dependencies[['package', 'dependency']].itertuples()]
    )
    
    return graph


def load_graph(ecosystem, date, force=False):
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


if __name__ == '__main__':
    for ecosystem in ECOSYSTEMS:
        print(ecosystem)
        packages, dependencies = load_data(ecosystem)
        
        for date in DATE_RANGE:
            print(' - ', date)
            
            filepath = (GRAPH_PATH / ecosystem / date.strftime('%Y-%m-%d.graphml.gz'))
            graph = create_graph(*create_snapshot(packages, dependencies, date))
            
            try:
                (GRAPH_PATH / ecosystem).mkdir()
            except FileExistsError as e: 
                pass
            graph.write_graphmlz(filepath.as_posix())
            
        print()
