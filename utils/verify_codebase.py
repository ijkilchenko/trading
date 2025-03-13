#!/usr/bin/env python
"""
Utility script to verify the necessity of each Python file in the trading system.
This script analyzes import relationships between modules and identifies unused files.
"""

import os
import re
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify the necessity of Python files in the trading system")
    parser.add_argument('--root_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="Root directory of the trading system (default: parent of this script's directory)")
    parser.add_argument('--visualize', action='store_true',
                        help="Generate a visualization of the dependency graph")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save the visualization (default: root_dir)")
    return parser.parse_args()

def find_python_files(root_dir):
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                python_files.append(rel_path)
    return python_files

def parse_imports(file_path):
    """Parse import statements from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expressions to match different types of imports
    import_patterns = [
        r'import\s+([a-zA-Z0-9_.]+)',  # import module
        r'from\s+([a-zA-Z0-9_.]+)\s+import',  # from module import ...
    ]
    
    imports = []
    for pattern in import_patterns:
        matches = re.findall(pattern, content)
        imports.extend(matches)
    
    return imports

def normalize_import(import_name, file_path, root_dir):
    """Normalize import names to match file paths."""
    # Convert relative imports to absolute
    if import_name.startswith('.'):
        # Get the package of the current module
        package_path = os.path.dirname(file_path)
        if import_name.startswith('..'):
            # Go up one level for each dot after the first
            dots = 0
            while import_name.startswith('.'):
                import_name = import_name[1:]
                dots += 1
            
            for _ in range(dots - 1):
                package_path = os.path.dirname(package_path)
            
            import_name = f"{os.path.basename(package_path)}.{import_name}" if import_name else os.path.basename(package_path)
        else:
            # Single dot import
            import_name = import_name[1:]  # Remove the leading dot
            package_name = os.path.basename(package_path)
            import_name = f"{package_name}.{import_name}" if import_name else package_name
    
    # Convert import paths to file paths
    parts = import_name.split('.')
    if parts[0] in ['data', 'models', 'strategies', 'backtesting', 'visualization', 'risk_management', 'simulation', 'trading', 'utils']:
        # Internal module
        if len(parts) > 1:
            file_path = os.path.join(*parts[:-1], f"{parts[-1]}.py")
            if not os.path.exists(os.path.join(root_dir, file_path)):
                file_path = os.path.join(*parts) + ".py"
        else:
            # Could be a directory with __init__.py or a single file
            file_path = os.path.join(parts[0], "__init__.py")
            if not os.path.exists(os.path.join(root_dir, file_path)):
                file_path = f"{parts[0]}.py"
        
        return file_path
    
    return None  # External module

def build_dependency_graph(python_files, root_dir):
    """Build a dependency graph of Python files."""
    graph = nx.DiGraph()
    
    # Add all Python files as nodes
    for file in python_files:
        graph.add_node(file)
    
    # Add edges for dependencies
    for file in python_files:
        full_path = os.path.join(root_dir, file)
        imports = parse_imports(full_path)
        
        for import_name in imports:
            normalized_import = normalize_import(import_name, file, root_dir)
            if normalized_import and normalized_import in python_files:
                graph.add_edge(file, normalized_import)
    
    return graph

def find_entry_points(graph):
    """Find potential entry points in the dependency graph."""
    # Entry points are nodes with zero in-degree or main.py
    entry_points = []
    for node in graph.nodes:
        if node == 'main.py' or graph.in_degree(node) == 0:
            entry_points.append(node)
    
    return entry_points

def find_unused_files(graph, entry_points):
    """Find files that are not reachable from any entry point."""
    # Get all nodes reachable from entry points
    reachable_nodes = set()
    for entry_point in entry_points:
        if entry_point in graph:
            reachable_nodes.update(nx.dfs_preorder_nodes(graph, entry_point))
    
    # Files not reachable from any entry point are considered unused
    all_nodes = set(graph.nodes)
    unused_files = all_nodes - reachable_nodes
    
    return unused_files

def find_highly_imported_files(graph):
    """Find files that are imported by many other files."""
    in_degrees = {}
    for node in graph.nodes:
        in_degrees[node] = graph.in_degree(node)
    
    # Sort by in-degree (number of files that import this file)
    highly_imported = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    
    return highly_imported

def visualize_dependency_graph(graph, output_path):
    """Generate a visualization of the dependency graph."""
    plt.figure(figsize=(20, 16))
    
    # Compute node sizes based on importance
    in_degrees = dict(graph.in_degree())
    max_in_degree = max(in_degrees.values()) if in_degrees else 1
    node_sizes = [200 + 300 * (in_degrees.get(node, 0) / max_in_degree) for node in graph.nodes]
    
    # Create a layout
    pos = nx.spring_layout(graph, k=0.15, iterations=50)
    
    # Add nodes with colors based on module type
    node_colors = []
    for node in graph.nodes:
        if 'data/' in node:
            node_colors.append('skyblue')
        elif 'models/' in node:
            node_colors.append('lightgreen')
        elif 'strategies/' in node:
            node_colors.append('salmon')
        elif 'backtesting/' in node:
            node_colors.append('orange')
        elif 'visualization/' in node:
            node_colors.append('purple')
        elif 'risk_management/' in node:
            node_colors.append('pink')
        elif 'simulation/' in node:
            node_colors.append('yellow')
        elif 'trading/' in node:
            node_colors.append('red')
        elif 'utils/' in node:
            node_colors.append('lightgray')
        elif 'tests/' in node:
            node_colors.append('lightblue')
        else:
            node_colors.append('gray')
    
    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True, arrowsize=10)
    
    # Add labels
    labels = {}
    for node in graph.nodes:
        # Simplify labels to make them more readable
        label = os.path.basename(node)
        labels[node] = label
    
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    # Add a legend
    legend_colors = {
        'Data': 'skyblue',
        'Models': 'lightgreen',
        'Strategies': 'salmon',
        'Backtesting': 'orange',
        'Visualization': 'purple',
        'Risk Management': 'pink',
        'Simulation': 'yellow',
        'Trading': 'red',
        'Utils': 'lightgray',
        'Tests': 'lightblue',
        'Other': 'gray'
    }
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
              for label, color in legend_colors.items()]
    plt.legend(handles=handles, loc='best')
    
    plt.title('Trading System Module Dependencies')
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dependency graph visualization saved to {output_path}")

def main():
    """Main function."""
    args = parse_args()
    root_dir = args.root_dir
    output_dir = args.output_dir or root_dir
    
    print(f"Analyzing Python files in {root_dir}")
    
    # Find all Python files
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Build dependency graph
    graph = build_dependency_graph(python_files, root_dir)
    print(f"Built dependency graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # Find entry points
    entry_points = find_entry_points(graph)
    print(f"Found {len(entry_points)} potential entry points:")
    for entry_point in entry_points:
        print(f"  - {entry_point}")
    
    # Find unused files
    unused_files = find_unused_files(graph, entry_points)
    print(f"Found {len(unused_files)} potentially unused files:")
    for file in sorted(unused_files):
        print(f"  - {file}")
    
    # Find highly imported files
    highly_imported = find_highly_imported_files(graph)
    print(f"Top 10 most imported files:")
    for file, in_degree in highly_imported[:10]:
        print(f"  - {file}: imported by {in_degree} files")
    
    # Generate visualization if requested
    if args.visualize:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'dependency_graph.png')
        visualize_dependency_graph(graph, output_path)

if __name__ == '__main__':
    main()
