import sys
sys.path.append('/home/yashaswi/.local/lib/python2.7/site-packages')

import numpy as np
from collections import deque
import json

class ConnectedComponent:
    def __init__(self, color, top_left, dimensions, node_id, bitmask):
        self.color = color
        self.top_left = top_left  # (row, col)
        self.dimensions = dimensions  # (height, width)
        self.node_id = node_id
        self.bitmask = bitmask
        self.children = []

    def __str__(self):
        # Format dimensions as "nxm" instead of "(n, m)"
        dim_str = f"({self.dimensions[0]}x{self.dimensions[1]})"
        children_str = " ".join([str(child) for child in self.children])
        return f"({self.color} {self.top_left} {dim_str} {self.node_id} {self.bitmask} {children_str})"

def find_connected_components(matrix):
    rows, cols = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    components = []
    next_id = 1

    # 8-neighbor directions: horizontal, vertical, and diagonal
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    # Find all connected components
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and matrix[i, j] != 0:
                color = matrix[i, j]
                # BFS to find all cells of this component
                queue = deque([(i, j)])
                visited[i, j] = True
                component_cells = [(i, j)]

                while queue:
                    r, c = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and matrix[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            component_cells.append((nr, nc))

                # Calculate bounding box
                min_row = min(cell[0] for cell in component_cells)
                max_row = max(cell[0] for cell in component_cells)
                min_col = min(cell[1] for cell in component_cells)
                max_col = max(cell[1] for cell in component_cells)

                top_left = (min_row, min_col)
                dimensions = (max_row - min_row + 1, max_col - min_col + 1)

                # Create bitmask relative to the bounding box
                bitmask = []
                for r in range(min_row, max_row + 1):
                    row_mask = []
                    for c in range(min_col, max_col + 1):
                        row_mask.append(1 if (r, c) in component_cells else 0)
                    bitmask.append(row_mask)

                # Create component
                component = ConnectedComponent(color, top_left, dimensions, next_id, bitmask)
                component.cells = set(component_cells)  # Store the actual cells for later use
                components.append(component)
                next_id += 1

    return components

def is_enclosed_by(inner, outer, matrix):
    """
    Check if inner component is completely enclosed by outer component.
    This means you cannot reach the inner component from the outside without
    passing through the outer component.
    """
    if inner.node_id == outer.node_id:
        return False

    rows, cols = matrix.shape

    # Create a copy of the matrix for flood fill
    temp_matrix = np.zeros_like(matrix)

    # Mark the outer component as 1
    for r, c in outer.cells:
        temp_matrix[r, c] = 1

    # Mark the inner component as 2
    for r, c in inner.cells:
        temp_matrix[r, c] = 2

    # Flood fill from outside the matrix
    queue = deque([(0, 0)])
    visited = np.zeros_like(temp_matrix, dtype=bool)
    visited[0, 0] = True

    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr, nc] and temp_matrix[nr, nc] != 1):

                visited[nr, nc] = True
                queue.append((nr, nc))

    # Check if any cell of the inner component was visited
    for r, c in inner.cells:
        if visited[r, c]:
            return False

    return True

def find_direct_parent(component, components, matrix):
    """Find the smallest component that directly encloses the given component"""
    direct_parent = None
    min_area = float('inf')

    for potential_parent in components:
        if potential_parent.node_id == component.node_id:
            continue

        if is_enclosed_by(component, potential_parent, matrix):
            area = potential_parent.dimensions[0] * potential_parent.dimensions[1]
            if area < min_area:
                min_area = area
                direct_parent = potential_parent

    return direct_parent

def build_component_tree(matrix):
    rows, cols = matrix.shape
    components = find_connected_components(matrix)

    # Create root node (background)
    root = ConnectedComponent(0, (0, 0), (rows, cols), 0, [[0 for _ in range(cols)] for _ in range(rows)])
    root.cells = set((r, c) for r in range(rows) for c in range(cols) if matrix[r, c] == 0)

    # Sort components by area (smallest first)
    components.sort(key=lambda c: c.dimensions[0] * c.dimensions[1])

    # Create dictionary to track parent-child relationships
    parent_map = {}

    # For each component, find its direct parent
    for comp in components:
        parent = find_direct_parent(comp, components, matrix)
        if parent:
            parent_map[comp.node_id] = parent
        else:
            parent_map[comp.node_id] = root

    # Add children to their parents
    for comp in components:
        parent = parent_map[comp.node_id]
        parent.children.append(comp)

    return root

def process_matrix(matrix):
    """Process the matrix and return the tree in infix notation"""
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    root = build_component_tree(matrix)
    return str(root)

def extract_trees_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    trees = {
        "train": [],
        "test": []
    }

    # Process training data
    for sample in data.get("train", []):
        input_matrix = sample["input"]
        output_matrix = sample["output"]

        input_tree = process_matrix(input_matrix)
        output_tree = process_matrix(output_matrix)

        trees["train"].append({
            "input_tree": input_tree,
            "output_tree": output_tree
        })

    # Process test data if available
    for sample in data.get("test", []):
        input_matrix = sample["input"]

        input_tree = process_matrix(input_matrix)

        trees["test"].append({
            "input_tree": input_tree
        })

    examples = ""

    for i, pair in enumerate(trees["train"]):
        examples += f"Train Example {i}\n"
        examples += f"Input Tree:\n"
        examples += pair["input_tree"]
        examples += "\nOutput tree:\n"
        examples += pair["output_tree"]
        examples += "\n\n"
    
    for i, pair in enumerate(trees["test"]):
        examples += f"Test Example \n"
        examples += f"Input Tree:\n"
        examples += pair["input_tree"]
    

    return examples
