"""
Civil Engineering Desktop Application
Development of a Desktop Application Integrating Graph Algorithms, 
PCA-Based Data Analysis, and Static and Dynamic Transportation Algorithms
in the Civil Engineering Sector

Author: [Kchibal Bilal]
Supervisor: Dr. EL MKHALET MOUNA
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA GENERATOR FOR MOCK DATA
# ============================================================================
class DataGenerator:
    """Generate mock civil engineering data for demonstration"""
    
    @staticmethod
    def generate_civil_engineering_data(n_samples=100):
        """Generate mock data for civil engineering analysis"""
        np.random.seed(42)
        data = {
            'Concrete_Strength_MPa': np.random.normal(30, 5, n_samples),
            'Water_Cement_Ratio': np.random.uniform(0.4, 0.6, n_samples),
            'Aggregate_Size_mm': np.random.uniform(10, 25, n_samples),
            'Slump_mm': np.random.normal(100, 20, n_samples),
            'Curing_Time_days': np.random.randint(7, 28, n_samples),
            'Temperature_C': np.random.normal(20, 5, n_samples),
            'Humidity_percent': np.random.uniform(60, 90, n_samples),
            'Density_kg_m3': np.random.normal(2400, 100, n_samples)
        }
        return pd.DataFrame(data)

# ============================================================================
# MODULE 1: PCA DATA ANALYSIS
# ============================================================================
class PCAAnalysisModule:
    """Module for Principal Component Analysis"""
    
    def __init__(self, data=None):
        self.data = data if data is not None else DataGenerator.generate_civil_engineering_data()
        self.scaler = StandardScaler()
        self.pca = None
        self.scaled_data = None
        self.pca_data = None
        
    def compute_statistics(self):
        """Compute means and standard deviations"""
        stats = pd.DataFrame({
            'Mean': self.data.mean(),
            'Std Dev': self.data.std()
        })
        return stats
    
    def compute_correlation_matrix(self):
        """Compute correlation matrix"""
        return self.data.corr()
    
    def perform_pca(self, n_components=2):
        """Perform PCA analysis"""
        self.scaled_data = self.scaler.fit_transform(self.data)
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        return self.pca
    
    def get_inertia(self):
        """Get explained variance (inertia)"""
        if self.pca is None:
            self.perform_pca()
        return self.pca.explained_variance_ratio_
    
    def get_contributions(self, component=0):
        """Get variable contributions to principal component"""
        if self.pca is None:
            self.perform_pca()
        contributions = pd.DataFrame(
            self.pca.components_[component],
            index=self.data.columns,
            columns=[f'PC{component+1}']
        )
        return contributions

# ============================================================================
# MODULE 2: GRAPH ALGORITHMS (OPERATIONAL RESEARCH)
# ============================================================================
class GraphAlgorithms:
    """Module for graph-based operational research algorithms"""
    
    @staticmethod
    def generate_graph(n_vertices, density):
        """Generate a random weighted graph"""
        G = nx.Graph()
        G.add_nodes_from(range(n_vertices))
        
        # Add edges based on density
        max_edges = n_vertices * (n_vertices - 1) // 2
        n_edges = int(max_edges * density / 100)
        
        edges_added = 0
        while edges_added < n_edges:
            u = np.random.randint(0, n_vertices)
            v = np.random.randint(0, n_vertices)
            if u != v and not G.has_edge(u, v):
                weight = np.random.randint(1, 20)
                G.add_edge(u, v, weight=weight)
                edges_added += 1
        
        return G
    
    @staticmethod
    def welsh_powell(G):
        """Welsh-Powell graph coloring algorithm"""
        return nx.greedy_color(G, strategy='largest_first')
    
    @staticmethod
    def dijkstra(G, source):
        """Dijkstra's shortest path algorithm"""
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            paths = nx.single_source_dijkstra_path(G, source, weight='weight')
            return lengths, paths
        except:
            return {}, {}
    
    @staticmethod
    def kruskal(G):
        """Kruskal's minimum spanning tree algorithm"""
        return nx.minimum_spanning_tree(G, weight='weight')
    
    @staticmethod
    def bellman_ford(G, source):
        """Bellman-Ford algorithm for shortest paths"""
        try:
            lengths = nx.single_source_bellman_ford_path_length(G, source, weight='weight')
            paths = nx.single_source_bellman_ford_path(G, source, weight='weight')
            return lengths, paths
        except:
            return {}, {}
    
    @staticmethod
    def ford_fulkerson(G, source, sink):
        """Ford-Fulkerson maximum flow algorithm"""
        try:
            flow_value = nx.maximum_flow_value(G, source, sink, capacity='weight')
            flow_dict = nx.maximum_flow(G, source, sink, capacity='weight')
            return flow_value, flow_dict
        except:
            return 0, {}

# ============================================================================
# MODULE 3: STATIC ALGORITHMS
# ============================================================================
class StaticAlgorithms:
    """Module for static transportation algorithms"""
    
    @staticmethod
    def floyd_warshall(adj_matrix):
        """Floyd-Warshall all-pairs shortest path"""
        n = len(adj_matrix)
        dist = np.array(adj_matrix, dtype=float)
        
        # Replace 0s with infinity except diagonal
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] == 0:
                    dist[i][j] = np.inf
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        return dist
    
    @staticmethod
    def wardrop_equilibrium(n_routes=3, n_iterations=100):
        """Simulate Wardrop User Equilibrium"""
        # Mock implementation for demonstration
        demand = 1000  # vehicles per hour
        routes = []
        
        for i in range(n_routes):
            # Travel time = free flow time + congestion term * flow
            free_flow = np.random.uniform(10, 30)
            congestion = np.random.uniform(0.01, 0.05)
            routes.append({'free_flow': free_flow, 'congestion': congestion, 'flow': 0})
        
        # Iterative assignment
        for iteration in range(n_iterations):
            # Calculate travel times
            travel_times = [r['free_flow'] + r['congestion'] * r['flow'] for r in routes]
            
            # Find minimum travel time route
            min_idx = np.argmin(travel_times)
            
            # Shift flow to minimum route
            for i, route in enumerate(routes):
                if i == min_idx:
                    route['flow'] += demand / n_iterations
                    
        return routes

# ============================================================================
# MODULE 4: DYNAMIC ALGORITHMS
# ============================================================================
class DynamicAlgorithms:
    """Module for dynamic transportation algorithms"""
    
    @staticmethod
    def cell_transmission_model(n_cells=10, n_steps=100):
        """Cell Transmission Model (CTM) simulation"""
        # Parameters
        max_density = 200  # veh/km
        max_flow = 2000    # veh/hour
        
        # Initialize cells
        density = np.zeros((n_steps, n_cells))
        density[0, :5] = max_density * 0.8  # Initial congestion
        
        # Simulate
        for t in range(1, n_steps):
            for i in range(n_cells):
                if i < n_cells - 1:
                    # Calculate flow between cells
                    flow = min(density[t-1, i] * 60, max_flow, 
                              (max_density - density[t-1, i+1]) * 60)
                    density[t, i] = density[t-1, i] - flow/60
                    density[t, i+1] = density[t-1, i+1] + flow/60
                else:
                    density[t, i] = density[t-1, i]
        
        return density
    
    @staticmethod
    def queue_model(arrival_rate=10, service_rate=12, n_steps=100):
        """Queue-based traffic model"""
        queue_length = np.zeros(n_steps)
        
        for t in range(1, n_steps):
            arrivals = np.random.poisson(arrival_rate)
            departures = min(queue_length[t-1] + arrivals, service_rate)
            queue_length[t] = queue_length[t-1] + arrivals - departures
        
        return queue_length
    
    @staticmethod
    def lwr_model(n_cells=50, n_steps=200):
        """Lighthill-Whitham-Richards (LWR) Model"""
        # Similar to CTM but with continuous density
        max_density = 180
        density = np.zeros((n_steps, n_cells))
        density[0, 10:20] = max_density * 0.7
        
        # LWR simulation with upwind scheme
        dx = 1.0  # km
        dt = 0.01  # hour
        
        for t in range(1, n_steps):
            for i in range(1, n_cells-1):
                # Simple upwind scheme
                density[t, i] = density[t-1, i] - dt/dx * (
                    density[t-1, i] - density[t-1, i-1]
                )
                density[t, i] = max(0, min(density[t, i], max_density))
        
        return density

# ============================================================================
# GUI APPLICATION
# ============================================================================
class CivilEngineeringApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Civil Engineering Analysis Application")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')
        
        # Initialize data
        self.pca_module = PCAAnalysisModule()
        self.graph_vertices = tk.IntVar(value=10)
        self.graph_density = tk.IntVar(value=30)
        
        # Show welcome screen
        self.show_welcome_screen()
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_welcome_screen(self):
        """Display welcome screen"""
        self.clear_window()
        
        frame = tk.Frame(self.root, bg='#2C3E50')
        frame.place(relx=0.5, rely=0.5, anchor='center')
        
        title = tk.Label(frame, 
                        text="Development of a Desktop Application\nIntegrating Graph Algorithms, PCA-Based Data Analysis,\nand Static and Dynamic Transportation Algorithms\nin the Civil Engineering Sector",
                        font=('Arial', 16, 'bold'),
                        bg='#2C3E50',
                        fg='white',
                        justify='center')
        title.pack(pady=20)
        
        student = tk.Label(frame,
                          text="Student: [Kchibal Bilal]",
                          font=('Arial', 12),
                          bg='#2C3E50',
                          fg='#ECF0F1')
        student.pack(pady=5)
        
        supervisor = tk.Label(frame,
                             text="Supervisor: Dr. EL MKHALET MOUNA",
                             font=('Arial', 12, 'italic'),
                             bg='#2C3E50',
                             fg='#ECF0F1')
        supervisor.pack(pady=5)
        
        start_btn = tk.Button(frame,
                             text="START",
                             command=self.show_main_dashboard,
                             font=('Arial', 14, 'bold'),
                             bg='#27AE60',
                             fg='white',
                             padx=50,
                             pady=15,
                             relief='raised',
                             cursor='hand2')
        start_btn.pack(pady=30)
    
    def show_main_dashboard(self):
        """Display main dashboard with three main buttons"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root,
                        text="Main Dashboard",
                        font=('Arial', 20, 'bold'),
                        bg='#2C3E50',
                        fg='white')
        title.pack(pady=30)
        
        # Button frame
        btn_frame = tk.Frame(self.root, bg='#2C3E50')
        btn_frame.pack(expand=True)
        
        # Data Analyst Button
        data_btn = tk.Button(btn_frame,
                            text="Data-Analyst PCA",
                            command=self.show_pca_module,
                            font=('Arial', 14, 'bold'),
                            bg='#3498DB',
                            fg='white',
                            width=25,
                            height=3,
                            relief='raised',
                            cursor='hand2')
        data_btn.pack(pady=15)
        
        # Operational Research Button
        or_btn = tk.Button(btn_frame,
                          text="Operational Research",
                          command=self.show_graph_module,
                          font=('Arial', 14, 'bold'),
                          bg='#E74C3C',
                          fg='white',
                          width=25,
                          height=3,
                          relief='raised',
                          cursor='hand2')
        or_btn.pack(pady=15)
        
        # Static & Dynamic Algorithms Button
        algo_btn = tk.Button(btn_frame,
                            text="Static & Dynamic Transportation",
                            command=self.show_algorithms_menu,
                            font=('Arial', 14, 'bold'),
                            bg='#9B59B6',
                            fg='white',
                            width=25,
                            height=3,
                            relief='raised',
                            cursor='hand2')
        algo_btn.pack(pady=15)
    
    def create_back_button(self, command):
        """Create a back button"""
        back_btn = tk.Button(self.root,
                            text="← Back",
                            command=command,
                            font=('Arial', 10),
                            bg='#95A5A6',
                            fg='white',
                            padx=10,
                            pady=5)
        back_btn.place(x=10, y=10)
    
    # ========================================================================
    # PCA MODULE GUI
    # ========================================================================
    def show_pca_module(self):
        """Display PCA analysis module"""
        self.clear_window()
        self.create_back_button(self.show_main_dashboard)
        
        title = tk.Label(self.root,
                        text="Data-Analyst PCA Module",
                        font=('Arial', 18, 'bold'),
                        bg='#2C3E50',
                        fg='white')
        title.pack(pady=20)
        
        btn_frame = tk.Frame(self.root, bg='#2C3E50')
        btn_frame.pack(expand=True)
        
        buttons = [
            ("Table of Means/SD", self.show_statistics),
            ("Correlation Matrix (Heatmap)", self.show_correlation),
            ("Computation of Inertias", self.show_inertia),
            ("Factorial Planes (Individuals)", self.show_factorial_individuals),
            ("Factorial Planes (Variables)", self.show_factorial_variables),
            ("Contribution Analysis", self.show_contributions)
        ]
        
        for text, command in buttons:
            btn = tk.Button(btn_frame,
                           text=text,
                           command=command,
                           font=('Arial', 12),
                           bg='#3498DB',
                           fg='white',
                           width=30,
                           height=2,
                           relief='raised')
            btn.pack(pady=8)
    
    def show_statistics(self):
        """Display statistics table"""
        stats = self.pca_module.compute_statistics()
        self.show_dataframe_window("Statistics: Means and Standard Deviations", stats)
    
    def show_correlation(self):
        """Display correlation matrix heatmap"""
        corr = self.pca_module.compute_correlation_matrix()
        
        window = tk.Toplevel(self.root)
        window.title("Correlation Matrix Heatmap")
        window.geometry("800x700")
        
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, square=True)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_inertia(self):
        """Display inertia (explained variance)"""
        inertia = self.pca_module.get_inertia()
        
        window = tk.Toplevel(self.root)
        window.title("Inertia - Explained Variance")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        components = [f'PC{i+1}' for i in range(len(inertia))]
        ax.bar(components, inertia * 100, color='#3498DB')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance (%)', fontsize=12)
        ax.set_title('Explained Variance by Principal Component', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add cumulative line
        cumsum = np.cumsum(inertia * 100)
        ax2 = ax.twinx()
        ax2.plot(components, cumsum, color='#E74C3C', marker='o', 
                linewidth=2, label='Cumulative')
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12)
        ax2.legend()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_factorial_individuals(self):
        """Display factorial plane with individuals"""
        self.pca_module.perform_pca(n_components=2)
        
        window = tk.Toplevel(self.root)
        window.title("Factorial Plane - Individuals")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(self.pca_module.pca_data[:, 0],
                           self.pca_module.pca_data[:, 1],
                           c=range(len(self.pca_module.pca_data)),
                           cmap='viridis',
                           alpha=0.6)
        
        ax.set_xlabel(f'PC1 ({self.pca_module.pca.explained_variance_ratio_[0]*100:.1f}%)',
                     fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca_module.pca.explained_variance_ratio_[1]*100:.1f}%)',
                     fontsize=12)
        ax.set_title('Factorial Plane - Individuals Projection',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        plt.colorbar(scatter, ax=ax, label='Individual ID')
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_factorial_variables(self):
        """Display factorial plane with variables (correlation circle)"""
        self.pca_module.perform_pca(n_components=2)
        
        window = tk.Toplevel(self.root)
        window.title("Factorial Plane - Variables")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot correlation circle
        for i, var in enumerate(self.pca_module.data.columns):
            x = self.pca_module.pca.components_[0, i]
            y = self.pca_module.pca.components_[1, i]
            ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, 
                    fc='#E74C3C', ec='#E74C3C')
            ax.text(x*1.15, y*1.15, var, fontsize=9, ha='center')
        
        # Draw circle
        circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('Correlation Circle - Variables', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_aspect('equal')
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_contributions(self):
        """Display contribution analysis"""
        contrib = self.pca_module.get_contributions(0)
        contrib['PC2'] = self.pca_module.get_contributions(1)
        self.show_dataframe_window("Variable Contributions to PCs", contrib)
    
    def show_dataframe_window(self, title, df):
        """Display dataframe in a new window"""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("600x400")
        
        frame = tk.Frame(window)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create Treeview
        tree = ttk.Treeview(frame)
        tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)
        
        # Configure columns
        tree["columns"] = list(df.columns)
        tree["show"] = "tree headings"
        
        # Format columns
        tree.column("#0", width=150)
        tree.heading("#0", text="Index")
        
        for col in df.columns:
            tree.column(col, width=100, anchor='center')
            tree.heading(col, text=col)
        
        # Insert data
        for idx, row in df.iterrows():
            values = [f"{val:.4f}" if isinstance(val, (int, float)) else val 
                     for val in row.values]
            tree.insert("", "end", text=idx, values=values)
    
    # ========================================================================
    # GRAPH MODULE GUI
    # ========================================================================
    def show_graph_module(self):
        """Display operational research graph module"""
        self.clear_window()
        self.create_back_button(self.show_main_dashboard)
        
        title = tk.Label(self.root,
                        text="Operational Research - Graph Algorithms",
                        font=('Arial', 18, 'bold'),
                        bg='#2C3E50',
                        fg='white')
        title.pack(pady=20)
        
        # Controls frame
        control_frame = tk.Frame(self.root, bg='#2C3E50')
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="Number of Vertices:",
                bg='#2C3E50', fg='white', font=('Arial', 10)).grid(row=0, column=0, padx=5)
        tk.Spinbox(control_frame, from_=1, to=100, textvariable=self.graph_vertices,
                  width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(control_frame, text="Graph Density (%):",
                bg='#2C3E50', fg='white', font=('Arial', 10)).grid(row=0, column=2, padx=5)
        tk.Spinbox(control_frame, from_=1, to=100, textvariable=self.graph_density,
                  width=10).grid(row=0, column=3, padx=5)
        
        # Algorithm buttons
        btn_frame = tk.Frame(self.root, bg='#2C3E50')
        btn_frame.pack(expand=True)
        
        algorithms = [
            ("Welsh-Powell Coloring", self.run_welsh_powell),
            ("Dijkstra Shortest Path", self.run_dijkstra),
            ("Kruskal MST", self.run_kruskal),
            ("Bellman-Ford", self.run_bellman_ford),
            ("Ford-Fulkerson Max Flow", self.run_ford_fulkerson)
        ]
        
        for text, command in algorithms:
            btn = tk.Button(btn_frame,
                           text=text,
                           command=command,
                           font=('Arial', 12),
                           bg='#E74C3C',
                           fg='white',
                           width=30,
                           height=2,
                           relief='raised')
            btn.pack(pady=8)
    
    def run_welsh_powell(self):
        """Run Welsh-Powell algorithm"""
        G = GraphAlgorithms.generate_graph(self.graph_vertices.get(), 
                                          self.graph_density.get())
        coloring = GraphAlgorithms.welsh_powell(G)
        
        window = tk.Toplevel(self.root)
        window.title("Welsh-Powell Graph Coloring")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        colors = [coloring[node] for node in G.nodes()]
        
        nx.draw(G, pos, node_color=colors, with_labels=True, 
               node_size=500, cmap='Set3', ax=ax, font_weight='bold')
        ax.set_title(f"Graph Coloring - {max(coloring.values())+1} colors used",
                    fontsize=14, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_dijkstra(self):
        """Run Dijkstra algorithm"""
        G = GraphAlgorithms.generate_graph(self.graph_vertices.get(),
                                          self.graph_density.get())
        
        if len(G.nodes()) == 0:
            messagebox.showwarning("Warning", "Graph has no nodes!")
            return
        
        source = 0
        lengths, paths = GraphAlgorithms.dijkstra(G, source)
        
        window = tk.Toplevel(self.root)
        window.title("Dijkstra Shortest Paths")
        window.geometry("900x600")
        
        # Split window
        left_frame = tk.Frame(window)
        left_frame.pack(side='left', fill='both', expand=True)
        
        right_frame = tk.Frame(window)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Graph visualization
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=500, ax=ax, font_weight='bold')
        
        # Highlight source
        nx.draw_networkx_nodes(G, pos, [source], node_color='red',
                              node_size=600, ax=ax)
        
        ax.set_title(f"Shortest Paths from Node {source}",
                    fontsize=12, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Results table
        text_widget = tk.Text(right_frame, wrap='word', font=('Courier', 9))
        text_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        text_widget.insert('1.0', f"Shortest Distances from Node {source}:\n")
        text_widget.insert('end', "="*40 + "\n")
        for node, dist in sorted(lengths.items()):
            text_widget.insert('end', f"Node {node}: {dist:.2f}\n")
        
        text_widget.config(state='disabled')
    
    def run_kruskal(self):
        """Run Kruskal MST algorithm"""
        G = GraphAlgorithms.generate_graph(self.graph_vertices.get(),
                                          self.graph_density.get())
        mst = GraphAlgorithms.kruskal(G)
        
        window = tk.Toplevel(self.root)
        window.title("Kruskal Minimum Spanning Tree")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        
        # Draw original graph in light gray
        nx.draw(G, pos, with_labels=True, node_color='lightgray',
               edge_color='lightgray', node_size=500, ax=ax, 
               font_weight='bold', alpha=0.3)
        
        # Draw MST in color
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, ax=ax)
        nx.draw_networkx_edges(mst, pos, edge_color='red', width=2, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold')
        
        total_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
        ax.set_title(f"Minimum Spanning Tree - Total Weight: {total_weight:.2f}",
                    fontsize=14, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_bellman_ford(self):
        """Run Bellman-Ford algorithm"""
        G = GraphAlgorithms.generate_graph(self.graph_vertices.get(),
                                          self.graph_density.get())
        
        if len(G.nodes()) == 0:
            messagebox.showwarning("Warning", "Graph has no nodes!")
            return
        
        source = 0
        lengths, paths = GraphAlgorithms.bellman_ford(G, source)
        
        window = tk.Toplevel(self.root)
        window.title("Bellman-Ford Algorithm")
        window.geometry("900x600")
        
        # Split window
        left_frame = tk.Frame(window)
        left_frame.pack(side='left', fill='both', expand=True)
        
        right_frame = tk.Frame(window)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Graph visualization
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightgreen',
               node_size=500, ax=ax, font_weight='bold')
        
        nx.draw_networkx_nodes(G, pos, [source], node_color='darkgreen',
                              node_size=600, ax=ax)
        
        ax.set_title(f"Bellman-Ford from Node {source}",
                    fontsize=12, fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Results
        text_widget = tk.Text(right_frame, wrap='word', font=('Courier', 9))
        text_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        text_widget.insert('1.0', f"Shortest Paths from Node {source}:\n")
        text_widget.insert('end', "="*40 + "\n")
        for node, dist in sorted(lengths.items()):
            text_widget.insert('end', f"Node {node}: {dist:.2f}\n")
        
        text_widget.config(state='disabled')
    
    def run_ford_fulkerson(self):
        """Run Ford-Fulkerson max flow algorithm"""
        G = GraphAlgorithms.generate_graph(self.graph_vertices.get(),
                                          self.graph_density.get())
        
        if len(G.nodes()) < 2:
            messagebox.showwarning("Warning", "Need at least 2 nodes!")
            return
        
        source = 0
        sink = max(G.nodes())
        flow_value, flow_dict = GraphAlgorithms.ford_fulkerson(G, source, sink)
        
        window = tk.Toplevel(self.root)
        window.title("Ford-Fulkerson Maximum Flow")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightyellow',
               node_size=500, ax=ax, font_weight='bold')
        
        # Highlight source and sink
        nx.draw_networkx_nodes(G, pos, [source], node_color='green',
                              node_size=600, ax=ax, label='Source')
        nx.draw_networkx_nodes(G, pos, [sink], node_color='red',
                              node_size=600, ax=ax, label='Sink')
        
        ax.set_title(f"Maximum Flow: {flow_value:.2f} (Source: {source}, Sink: {sink})",
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    # ========================================================================
    # STATIC & DYNAMIC ALGORITHMS MODULE
    # ========================================================================
    def show_algorithms_menu(self):
        """Display algorithms selection menu"""
        self.clear_window()
        self.create_back_button(self.show_main_dashboard)
        
        title = tk.Label(self.root,
                        text="Static & Dynamic Transportation Algorithms",
                        font=('Arial', 18, 'bold'),
                        bg='#2C3E50',
                        fg='white')
        title.pack(pady=20)
        
        frame = tk.Frame(self.root, bg='#2C3E50')
        frame.pack(expand=True)
        
        # Static Algorithms
        static_label = tk.Label(frame, text="Static Algorithms",
                               font=('Arial', 14, 'bold'),
                               bg='#2C3E50', fg='#ECF0F1')
        static_label.pack(pady=10)
        
        static_btns = [
            ("Floyd-Warshall Algorithm", self.run_floyd_warshall),
            ("Wardrop Equilibrium", self.run_wardrop)
        ]
        
        for text, command in static_btns:
            btn = tk.Button(frame, text=text, command=command,
                           font=('Arial', 11), bg='#16A085', fg='white',
                           width=35, height=2, relief='raised')
            btn.pack(pady=5)
        
        # Separator
        tk.Label(frame, text="", bg='#2C3E50').pack(pady=10)
        
        # Dynamic Algorithms
        dynamic_label = tk.Label(frame, text="Dynamic Algorithms",
                                font=('Arial', 14, 'bold'),
                                bg='#2C3E50', fg='#ECF0F1')
        dynamic_label.pack(pady=10)
        
        dynamic_btns = [
            ("Cell Transmission Model (CTM)", self.run_ctm),
            ("Queue-Based Model", self.run_queue_model),
            ("LWR Model", self.run_lwr),
            ("Microscopic Traffic Simulation", self.run_microscopic)
        ]
        
        for text, command in dynamic_btns:
            btn = tk.Button(frame, text=text, command=command,
                           font=('Arial', 11), bg='#8E44AD', fg='white',
                           width=35, height=2, relief='raised')
            btn.pack(pady=5)
    
    def run_floyd_warshall(self):
        """Run Floyd-Warshall algorithm"""
        n = 5  # Example with 5 nodes
        
        # Create random adjacency matrix
        adj_matrix = np.random.randint(1, 20, (n, n))
        np.fill_diagonal(adj_matrix, 0)
        
        result = StaticAlgorithms.floyd_warshall(adj_matrix)
        
        window = tk.Toplevel(self.root)
        window.title("Floyd-Warshall All-Pairs Shortest Paths")
        window.geometry("900x600")
        
        # Create notebook for before/after
        notebook = ttk.Notebook(window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Original matrix
        frame1 = tk.Frame(notebook)
        notebook.add(frame1, text="Original Adjacency Matrix")
        
        fig1 = Figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        im1 = ax1.imshow(adj_matrix, cmap='Blues', aspect='auto')
        ax1.set_title("Original Distance Matrix", fontsize=14, fontweight='bold')
        
        for i in range(n):
            for j in range(n):
                text = ax1.text(j, i, f'{adj_matrix[i, j]}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        fig1.colorbar(im1, ax=ax1)
        
        canvas1 = FigureCanvasTkAgg(fig1, frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        # Result matrix
        frame2 = tk.Frame(notebook)
        notebook.add(frame2, text="Shortest Paths Matrix")
        
        fig2 = Figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        
        result_display = result.copy()
        result_display[result_display == np.inf] = 999
        
        im2 = ax2.imshow(result_display, cmap='Greens', aspect='auto')
        ax2.set_title("All-Pairs Shortest Paths", fontsize=14, fontweight='bold')
        
        for i in range(n):
            for j in range(n):
                val = result[i, j]
                text_val = f'{val:.0f}' if val != np.inf else '∞'
                ax2.text(j, i, text_val, ha="center", va="center",
                        color="black", fontweight='bold')
        
        fig2.colorbar(im2, ax=ax2)
        
        canvas2 = FigureCanvasTkAgg(fig2, frame2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)
    
    def run_wardrop(self):
        """Run Wardrop Equilibrium simulation"""
        routes = StaticAlgorithms.wardrop_equilibrium(n_routes=4, n_iterations=100)
        
        window = tk.Toplevel(self.root)
        window.title("Wardrop User Equilibrium")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        route_names = [f'Route {i+1}' for i in range(len(routes))]
        flows = [r['flow'] for r in routes]
        travel_times = [r['free_flow'] + r['congestion'] * r['flow'] for r in routes]
        
        x = np.arange(len(routes))
        width = 0.35
        
        ax.bar(x - width/2, flows, width, label='Flow (veh/h)', color='#3498DB')
        ax2 = ax.twinx()
        ax2.bar(x + width/2, travel_times, width, label='Travel Time (min)', color='#E74C3C')
        
        ax.set_xlabel('Routes', fontsize=12)
        ax.set_ylabel('Flow (vehicles/hour)', fontsize=12, color='#3498DB')
        ax2.set_ylabel('Travel Time (minutes)', fontsize=12, color='#E74C3C')
        ax.set_title('Wardrop User Equilibrium - Route Distribution',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(route_names)
        
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_ctm(self):
        """Run Cell Transmission Model"""
        density = DynamicAlgorithms.cell_transmission_model(n_cells=15, n_steps=80)
        
        window = tk.Toplevel(self.root)
        window.title("Cell Transmission Model (CTM)")
        window.geometry("900x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(density.T, aspect='auto', cmap='hot', origin='lower',
                      extent=[0, density.shape[0], 0, density.shape[1]])
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Cell Number', fontsize=12)
        ax.set_title('Cell Transmission Model - Traffic Density Evolution',
                    fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density (veh/km)', fontsize=10)
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_queue_model(self):
        """Run Queue-Based Model"""
        queue = DynamicAlgorithms.queue_model(arrival_rate=10, service_rate=12, n_steps=100)
        
        window = tk.Toplevel(self.root)
        window.title("Queue-Based Traffic Model")
        window.geometry("800x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(queue, linewidth=2, color='#3498DB')
        ax.fill_between(range(len(queue)), queue, alpha=0.3, color='#3498DB')
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Queue Length (vehicles)', fontsize=12)
        ax.set_title('Queue-Based Model - Queue Evolution',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        avg_queue = np.mean(queue)
        max_queue = np.max(queue)
        ax.axhline(y=avg_queue, color='red', linestyle='--', 
                  label=f'Average: {avg_queue:.1f}')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_lwr(self):
        """Run LWR Model"""
        density = DynamicAlgorithms.lwr_model(n_cells=50, n_steps=150)
        
        window = tk.Toplevel(self.root)
        window.title("Lighthill-Whitham-Richards (LWR) Model")
        window.geometry("900x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(density.T, aspect='auto', cmap='viridis', origin='lower',
                      extent=[0, density.shape[0], 0, density.shape[1]])
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Space Cell', fontsize=12)
        ax.set_title('LWR Model - Traffic Wave Propagation',
                    fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density (veh/km)', fontsize=10)
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def run_microscopic(self):
        """Run Microscopic Traffic Simulation"""
        n_vehicles = 50
        n_steps = 100
        road_length = 1000  # meters
        
        # Initialize vehicles
        positions = np.random.uniform(0, road_length, n_vehicles)
        velocities = np.random.uniform(10, 25, n_vehicles)  # m/s
        
        # Simulation
        history = np.zeros((n_steps, n_vehicles))
        
        for t in range(n_steps):
            history[t] = positions.copy()
            
            # Simple car-following model
            for i in range(n_vehicles):
                # Find vehicle ahead
                others = positions[positions > positions[i]]
                if len(others) > 0:
                    gap = np.min(others) - positions[i]
                    if gap < 50:  # Safety distance
                        velocities[i] = max(0, velocities[i] - 2)
                    else:
                        velocities[i] = min(25, velocities[i] + 1)
                
                # Update position
                positions[i] += velocities[i] * 0.1  # dt = 0.1s
                
                # Wrap around
                if positions[i] > road_length:
                    positions[i] = 0
        
        window = tk.Toplevel(self.root)
        window.title("Microscopic Traffic Simulation")
        window.geometry("900x600")
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for i in range(n_vehicles):
            ax.plot(history[:, i], alpha=0.6, linewidth=1)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Position (m)', fontsize=12)
        ax.set_title('Microscopic Traffic Simulation - Vehicle Trajectories',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = CivilEngineeringApp(root)
    root.mainloop()


"""
=============================================================================
INSTRUCTIONS FOR CONVERTING TO .EXE FILE
=============================================================================

1. Install PyInstaller:
   pip install pyinstaller

2. Navigate to the directory containing this Python file

3. Run one of these commands:

   For a single file executable:
   pyinstaller --onefile --windowed --name="CivilEngineeringApp" your_script_name.py

   For a directory-based executable (faster startup):
   pyinstaller --windowed --name="CivilEngineeringApp" your_script_name.py

4. The executable will be created in the 'dist' folder

5. If you get import errors, use:
   pyinstaller --onefile --windowed --hidden-import=sklearn.utils._typedefs --hidden-import=sklearn.neighbors._partition_nodes --name="CivilEngineeringApp" your_script_name.py

Additional Notes:
- --windowed: Prevents console window from appearing
- --onefile: Creates a single executable file
- --name: Sets the name of your application
- The first run may take time as PyInstaller analyzes dependencies

Alternative using Auto-py-to-exe (GUI):
1. pip install auto-py-to-exe
2. Run: auto-py-to-exe
3. Select your script and configure options through the GUI
4. Click "Convert .py to .exe"

Requirements file (requirements.txt):
numpy
pandas
matplotlib
seaborn
scikit-learn
networkx
pillow

Install all at once with: pip install -r requirements.txt
=============================================================================
"""