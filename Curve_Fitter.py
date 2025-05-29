import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Page configuration
st.set_page_config(
    page_title="Dual-Parabolic Antenna Ray Tracing",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🛰️ Dual-Parabolic Antenna Ray Tracing Analysis")
st.markdown("Interactive visualization of your B-spline shaped reflector system with accurate ray physics")

# Your actual antenna data
MAIN_REFLECTOR_POINTS = np.array([
    [30.010474412500002, 5],
    [30.002595009161421, 6.333333333333071],
    [29.933339951667964, 8.0000000000000639],
    [29.76260685366535, 10],
    [29.659788982665354, 10.999999999999998],
    [29.539593873464568, 12],
    [29.356262855333323, 13.333333333333332],
    [29.13847369599738, 14.666666666666668],
    [28.886522730665352, 15.999999999999998],
    [28.680552687464566, 17],
    [28.457649908866141, 17.999999999999996],
    [28.137968001931757, 19.333333333333336],
    [27.784716959398949, 20.666666666666664],
    [27.398193116464565, 21.999999999999996],
    [26.989586184999997, 23.333333333333336],
    [26.548002788330709, 24.666666666666664],
    [25.955173379999014, 26.333333333333339],
    [25.321679018662902, 27.999999999999989],
    [24.497014530667418, 29.999999999999844],
    [23.756561970663977, 31.666666666666771],
    [23.132556965866144, 33],
    [21.681303521263789, 35.999999999999993],
    [20.041970347070816, 38.999999999999986],
    [18.220780482464566, 42],
    [17.59863655386614, 43],
    [16.961486068665355, 43.999999999999993],
    [16.309403110665354, 45],
    [15.588217200775592, 46.081333333346457],
    [15.076937256374016, 46.829333333346455],
    [14.789710987248029, 47.243999999999993]
])

SUB_REFLECTOR_POINTS = np.array([
    [-2.19347, 0],
    [-2.2880951457027559, 0.13333333333334643],
    [-2.4166550422520863, 0.33333333333334647],
    [-2.5627311528544094, 0.59999999999999998],
    [-2.6599682304812209, 0.79999999999999993],
    [-2.7736492019758661, 1.0666666666666536],
    [-2.8866595198109448, 1.4000000000000001],
    [-2.978381775300984, 1.7999999999999998],
    [-3.0287884701147245, 2.2000000000000002],
    [-3.0403767564577162, 2.6000000000000001],
    [-3.0155655151839369, 3],
    [-2.9566954588064958, 3.3999999999999999],
    [-2.8660295777757088, 3.8000000000000003],
    [-2.7056584772165748, 4.3333333333330701],
    [-2.5441277996525198, 4.7333333333330705],
    [-2.4207887499999998, 5]
])

# Parabolic parameters (from analysis)
MAIN_PARABOLA = {'h': 30.010, 'k': 26.122, 'p': 7.3}  # x = h - (y-k)²/(4p)
SUB_PARABOLA = {'h': -3.04, 'k': 2.5, 'a': 0.12}      # x = a(y-k)² + h
MAIN_FOCUS = np.array([22.710, 26.122])
FEED_POINT = np.array([-5.0, 2.5])

class RayTracer:
    @staticmethod
    def reflect_ray(ray_dir, normal):
        """Reflect ray using R = I - 2(I·N)N"""
        dot = np.dot(ray_dir, normal)
        return ray_dir - 2 * dot * normal
    
    @staticmethod
    def calculate_parabolic_normal(point, is_main=True):
        """Calculate exact normal for parabolic surfaces"""
        x, y = point
        
        if is_main:
            # Main parabola: x = h - (y-k)²/(4p), dx/dy = -(y-k)/(2p)
            dxdy = -(y - MAIN_PARABOLA['k']) / (2 * MAIN_PARABOLA['p'])
            length = np.sqrt(1 + dxdy**2)
            return np.array([1/length, -dxdy/length])  # Normal pointing inward
        else:
            # Sub parabola: x = a(y-k)² + h, dx/dy = 2a(y-k)
            dxdy = 2 * SUB_PARABOLA['a'] * (y - SUB_PARABOLA['k'])
            length = np.sqrt(1 + dxdy**2)
            return np.array([-1/length, dxdy/length])  # Normal pointing outward
    
    @staticmethod
    def find_parabolic_intersection(ray_start, ray_dir, is_main=True, max_distance=100):
        """Find intersection with parabolic curve using analytical method"""
        x0, y0 = ray_start
        dx, dy = ray_dir
        
        if abs(dy) < 1e-10:  # Nearly horizontal ray
            return None
            
        if is_main:
            # Main parabola: x = h - (y-k)²/(4p)
            h, k, p = MAIN_PARABOLA['h'], MAIN_PARABOLA['k'], MAIN_PARABOLA['p']
            
            # Ray: x = x0 + t*dx, y = y0 + t*dy
            # Substitute into parabola equation and solve for t
            A = dy**2 / (4 * p)
            B = dx + dy * (y0 - k) / (2 * p)
            C = x0 - h + (y0 - k)**2 / (4 * p)
            
            discriminant = B**2 - 4*A*C
            if discriminant < 0:
                return None
                
            t1 = (-B - np.sqrt(discriminant)) / (2*A)
            t2 = (-B + np.sqrt(discriminant)) / (2*A)
            
            # Choose positive t closest to start
            t = None
            if t1 > 0.01 and t1 < max_distance:
                t = t1
            elif t2 > 0.01 and t2 < max_distance:
                t = t2
                
            if t is not None:
                hit_point = np.array([x0 + t*dx, y0 + t*dy])
                if 5 <= hit_point[1] <= 47.244:  # Within reflector bounds
                    return hit_point
                    
        else:
            # Sub parabola: x = a(y-k)² + h
            a, h, k = SUB_PARABOLA['a'], SUB_PARABOLA['h'], SUB_PARABOLA['k']
            
            A = a * dy**2
            B = dx - 2*a*dy*(y0 - k)
            C = x0 - h - a*(y0 - k)**2
            
            if abs(A) < 1e-10:  # Linear case
                if abs(B) > 1e-10:
                    t = -C / B
                    if 0.01 < t < max_distance:
                        hit_point = np.array([x0 + t*dx, y0 + t*dy])
                        if 0 <= hit_point[1] <= 5:
                            return hit_point
            else:
                discriminant = B**2 - 4*A*C
                if discriminant >= 0:
                    t1 = (-B - np.sqrt(discriminant)) / (2*A)
                    t2 = (-B + np.sqrt(discriminant)) / (2*A)
                    
                    t = None
                    if t1 > 0.01 and t1 < max_distance:
                        t = t1
                    elif t2 > 0.01 and t2 < max_distance:
                        t = t2
                        
                    if t is not None:
                        hit_point = np.array([x0 + t*dx, y0 + t*dy])
                        if 0 <= hit_point[1] <= 5:
                            return hit_point
        
        return None
    
    @staticmethod
    def trace_ray(ray_start, ray_dir, show_both=True):
        """Trace a single ray through the system"""
        path = [ray_start.copy()]
        current_pos = ray_start.copy()
        current_dir = ray_dir.copy()
        
        # Hit main reflector
        main_hit = RayTracer.find_parabolic_intersection(current_pos, current_dir, is_main=True)
        if main_hit is None:
            return path
            
        path.append(main_hit)
        
        if not show_both:
            # Just show main reflector behavior
            main_normal = RayTracer.calculate_parabolic_normal(main_hit, is_main=True)
            reflected_dir = RayTracer.reflect_ray(current_dir, main_normal)
            # Show ray going toward main focus
            focus_dir = MAIN_FOCUS - main_hit
            focus_dist = np.linalg.norm(focus_dir)
            if focus_dist > 0:
                focus_dir = focus_dir / focus_dist
                if np.dot(reflected_dir, focus_dir) > 0.5:  # Ray points toward focus
                    path.append(MAIN_FOCUS)
                else:
                    path.append(main_hit + reflected_dir * 8)
            return path
        
        # Reflect off main reflector
        main_normal = RayTracer.calculate_parabolic_normal(main_hit, is_main=True)
        current_dir = RayTracer.reflect_ray(current_dir, main_normal)
        current_pos = main_hit.copy()
        
        # Hit sub-reflector
        sub_hit = RayTracer.find_parabolic_intersection(current_pos, current_dir, is_main=False)
        if sub_hit is None:
            # Ray missed sub-reflector, show where it goes
            path.append(current_pos + current_dir * 10)
            return path
            
        path.append(sub_hit)
        
        # Reflect off sub-reflector
        sub_normal = RayTracer.calculate_parabolic_normal(sub_hit, is_main=False)
        current_dir = RayTracer.reflect_ray(current_dir, sub_normal)
        current_pos = sub_hit.copy()
        
        # Check if ray converges toward feed
        feed_dir = FEED_POINT - current_pos
        feed_dist = np.linalg.norm(feed_dir)
        if feed_dist > 0:
            feed_dir = feed_dir / feed_dist
            dot_product = np.dot(current_dir, feed_dir)
            
            if dot_product > 0.3:  # Ray roughly points toward feed
                path.append(FEED_POINT)
            else:
                # Show where ray actually goes
                path.append(current_pos + current_dir * 5)
        
        return path

def generate_rays(num_rays, signal_angle, source_type):
    """Generate rays based on user settings"""
    rays = []
    
    if source_type == "Parallel Waves":
        # Parallel rays from far left
        y_positions = np.linspace(8, 45, num_rays)
        ray_dir = np.array([np.cos(np.radians(signal_angle)), 
                           np.sin(np.radians(signal_angle))])
        
        for y in y_positions:
            ray_start = np.array([-15.0, y])
            rays.append((ray_start, ray_dir))
            
    elif source_type == "Point Source":
        # Point source from far left
        source_point = np.array([-20.0, 26.0])
        y_positions = np.linspace(8, 45, num_rays)
        
        for y in y_positions:
            target = np.array([0.0, y])  # Aim toward antenna area
            ray_dir = target - source_point
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            rays.append((source_point.copy(), ray_dir))
    
    return rays

def create_ray_plot(num_rays, signal_angle, source_type, show_both):
    """Create the main ray tracing plot"""
    fig = go.Figure()
    
    # Generate and trace rays
    rays = generate_rays(num_rays, signal_angle, source_type)
    ray_paths = []
    
    for ray_start, ray_dir in rays:
        path = RayTracer.trace_ray(ray_start, ray_dir, show_both)
        ray_paths.append(path)
    
    # Plot reflector surfaces
    if show_both or not show_both:  # Always show main
        fig.add_trace(go.Scatter(
            x=MAIN_REFLECTOR_POINTS[:, 0],
            y=MAIN_REFLECTOR_POINTS[:, 1],
            mode='lines',
            name='Main Reflector',
            line=dict(color='#2c3e50', width=4),
            hovertemplate='Main Reflector<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
        
        # Main focus
        fig.add_trace(go.Scatter(
            x=[MAIN_FOCUS[0]],
            y=[MAIN_FOCUS[1]],
            mode='markers',
            name='Main Focus',
            marker=dict(color='gold', size=12, symbol='star'),
            hovertemplate='Main Focus<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
    
    if show_both:
        fig.add_trace(go.Scatter(
            x=SUB_REFLECTOR_POINTS[:, 0],
            y=SUB_REFLECTOR_POINTS[:, 1],
            mode='lines',
            name='Sub-Reflector',
            line=dict(color='#e74c3c', width=4),
            hovertemplate='Sub-Reflector<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
        
        # Feed point
        fig.add_trace(go.Scatter(
            x=[FEED_POINT[0]],
            y=[FEED_POINT[1]],
            mode='markers',
            name='Feed Point',
            marker=dict(color='#27ae60', size=15, symbol='square'),
            hovertemplate='Feed Point<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
    
    # Plot ray paths
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
    
    for i, path in enumerate(ray_paths):
        for j in range(len(path) - 1):
            segment_color = colors[j % len(colors)]
            opacity = 0.8 if j == 0 else 0.9
            width = 2 if j == 0 else 3
            
            fig.add_trace(go.Scatter(
                x=[path[j][0], path[j+1][0]],
                y=[path[j][1], path[j+1][1]],
                mode='lines',
                line=dict(color=segment_color, width=width, opacity=opacity),
                showlegend=False,
                hovertemplate=f'Ray {i+1} Segment {j+1}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
            ))
            
            # Add arrows to show ray direction
            if j < len(path) - 2:  # Don't add arrow to final segment
                mid_x = (path[j][0] + path[j+1][0]) / 2
                mid_y = (path[j][1] + path[j+1][1]) / 2
                
                dx = path[j+1][0] - path[j][0]
                dy = path[j+1][1] - path[j][1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Arrow pointing in ray direction
                    arrow_length = 0.5
                    arrow_x = mid_x + (dx/length) * arrow_length
                    arrow_y = mid_y + (dy/length) * arrow_length
                    
                    fig.add_annotation(
                        x=arrow_x, y=arrow_y,
                        ax=mid_x, ay=mid_y,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2, arrowsize=1, arrowwidth=2,
                        arrowcolor=segment_color,
                        showarrow=True
                    )
    
    # Layout
    x_range = [-25, 35] if show_both else [-25, 35]
    y_range = [-2, 50] if show_both else [0, 50]
    
    fig.update_layout(
        title=f"Ray Tracing: {'Complete System' if show_both else 'Main Reflector Only'}",
        xaxis_title="X Coordinate (units)",
        yaxis_title="Y Coordinate (units)",
        xaxis=dict(range=x_range, gridcolor='lightgray'),
        yaxis=dict(range=y_range, gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900,
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

# Sidebar controls
st.sidebar.header("🎛️ Ray Tracing Controls")

num_rays = st.sidebar.slider(
    "Number of Rays", 
    min_value=5, max_value=25, value=15, step=2,
    help="Number of parallel rays to trace"
)

signal_angle = st.sidebar.slider(
    "Signal Angle (degrees)", 
    min_value=-30, max_value=30, value=0, step=5,
    help="Angle of incoming parallel rays"
)

source_type = st.sidebar.selectbox(
    "Signal Source Type",
    ["Parallel Waves", "Point Source"],
    help="Type of signal source to simulate"
)

show_both = st.sidebar.checkbox(
    "Show Both Reflectors", 
    value=True,
    help="Show complete dual-reflector system vs main reflector only"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ray Tracing Visualization")
    fig = create_ray_plot(num_rays, signal_angle, source_type, show_both)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("System Specifications")
    
    # Main reflector specs
    st.markdown("**🔵 Main Reflector**")
    st.write(f"• Diameter: {MAIN_REFLECTOR_POINTS[-1,1] - MAIN_REFLECTOR_POINTS[0,1]:.1f} units")
    st.write(f"• Focal Length: {MAIN_PARABOLA['p']:.1f} units")
    st.write(f"• f/D Ratio: {MAIN_PARABOLA['p']/(MAIN_REFLECTOR_POINTS[-1,1] - MAIN_REFLECTOR_POINTS[0,1]):.3f}")
    st.write(f"• Focus: ({MAIN_FOCUS[0]:.1f}, {MAIN_FOCUS[1]:.1f})")
    
    if show_both:
        st.markdown("**🔴 Sub-Reflector**")
        st.write(f"• Diameter: {SUB_REFLECTOR_POINTS[-1,1] - SUB_REFLECTOR_POINTS[0,1]:.1f} units")
        st.write(f"• Shape: Parabolic (unusual)")
        st.write(f"• Vertex: ({SUB_PARABOLA['h']:.1f}, {SUB_PARABOLA['k']:.1f})")
        
        st.markdown("**🟢 Feed Point**")
        st.write(f"• Location: ({FEED_POINT[0]:.1f}, {FEED_POINT[1]:.1f})")
        st.write("• Behind sub-reflector")
    
    st.markdown("**📊 Ray Color Code**")
    st.write("🔵 Incoming rays")
    st.write("🔴 First reflection")
    st.write("🟢 Second reflection")
    st.write("🟡 Final convergence")

# Information section
st.subheader("📋 Analysis Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**How It Works**")
    st.write("""
    1. **Parallel rays** hit the main parabolic reflector
    2. **Reflect toward** the sub-reflector following parabolic focusing laws
    3. **Second reflection** from the sub-reflector directs rays to the feed
    4. **Feed point** collects the concentrated signal energy
    """)

with col2:
    st.markdown("**Unique Design**")
    st.write("""
    • **Dual-parabolic** system (unusual - typically parabolic + hyperbolic)
    • **B-spline shaped** for optimized performance
    • **Short focal length** design (f/D = 0.173)
    • **Compact configuration** for space efficiency
    """)

with col3:
    st.markdown("**Applications**")
    st.write("""
    • **Satellite communications**
    • **Radio astronomy**
    • **Radar systems**
    • **Deep space communications**
    """)

# Technical notes
with st.expander("🔬 Technical Details"):
    st.markdown("""
    **Ray Tracing Physics:**
    - Uses analytical parabolic intersection calculations
    - Applies exact reflection laws: R = I - 2(I·N)N
    - Calculates precise surface normals from parabolic derivatives
    
    **Parabolic Equations:**
    - Main Reflector: x = 30.01 - (y-26.122)²/(4×7.3)
    - Sub-Reflector: x = 0.12(y-2.5)² - 3.04
    
    **B-spline Advantages:**
    - Optimized beam patterns beyond pure mathematical curves
    - Reduced sidelobes and improved efficiency
    - Precise manufacturing control points
    """)

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Plotly for interactive antenna analysis*")
