"""
Work Package 4.2 Stiffness & Deflection Solver 
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Output directory setup
SCRATCH_DIR = "/workspace/scratch/wp4_stiffness"
OUT_DIR = "/workspace/out"
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# matplotlib style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# SECTION 1: input params
# ==============================================================================

# --- A. WING GEOMETRY & AIRFOIL (WP4.3) ---
b_total = None               # Total wingspan [m] (e.g., 46.18)
c_root = None                # Wing root chord [m] (e.g., 5.86)
c_tip = None                 # Wing tip chord [m] (e.g., 1.72)
t_over_c = None              # Airfoil max thickness-to-chord ratio t/c [-] (e.g., 0.121)

# Airfoil Spar Chordwise Locations (as fraction of chord c)
x_fs_frac = None             # Front spar station x/c (e.g., 0.15 for 15% c)
x_ms_frac = None             # Middle spar station x/c for multi-cell (e.g., 0.40 for 40% c)
x_rs_frac = None             # Rear spar station x/c (e.g., 0.65 for 65% c)

# Airfoil Section Height Factors (ratio of spar height to max thickness t_max)
h_fs_factor = None           # Height factor at front spar (e.g., 0.85 = 85% of t_max)
h_ms_factor = None           # Height factor at middle spar (e.g., 0.95 = 95% of t_max)
h_rs_factor = None           # Height factor at rear spar (e.g., 0.65 = 65% of t_max)

# --- B. MATERIAL PROPERTIES (AL2024-T81 default) ---
E_mat = None                 # Young's Modulus E [Pa] (e.g., 73.1e9)
G_mat = None                 # Shear Modulus G [Pa] (e.g., 28.0e9)
rho_mat = None               # Material Density rho [kg/m^3] (e.g., 2780.0)

# --- C. REGULATORY & STIFFNESS LIMITS (WP4 Reader) ---
deflection_limit_pct = None  # Max tip deflection as fraction of total wingspan (e.g., 0.15 = 15% b)
twist_limit_deg = None       # Max tip twist angle [deg] (e.g., 10.0)

# --- D. CRITICAL LOAD FACTORS & WP4.1 EXPORTED ARRAYS ---
n_pos = None                 # Critical positive limit load factor (e.g., +2.5)
n_neg = None                 # Critical negative limit load factor (e.g., -1.0)

# Benchmark curve root values (used if 1D arrays below are left as None)
M0_pos_benchmark = None      # Net root bending moment at +2.5g [N*m] (e.g., 5.25e6)
T0_pos_benchmark = None      # Net root torque at +2.5g [N*m] (e.g., 3.20e5)

# 1D arrays exported directly from WP4.1 (XFLR5 + Inertial integration):
""" 
add imported csv data below
posdata_csv = np.loadtxt('C:\workspace\scratch\WP4_1_loads_pos.csv', delimiter=',', skiprows=1)
example: y_wp41_input = posdata_csv[:, 0]
"""
#import csv data
#posdata_csv = np.loadtxt('C:\workspace\scratch\WP4_1_loads_pos.csv', delimiter=',', skiprows=1)
#leave skiprows 1 because 1 header 
y_wp41_input = None       # Spanwise stations array [m] from root (0) to tip (b/2)
Mx_pos_input = None          # Net Bending Moment array at +n [N*m]
Mx_neg_input = None         # Net Bending Moment array at -n [N*m]
T_pos_input = None           # Net Torque array at +n [N*m]
T_neg_input = None           # Net Torque array at -n [N*m]

# --- E. PRELIMINARY STRUCTURAL DESIGN OPTIONS (WP4.3) ---
user_options = None          # Leave None for default 3-option benchmark, or paste custom list shown below if using None adjust it in Section 2 (default values)

# ==============================================================================
# Example custom user options:
#IMPORT WING BOX DIMENSION VALUES FROM WP4.3
"""IMPORT WING BOX DIMENSION VALUES FROM WP4.3"""
#IMPORT WING BOX DIMENSION VALUES FROM WP4.3
# ==============================================================================
"""
user_options = [
    {
        "name": "Option 1 (Single-Cell 2-Spar)",
        "cell_type": "single_cell",      # 'single_cell' or 'multi_cell'
        "t_skin_root": 0.0100,           # Root skin thickness [m]
        "t_skin_tip": 0.0030,            # Tip skin thickness [m]
        "t_spar_root": 0.0120,           # Root spar web thickness [m]
        "t_spar_tip": 0.0040,            # Tip spar web thickness [m]
        "str_count_root": 32,            # Number of stringers at root
        "str_count_tip": 10,             # Number of stringers at tip
        "str_area": 0.00035              # Stringer cross-sectional area [m^2]
    },
    {
        "name": "Option 2 (Multi-Cell 3-Spar Baseline)",
        "cell_type": "multi_cell",
        "t_skin_root": 0.0080, "t_skin_tip": 0.0025,
        "t_spar_root": 0.0080, "t_spar_tip": 0.0035,
        "str_count_root": 28, "str_count_tip": 8,
        "str_area": 0.00030
    },
    {
        "name": "Option 3 (Multi-Cell Heavy-Skin)",
        "cell_type": "multi_cell",
        "t_skin_root": 0.0120, "t_skin_tip": 0.0040,
        "t_spar_root": 0.0100, "t_spar_tip": 0.0040,
        "str_count_root": 20, "str_count_tip": 6,
        "str_area": 0.00028
    }
]
"""

# SECTION 2: DEFAULT VALUES fallback
# ==============================================================================
def apply_defaults_and_validate():
    global b_total, c_root, c_tip, t_over_c
    global x_fs_frac, x_ms_frac, x_rs_frac
    global h_fs_factor, h_ms_factor, h_rs_factor
    global E_mat, G_mat, rho_mat, deflection_limit_pct, twist_limit_deg
    global n_pos, n_neg, M0_pos_benchmark, T0_pos_benchmark, user_options
    
    print("\n" + "="*80)
    print("      INITIALIZING WP4.2 STIFFNESS & DEFLECTION SOLVER PARAMETERS")
    print("="*80)
    
    if b_total is None:
        b_total = 46.18
        print(" -> Wing total span 'b_total' set to default baseline: 46.18 m")
    if c_root is None:
        c_root = 5.86
        print(" -> Root chord 'c_root' set to default baseline: 5.86 m")
    if c_tip is None:
        c_tip = 1.72
        print(" -> Tip chord 'c_tip' set to default baseline: 1.72 m")
    if t_over_c is None:
        t_over_c = 0.121
        print(" -> Airfoil t/c 't_over_c' set to default baseline: 0.121 (RAE 2822)")
        
    if x_fs_frac is None:
        x_fs_frac = 0.15
        print(" -> Front spar position 'x_fs_frac' set to 0.15 (15% chord)")
    if x_ms_frac is None:
        x_ms_frac = 0.40
        print(" -> Middle spar position 'x_ms_frac' set to 0.40 (40% chord)")
    if x_rs_frac is None:
        x_rs_frac = 0.65
        print(" -> Rear spar position 'x_rs_frac' set to 0.65 (65% chord)")

    if h_fs_factor is None:
        h_fs_factor = 0.85
        print(" -> Front spar height factor 'h_fs_factor' set to 0.85 (85% t_max)")
    if h_ms_factor is None:
        h_ms_factor = 0.95
        print(" -> Middle spar height factor 'h_ms_factor' set to 0.95 (95% t_max)")
    if h_rs_factor is None:
        h_rs_factor = 0.65
        print(" -> Rear spar height factor 'h_rs_factor' set to 0.65 (65% t_max)")

    if E_mat is None:
        E_mat = 73.1e9
        print(" -> Young's Modulus 'E_mat' set to default AL2024-T81: 73.1 GPa")
    if G_mat is None:
        G_mat = 28.0e9
        print(" -> Shear Modulus 'G_mat' set to default AL2024-T81: 28.0 GPa")
    if rho_mat is None:
        rho_mat = 2780.0
        
    if deflection_limit_pct is None:
        deflection_limit_pct = 0.15
    if twist_limit_deg is None:
        twist_limit_deg = 10.0

    if n_pos is None:
        n_pos = 2.5
    if n_neg is None:
        n_neg = -1.0

    if M0_pos_benchmark is None:
        M0_pos_benchmark = 5.25e6
    if T0_pos_benchmark is None:
        T0_pos_benchmark = 3.20e5

    if user_options is None:
        print(" -> 'user_options' set to default 3 benchmark preliminary options.")
        user_options = [
            {
                "name": "Option 1 (Single-Cell 2-Spar)",
                "cell_type": "single_cell",
                "t_skin_root": 0.0100, "t_skin_tip": 0.0030,
                "t_spar_root": 0.0120, "t_spar_tip": 0.0040,
                "str_count_root": 32, "str_count_tip": 10,
                "str_area": 0.00035
            },
            {
                "name": "Option 2 (Multi-Cell 3-Spar Baseline)",
                "cell_type": "multi_cell",
                "t_skin_root": 0.0080, "t_skin_tip": 0.0025,
                "t_spar_root": 0.0080, "t_spar_tip": 0.0035,
                "str_count_root": 28, "str_count_tip": 8,
                "str_area": 0.00030
            },
            {
                "name": "Option 3 (Multi-Cell Heavy-Skin)",
                "cell_type": "multi_cell",
                "t_skin_root": 0.0120, "t_skin_tip": 0.0040,
                "t_spar_root": 0.0100, "t_spar_tip": 0.0040,
                "str_count_root": 20, "str_count_tip": 6,
                "str_area": 0.00028
            }
        ]
    print("="*80 + "\n")


# ==============================================================================
# SECTION 3: imported load interpolation
# ==============================================================================
def get_load_functions(b_half, y_grid):
    # checks if array lengths match, otherwise error
    if y_wp41_input is not None:
        n_pts = len(y_wp41_input)
        load_arrays = {
            'Mx_pos_input': Mx_pos_input,
            'Mx_neg_input': Mx_neg_input,
            'T_pos_input':   T_pos_input,
            'T_neg_input':   T_neg_input,
        }
        for name, arr in load_arrays.items():
            if arr is not None and len(arr) != n_pts:
                raise ValueError(
                    f"Array length mismatch in WP4.1 inputs! 'y_wp41_input' has {n_pts} points, "
                    f"but '{name}' has {len(arr)} points. All exported arrays must have identical lengths."
                )
    #inteprolation code bvelow
    if y_wp41_input is not None and Mx_pos_input is not None:
        print(" -> Using WP4.1 exported arrays for positive load case (+n).")
        Mx_pos_func = interp1d(y_wp41_input, Mx_pos_input, kind='cubic', fill_value='extrapolate')
    else:
        M0_pos = M0_pos_benchmark * (n_pos / 2.5)
        Mx_pos_func = lambda y: M0_pos * np.maximum(0.0, (1.0 - y / b_half)) ** 2.0

    if y_wp41_input is not None and Mx_neg_input is not None:
        print(" -> Using WP4.1 exported arrays for negative load case (-n).")
        Mx_neg_func = interp1d(y_wp41_input, Mx_neg_input, kind='cubic', fill_value='extrapolate')
    else:
        M0_neg = M0_pos_benchmark * (n_neg / 2.5)
        Mx_neg_func = lambda y: M0_neg * np.maximum(0.0, (1.0 - y / b_half)) ** 2.0

    if y_wp41_input is not None and T_pos_input is not None:
        print(" -> Using WP4.1 exported arrays for positive torque (+n).")
        T_pos_func = interp1d(y_wp41_input, T_pos_input, kind='cubic', fill_value='extrapolate')
    else:
        T0_pos = T0_pos_benchmark * (n_pos / 2.5)
        T_pos_func = lambda y: T0_pos * np.maximum(0.0, (1.0 - y / b_half)) ** 1.8

    if y_wp41_input is not None and T_neg_input is not None:
        print(" -> Using WP4.1 exported arrays for negative torque (-n).")
        T_neg_func = interp1d(y_wp41_input, T_neg_input, kind='cubic', fill_value='extrapolate')
    else:
        T0_neg = T0_pos_benchmark * (n_neg / 2.5)
        T_neg_func = lambda y: T0_neg * np.maximum(0.0, (1.0 - y / b_half)) ** 1.8

    return Mx_pos_func, Mx_neg_func, T_pos_func, T_neg_func


# ==============================================================================
# SECTION 4: properties, Ixx and J for both multi cell and single cell
# ==============================================================================
class WingBoxSection:
    def __init__(self, name, cell_type, t_skin_root, t_skin_tip,
                 t_spar_root, t_spar_tip, str_count_root, str_count_tip, str_area):
        self.name = name
        self.cell_type = cell_type
        self.t_skin_root = t_skin_root
        self.t_skin_tip = t_skin_tip
        self.t_spar_root = t_spar_root
        self.t_spar_tip = t_spar_tip
        self.str_count_root = str_count_root
        self.str_count_tip = str_count_tip
        self.str_area = str_area

    def get_properties(self, y, b_half):
        c = c_root - (c_root - c_tip) * (y / b_half)
        eta = y / b_half
        
        t_skin = self.t_skin_root - (self.t_skin_root - self.t_skin_tip) * eta
        t_spar = self.t_spar_root - (self.t_spar_root - self.t_spar_tip) * eta
        # this is interpolation assumption for number of stringers look below in comment to find way to do it after stringer values are found
        n_str_float = int(interp1d([0, b_half], [self.str_count_root, self.str_count_tip], kind='linear', fill_value='extrapolate')(y))
        n_str = max(0, 2 * int(round(n_str_float / 2)))
        #preferably use method below for stringer count
        """
        FOR STRINGER AMOUNTS KNOWN 
        if eta < 0.25:
            n_str = self.str_count_root
        elif eta < 0.50:
            n_str = 24
        elif eta < 0.70:
            n_str = 18
        elif eta < 0.85:
            n_str = 12
        else:
            n_str = self.str_count_tip
    """
        x_fs = x_fs_frac * c
        x_rs = x_rs_frac * c
        box_width = x_rs - x_fs
        
        h_fs = c * t_over_c * h_fs_factor
        h_rs = c * t_over_c * h_rs_factor
        h_avg = 0.5 * (h_fs + h_rs)
        
        # --- A. BENDING MOMENT OF INERTIA (Ixx) ---
        I_skins = 2.0 * (box_width * t_skin * (0.5 * h_avg) ** 2)
        
        if self.cell_type == 'single_cell':
            I_spars = (1.0 / 12.0) * t_spar * (h_fs ** 3 + h_rs ** 3)
        elif self.cell_type == 'multi_cell':
            h_ms = c * t_over_c * h_ms_factor
            I_spars = (1.0 / 12.0) * t_spar * (h_fs ** 3 + h_ms ** 3 + h_rs ** 3)
        else:
            raise ValueError(
        f"Unsupported cell_type: {self.cell_type}"
        )
            
        I_stringers = 0.0
        if n_str > 0:
            n_top = n_str // 2
            n_bot = n_str - n_top
            z_top = 0.5 * h_avg
            z_bot = -0.5 * h_avg
            I_stringers = (n_top * self.str_area * z_top ** 2) + (n_bot * self.str_area * z_bot ** 2)
            
        Ixx = I_skins + I_spars + I_stringers
        
        # --- B. TORSIONAL CONSTANT (J) ---
        if self.cell_type == 'single_cell':
            A_enclosed = box_width * h_avg
            ds_over_t = (2.0 * box_width / t_skin) + (h_fs / t_spar) + (h_rs / t_spar)
            J = 4.0 * (A_enclosed ** 2) / ds_over_t
        elif self.cell_type == 'multi_cell':
            # Multi-Cell Formula (3 spars -> 2 cells)
            h_ms = c * t_over_c * h_ms_factor
            width_cell = (x_rs_frac - x_fs_frac) * 0.5 * c  ## assumes the cells are of same width
            h_avg1 = 0.5 * (h_fs + h_ms)
            h_avg2 = 0.5 * (h_ms + h_rs)
            
            A1 = width_cell * h_avg1
            A2 = width_cell * h_avg2
            
            a11 = (1.0 / (2.0 * A1)) * ((2.0 * width_cell / t_skin) + (h_fs / t_spar) + (h_ms / t_spar))
            a12 = -(1.0 / (2.0 * A1)) * (h_ms / t_spar)
            a21 = -(1.0 / (2.0 * A2)) * (h_ms / t_spar)
            a22 = (1.0 / (2.0 * A2)) * ((2.0 * width_cell / t_skin) + (h_rs / t_spar) + (h_ms / t_spar))
            
            M = np.array([      ## setup matrix
                [2.0 * A1, 2.0 * A2],
                [(a11 - a21), (a12 - a22)]
            ])
            rhs = np.array([1.0, 0.0]) # right side matrix T = 1 # difference between twist rate = 0
            
            q1, q2 = np.linalg.solve(M, rhs)
            rate_of_twist = a11 * q1 + a12 * q2 ## G 𝑑θ𝑑𝑦=𝑇𝐽
            J = 1.0 / rate_of_twist # because T = 1

        return Ixx, J


# ==============================================================================
# SECTION 5: DEFLECTION & TWIST integration
# ==============================================================================
def solve_option_stiffness(opt_obj, b_half, y_stations, Mx_func, T_func):
    Ixx_vals, J_vals = [], []
    for y in y_stations:
        ixx, j_val = opt_obj.get_properties(y, b_half)
        Ixx_vals.append(ixx)
        J_vals.append(j_val)
        
    Ixx_vals = np.array(Ixx_vals)
    J_vals = np.array(J_vals)
    
    Ixx_func = interp1d(y_stations, Ixx_vals, kind='cubic', fill_value='extrapolate')
    J_func = interp1d(y_stations, J_vals, kind='cubic', fill_value='extrapolate')
    
    # Pass 1: Bending Slope dv/dy
    slope_list = []
    for y in y_stations:
        if y == 0:
            slope_list.append(0.0)
        else:
            integrand_slope = lambda eta: -Mx_func(eta) / (E_mat * Ixx_func(eta))
            slope_val, _ = quad(integrand_slope, 0, y, limit=100)
            slope_list.append(slope_val)
            
    dv_dy_func = interp1d(y_stations, np.array(slope_list), kind='cubic', fill_value='extrapolate')
    
    # Pass 2: Bending Deflection v(y)
    v_list = []
    for y in y_stations:
        if y == 0:
            v_list.append(0.0)
        else:
            v_val, _ = quad(dv_dy_func, 0, y, limit=100)
            v_list.append(v_val)
            
    # Pass 3: Twist Angle theta(y)
    theta_list = []
    for y in y_stations:
        if y == 0:
            theta_list.append(0.0)
        else:
            integrand_twist = lambda eta: T_func(eta) / (G_mat * J_func(eta))
            theta_val, _ = quad(integrand_twist, 0, y, limit=100)
            theta_list.append(theta_val)
            
    theta_deg = np.degrees(np.array(theta_list))
    
    return Ixx_vals, J_vals, np.array(v_list), theta_deg


# ==============================================================================
# SECTION 6:plots & solver
# ==============================================================================
def main():
    apply_defaults_and_validate()
    
    b_half = b_total / 2.0
    v_tip_limit = deflection_limit_pct * b_total
    
    y_stations = np.linspace(0, b_half, 51)
    
    Mx_pos_func, Mx_neg_func, T_pos_func, T_neg_func = get_load_functions(b_half, y_stations)
    
    options_objs = []
    for opt in user_options:
        options_objs.append(
            WingBoxSection(
                name=opt["name"],
                cell_type=opt["cell_type"],
                t_skin_root=opt["t_skin_root"], t_skin_tip=opt["t_skin_tip"],
                t_spar_root=opt["t_spar_root"], t_spar_tip=opt["t_spar_tip"],
                str_count_root=opt["str_count_root"], str_count_tip=opt["str_count_tip"],
                str_area=opt["str_area"]
            )
        )
        
    print("solving stiffness deflection and twist distributions ")
    results = {}
    for opt_obj in options_objs:
        Ixx, J, v_pos, theta_pos = solve_option_stiffness(opt_obj, b_half, y_stations, Mx_pos_func, T_pos_func)
        _, _, v_neg, theta_neg = solve_option_stiffness(opt_obj, b_half, y_stations, Mx_neg_func, T_neg_func)
        
        results[opt_obj.name] = {
            'Ixx': Ixx, 'J': J,
            'v_pos': v_pos, 'v_neg': v_neg,
            'theta_pos': theta_pos, 'theta_neg': theta_neg,
            'v_tip_pos': v_pos[-1], 'v_tip_neg': v_neg[-1],
            'theta_tip_pos': theta_pos[-1], 'theta_tip_neg': theta_neg[-1]
        }
        
    # Console Summary Table
    print("\n" + "="*95)
    print("WORK PACKAGE 4.2 STIFFNESS COMPLIANCE SUMMARY")
    print("="*95)
    print(f"Regulatory Limits: Max Tip Deflection <= {v_tip_limit:.3f} m (15% span), Max Tip Twist <= +/- {twist_limit_deg:.1f} deg")
    print(f"{'Design Option':<35} | {'Tip v (+n) [m]':<15} | {'Tip v (-n) [m]':<15} | {'Tip Twist (+n)':<16} | {'Compliant?'}")
    print("-" * 100)

    for name, res in results.items():
        v_pos_ok = abs(res['v_tip_pos']) <= v_tip_limit
        v_neg_ok = abs(res['v_tip_neg']) <= v_tip_limit
        th_pos_ok = abs(res['theta_tip_pos']) <= twist_limit_deg
        th_neg_ok = abs(res['theta_tip_neg']) <= twist_limit_deg
        
        is_ok = v_pos_ok and v_neg_ok and th_pos_ok and th_neg_ok
        status = "YES (PASSED)" if is_ok else "NO (FAILED)"
        print(f"{name:<35} | {res['v_tip_pos']:<15.3f} | {res['v_tip_neg']:<15.3f} | {res['theta_tip_pos']:<16.2f}° | {status}")
    print("="*95 + "\n")

    # Color palette for plots (Okabe-Ito hex values)
    color_palette = ['#0072b2', '#e69f00', '#009e73']

    # --------------------------------------------------------------------------
    # PART A: INDIVIDUAL OPTION PLOTS (STRICT WP4.2 DELIVERABLES)
    # --------------------------------------------------------------------------
    for name, res in results.items():
        clean_name = "_".join(name.split()[:2])
        
        # 1. Individual Ixx(y)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(y_stations, res['Ixx'] * 1e3, color='#0072b2', linewidth=2.2)
        ax.set_title(f"Moment of Inertia Ixx(y) — {name}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
        ax.set_ylabel("Ixx [10^-3 m^4]", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRATCH_DIR, f"Ixx_distribution_{clean_name}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

        # 2. Individual J(y)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(y_stations, res['J'] * 1e3, color='#e69f00', linewidth=2.2)
        ax.set_title(f"Torsional Constant J(y) — {name}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
        ax.set_ylabel("Torsional Constant J [10^-3 m^4]", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRATCH_DIR, f"J_distribution_{clean_name}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

        # 3. Individual Bending Deflection v(y) (+n and -n load cases)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(y_stations, res['v_pos'], label=f"+{n_pos:g}g Load Case", color='#0072b2', linewidth=2.2, linestyle='-')
        ax.plot(y_stations, res['v_neg'], label=f"{n_neg:g}g Load Case", color='#d55e00', linewidth=2.0, linestyle='--')
        ax.axhline(v_tip_limit, color='#8a8a8a', linestyle=':', label=f"Max Limit (+15% b = +{v_tip_limit:.2f} m)")
        ax.axhline(-v_tip_limit, color='#8a8a8a', linestyle=':', label=f"Min Limit (-15% b = -{v_tip_limit:.2f} m)")
        ax.set_title(f"Bending Deflection v(y) — {name}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
        ax.set_ylabel("Vertical Deflection v(y) [m]", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRATCH_DIR, f"bending_deflection_{clean_name}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

        # 4. Individual Twist Angle theta(y) (+n and -n load cases)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(y_stations, res['theta_pos'], label=f"+{n_pos:g}g Load Case", color='#0072b2', linewidth=2.2, linestyle='-')
        ax.plot(y_stations, res['theta_neg'], label=f"{n_neg:g}g Load Case", color='#d55e00', linewidth=2.0, linestyle='--')
        ax.axhline(twist_limit_deg, color='#8a8a8a', linestyle=':', label=f"Max Limit (+{twist_limit_deg:.0f}°)")
        ax.axhline(-twist_limit_deg, color='#8a8a8a', linestyle=':', label=f"Min Limit (-{twist_limit_deg:.0f}°)")
        ax.set_title(f"Wing Twist Distribution theta(y) — {name}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
        ax.set_ylabel("Twist Angle theta [deg]", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRATCH_DIR, f"twist_distribution_{clean_name}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --------------------------------------------------------------------------
    # PART B: COMBINED COMPARISON PLOTS (ALL OPTIONS ON SINGLE AXES)
    # --------------------------------------------------------------------------
    # Combined Ixx(y)
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (name, res) in enumerate(results.items()):
        ax.plot(y_stations, res['Ixx'] * 1e3, label=name, color=color_palette[idx % 3], linewidth=2.2)
    ax.set_title("Spanwise Moment of Inertia Ixx(y) Comparison", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
    ax.set_ylabel("Ixx [10^-3 m^4]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_DIR, "Ixx_distribution_all_options.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Combined J(y)
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (name, res) in enumerate(results.items()):
        ax.plot(y_stations, res['J'] * 1e3, label=name, color=color_palette[idx % 3], linewidth=2.2)
    ax.set_title("Spanwise Torsional Constant J(y) Comparison", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
    ax.set_ylabel("Torsional Constant J [10^-3 m^4]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_DIR, "J_distribution_all_options.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Combined Deflection v(y)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (name, res) in enumerate(results.items()):
        c_val = color_palette[idx % 3]
        ax.plot(y_stations, res['v_pos'], label=f"{name} (+n)", color=c_val, linewidth=2.2)
        ax.plot(y_stations, res['v_neg'], label=f"{name} (-n)", color=c_val, linewidth=1.8, linestyle="--")
    ax.axhline(v_tip_limit, color='#8a8a8a', linestyle=':', label=f"Max Limit (+15% b = {v_tip_limit:.2f} m)")
    ax.axhline(-v_tip_limit, color='#8a8a8a', linestyle=':', label=f"Min Limit (-15% b = {-v_tip_limit:.2f} m)")
    ax.set_title("Bending Deflection v(y) Profiles vs. 15% Wingspan Limit", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
    ax.set_ylabel("Vertical Deflection v(y) [m]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_DIR, "bending_deflection_all_options.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Combined Twist theta(y)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, (name, res) in enumerate(results.items()):
        c_val = color_palette[idx % 3]
        ax.plot(y_stations, res['theta_pos'], label=f"{name} (+n)", color=c_val, linewidth=2.2)
        ax.plot(y_stations, res['theta_neg'], label=f"{name} (-n)", color=c_val, linewidth=1.8, linestyle="--")
    ax.axhline(twist_limit_deg, color='#8a8a8a', linestyle=':', label=f"Max Limit (+{twist_limit_deg:.0f} deg)")
    ax.axhline(-twist_limit_deg, color='#8a8a8a', linestyle=':', label=f"Min Limit (-{twist_limit_deg:.0f} deg)")
    ax.set_title("Wing Twist Distribution theta(y) vs. +/-10 deg Limit", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Spanwise Position y [m]", fontsize=10)
    ax.set_ylabel("Twist Angle theta [deg]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRATCH_DIR, "twist_distribution_all_options.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print("All individual and combined plots successfully generated in scratch.")


if __name__ == '__main__':
    main()
