"""
WP4.1 - Step 1: Preliminary Internal Loads Script Updated with Final WP3 Weights & Geometry.

Updates applied relative to code1.py (which used WP1/WP2 estimates and Table A.1 Class I fractions),
now verified directly against WP1.pdf, WP2.pdf, and WP3.pdf (Table 5.1, Sections 2.1.4 & 4.2):
- Aircraft weight configurations updated to final converged WP3 values:
    * W1 (OEW)             = 46,261.0 kg (fuel = 0 kg)          [WP3 Table 5.1]
    * W2 (MTOW)            = 88,515.0 kg (OEW + MPW + fuel)     [WP3 Table 5.1]
    * MPW (Max Payload)    = 18,960.0 kg                        [WP1 REQ-PAYLOAD-01, WP3 Table 5.1]
    * W3 (OEW + MPW, MZFW) = 65,221.0 kg (fuel = 0 kg)          [Fixes Excel W3=69,555 kg OEW+fuel swap]
    * Wing fuel in W2      = 23,294.0 kg (MTOW - MZFW)          [Both wings]
- Component masses updated from WP3 Table A.1 Class I fractions to final WP3 Class II values:
    * Wing structure (both): 11.43% * 88,515 = 10,117.3 kg      [WP3 Table 5.1 Class II breakdown]
    * Engine + nacelle + pylon (each): 1.3 * 3,221 = 4,187.3 kg [WP3 Sec. 4.2.6 Eq. 4.40, PW2037]
- Propulsion & geometry updated to final converged WP3 values:
    * Max take-off thrust per engine: 170.8 kN (341.6 kN total) [WP3 Table 2.4 & Table 5.1, PW2037]
    * Spanwise engine station: 0.35 * b/2 = 8.0815 m            [WP3 Sec. 2.1.4: 35% semi-span]
    * Wing planform: b = 46.18 m, S = 177.70 m^2, c_r = 5.95 m, c_t = 1.74 m, le_sw = 32.66 deg
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def _interp(y_stations, values):
    return interp1d(y_stations, values, kind="cubic", fill_value="extrapolate")


G = 9.80665  # [m/s^2]
_trapz = np.trapezoid

# Final Converged WP3 Aircraft Weight Configurations (WP3.pdf Table 5.1, pp. 33-34)
OEW = 46261.0              # Operating Empty Weight [kg] (W1)
MTOW = 88515.0             # Maximum Take-Off Weight [kg] (W2: OEW + MPW + fuel)
MPW = 18960.0              # Maximum Payload Weight [kg] (REQ-PAYLOAD-01)
MZFW = OEW + MPW           # Maximum Zero-Fuel Weight [kg] (W3: OEW + MPW = 65,221.0 kg)
M_FUEL_MTOW = MTOW - MZFW  # Wing fuel mass at MTOW (W2) = 23,294.0 [kg]


# ------------------------------------------------------------------------------ 1 DATA CLASSES
@dataclass
class Wing:
    """Final converged WP3 trapezoidal wing geometry (WP3.pdf Table 5.1)."""
    b: float = 46.18       # wingspan [m] (XFLR5 = 46.181 m)
    S: float = 177.70      # wing reference area, flaps retracted [m^2] (XFLR5 = 177.564 m^2)
    c_r: float = 5.95      # root chord [m] (XFLR5 = 5.950 m)
    c_t: float = 1.74      # tip chord [m] (XFLR5 = 1.740 m)
    le_sw: float = 32.66   # leading-edge sweep angle [deg]

    def chord(self, y):
        # straight taper: c_r at the root (y = 0), c_t at the tip (y = b/2)
        return self.c_r + (self.c_t - self.c_r) * np.asarray(y) / (self.b / 2)


@dataclass
class Masses:
    """Final WP3 Class II component masses and PW2037 propulsion parameters (WP3.pdf)."""
    m_wing_both: float = 0.1143 * MTOW           # Class II wing mass, BOTH wings [kg] (= 10,117.26 kg; Table 5.1)
    m_fuel_both: float = M_FUEL_MTOW             # wing fuel at MTOW, BOTH wings [kg] (= 23,294.0 kg)
    fuel_y_end_frac: float = 0.85                # fuel tank runs from root to 85% half-span (ASSUMPTION)
    m_engine_each: float = 1.3 * 3221.0          # PW2037 installed (dry + nacelle/pylon) [kg] (= 4,187.3 kg; Eq. 4.40)
    y_engine: float = 0.35 * (46.18 / 2.0)       # spanwise engine station = 35% semi-span [m] (= 8.0815 m; Sec. 2.1.4)
    engine_arm_fwd: float = 1.5                  # engine CG ahead of shear centre [m] (ASSUMPTION)
    thrust_each: float = 170.8e3                 # PW2037 max take-off thrust per engine [N] (Table 2.4 & Table 5.1)
    engine_below_axis: float = 1.5               # thrust line below shear centre [m] (ASSUMPTION)


@dataclass
class LoadCase:
    """One operating point from the V-n manoeuvre diagram."""
    name: str              # label for reporting/plotting
    n: float               # limit load factor [-]
    mass: float            # total aircraft mass [kg]
    q: float               # dynamic pressure 0.5 * rho_0 * V_EAS^2 [Pa]
    fuel_mass: float = M_FUEL_MTOW  # fuel carried in both wings for this configuration [kg]


# ------------------------------------------------------------------------------ 2 XFLR5 LOADER
def read_xflr_csv(path):
    """Reads one XFLR5 csv export. Returns (Main Wing table, total CL from header).
    Columns: 0 y-span, 1 Chord, 2 Ai, 3 Cl, 4 PCd, 5 ICd, 6 CmGeom, 7 CmAirf@chord/4, ..."""
    with open(path, encoding="utf-8", errors="ignore") as fp:
        lines = fp.read().splitlines()
    CL = None
    for ln in lines:
        if ln.replace(" ", "").startswith("CL="):
            CL = float(ln.split(",")[1])
            break
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "Main Wing")
    rows = []
    for ln in lines[start + 2:]:
        try:
            rows.append([float(v) for v in ln.split(",")])
        except ValueError:
            break
    return np.array(rows), CL


class XFLRData:
    """Two XFLR5 inviscid 3D-panel runs (alpha = 0 and 10 deg) turned into functions of y."""

    def __init__(self, path0, path10):
        t0, self.CL0_file = read_xflr_csv(path0)
        t10, self.CL10_file = read_xflr_csv(path10)
        self.f = {}
        for tag, t in (("0", t0), ("10", t10)):
            y = t[:, 0]                                # whole span (-b/2..b/2), so the root is interpolated
            self.f["Ai" + tag] = _interp(y, t[:, 2])   # induced angle of attack [deg]
            self.f["Cl" + tag] = _interp(y, t[:, 3])   # local lift coefficient
            self.f["ICd" + tag] = _interp(y, t[:, 5])  # induced drag coefficient
            self.f["Cm" + tag] = _interp(y, t[:, 7])   # pitching moment coefficient about c/4

        # Geometry that XFLR5 actually used (recovered from the Chord column)
        yh, ch = t0[t0[:, 0] > 0, 0], t0[t0[:, 0] > 0, 1]
        slope, self.c_r = np.polyfit(yh, ch, 1)
        edge = 0.0
        for yk in yh:
            edge = 2 * yk - edge
        self.half_span = edge
        self.c_t = self.c_r + slope * edge
        self.S = (self.c_r + self.c_t) * edge

    def wing(self, le_sw=32.66):
        return Wing(b=2 * self.half_span, S=self.S, c_r=self.c_r, c_t=self.c_t, le_sw=le_sw)

    def at(self, y, k):
        # k = 0 -> alpha = 0 deg run, k = 1 -> alpha = 10 deg run
        mix = lambda a: self.f[a + "0"](y) + k * (self.f[a + "10"](y) - self.f[a + "0"](y))
        Cl = mix("Cl")
        Ai = mix("Ai")
        ICd = -Cl * np.sin(np.radians(Ai))  # physically quadratic & >= 0 for negative alpha too
        Cm = mix("Cm")
        return Cl, ICd, Cm


# ------------------------------------------------------------------------------ 3 HELPERS
def from_tip(y, f):
    """F(y) = integral of f from y to the free tip b/2."""
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(y)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def points_outboard(y, points):
    """Point load (y_p, value) acts only on stations inboard of y_p (y < y_p)."""
    return sum((np.where(y < yp, v, 0.0) for yp, v in points), np.zeros_like(y))


# ------------------------------------------------------------------------------ 4 MAIN CALCULATION
def internal_loads(case, wing, xflr, mass, x_sc_frac=0.40, npts=2001):
    y = np.linspace(0.0, wing.b / 2, npts)
    c = wing.chord(y)
    sweep = np.radians(wing.le_sw)
    n = case.n

    # 4a. Aerodynamic load
    CL_d = n * case.mass * G / (case.q * wing.S)
    CL0 = 2 * _trapz(xflr.f["Cl0"](y) * c, y) / wing.S
    CL10 = 2 * _trapz(xflr.f["Cl10"](y) * c, y) / wing.S
    k = (CL_d - CL0) / (CL10 - CL0)
    alpha = np.radians(10.0 * k)
    Cl, ICd, Cm = xflr.at(y, k)
    L = Cl * case.q * c
    D = ICd * case.q * c
    Mp = Cm * case.q * c ** 2
    N = np.cos(alpha) * L + np.sin(alpha) * D
    t_aero = N * (x_sc_frac * c - 0.25 * c) + Mp

    # 4b. Inertial load: wing structure ~ c, wing fuel ~ c^2 (inside tank), times -n*g
    mw = (mass.m_wing_both / 2) * c / _trapz(c, y)
    if case.fuel_mass > 0.0:
        mf_shape = np.where(y <= mass.fuel_y_end_frac * wing.b / 2, c ** 2, 0.0)
        mf = (case.fuel_mass / 2) * mf_shape / _trapz(mf_shape, y)
    else:
        mf = np.zeros_like(y)
    w_inertia = -n * G * (mw + mf)

    # 4c. Engine point loads (PW2037 at y_engine = 8.0815 m)
    P_eng = -n * G * mass.m_engine_each
    T_eng = P_eng * mass.engine_arm_fwd
    T_eng += mass.thrust_each * np.cos(sweep) * mass.engine_below_axis
    axial = np.where(y < mass.y_engine, -mass.thrust_each * np.sin(sweep), 0.0)  # compression negative

    # 4d. Integrate from the free wing tip (Appendix C/D sign conventions)
    S_cont = from_tip(y, N + w_inertia)
    S = S_cont + points_outboard(y, [(mass.y_engine, P_eng)])
    M = -from_tip(y, S_cont) - np.where(y < mass.y_engine, P_eng * (mass.y_engine - y), 0.0)
    T = from_tip(y, t_aero) + points_outboard(y, [(mass.y_engine, T_eng)])

    checks = {
        "CL_d": CL_d,
        "alpha_d [deg]": np.degrees(alpha),
        "CL0  integrated / file": CL0 / xflr.CL0_file,
        "CL10 integrated / file": CL10 / xflr.CL10_file,
        "lift (both wings) / (n W)": 2 * _trapz(L, y) / (n * case.mass * G),
        "S(root) [kN]": S[0] / 1e3,
        "M(root) [kNm]": M[0] / 1e3,
        "M(root) direct sum [kNm]": -(_trapz((N + w_inertia) * y, y) + P_eng * mass.y_engine) / 1e3,
        "T(root) [kNm]": T[0] / 1e3,
        "A(root) [kN]": axial[0] / 1e3,
    }
    return dict(y=y, S=S, M=M, T=T, A=axial, N=N, w_inertia=w_inertia, checks=checks)


# ------------------------------------------------------------------------------ 5 PLOT
def plot_diagrams(res, title, y_engine):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7.5, 9))
    data = [
        (res["S"] / 1e3, "S [kN]"),
        (res["M"] / 1e3, "M [kNm]"),
        (res["T"] / 1e3, "T [kNm]"),
        (res["A"] / 1e3, "Axial force A [kN]"),
    ]
    for a, (v, lab) in zip(ax, data):
        a.plot(res["y"], v)
        a.set_ylabel(lab)
        a.grid(True)
        a.axvline(y_engine, ls="--", c="gray", lw=0.8)
    ax[0].set_title(title)
    ax[-1].set_xlabel("y [m]")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------ 6 RUN
if __name__ == "__main__":
    xflr = XFLRData("MainWing_a=0.00_v=10.00ms.csv", "MainWing_a=10.00_v=10.00ms.csv")
    wp3 = Wing()
    wing = xflr.wing()
    print("geometry        WP3 values   XFLR5 file")
    for name in ("b", "S", "c_r", "c_t"):
        print(f"  {name:4s} {getattr(wp3, name):12.3f} {getattr(wing, name):12.3f}")

    masses = Masses()
    print(f"\nWP3 Final Masses: MTOW={MTOW:.1f} kg, OEW={OEW:.1f} kg, MPW={MPW:.1f} kg, "
          f"MZFW={MZFW:.1f} kg, Fuel(W2)={M_FUEL_MTOW:.1f} kg, "
          f"Wing(both)={masses.m_wing_both:.1f} kg, Engine(each)={masses.m_engine_each:.1f} kg, "
          f"Thrust(each)={masses.thrust_each/1e3:.1f} kN, y_eng={masses.y_engine:.3f} m")

    # Placeholder V_EAS = 160 m/s until WP4.3 load cases are incorporated in Step 2
    q_placeholder = 0.5 * 1.225 * 160.0 ** 2
    cases = [
        LoadCase("positive, n = +2.5 (W2 MTOW)", 2.5, MTOW, q_placeholder, M_FUEL_MTOW),
        LoadCase("negative, n = -1.0 (W2 MTOW)", -1.0, MTOW, q_placeholder, M_FUEL_MTOW),
    ]
    for cs in cases:
        res = internal_loads(cs, wing, xflr, masses)
        print(f"\n{cs.name}")
        for key, v in res["checks"].items():
            print(f"  {key:28s} {v:10.3f}")
        fig = plot_diagrams(res, f"WP4.1 (WP3 Class II Weights) - {cs.name}", masses.y_engine)
        fig.savefig(f"WP4_1_step1_{'pos' if cs.n > 0 else 'neg'}.png", dpi=150)
        plt.close(fig)
