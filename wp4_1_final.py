"""
WP4.1 - Final Internal Load Diagrams Incorporating Final WP3 Class II Values & WP4.3 Load Cases.

Implements the complete Task 4.1 methodology for Group A05 (AE2111-I Aircraft Design):
  1. Uses final converged WP3 Class II aircraft weights, PW2037 propulsion specs, and wing geometry
     verified directly from WP1.pdf, WP2.pdf, and WP3.pdf (Table 5.1, Sec. 2.1.4, Sec. 4.2):
       - W1 (OEW)                = 46,261.0 kg (fuel = 0 kg)
       - W2 (MTOW)               = 88,515.0 kg (OEW + MPW + fuel -> wing fuel = 23,294.0 kg)
       - W3 (MZFW = OEW + MPW)   = 65,221.0 kg (MPW = 18,960.0 kg per REQ-PAYLOAD-01, fuel = 0 kg)
       - Wing structure (both)   = 10,117.3 kg (11.43% of MTOW, Class II Raymer)
       - Engine installed (each) = 4,187.3 kg  (1.3 * 3,221 kg dry PW2037, Eq. 4.40)
       - Max thrust (each)       = 170.8 kN    (PW2037, Table 2.4 & Table 5.1)
       - Engine spanwise station = 8.0815 m    (35% of final semi-span 46.18 / 2, Sec. 2.1.4)
  2. Incorporates all 30 clean-wing symmetric manoeuvre load cases (LC-1 to LC-30)
     from the WP4.3 (V_EAS - n) diagrams across W1/W2/W3 and Sea Level / Cruise,
     with CS-25 and WP3 corrections applied to the Excel sheet bugs:
       - W3 uses OEW + MPW = 65,221 kg (corrects Excel 69,555 kg which was OEW + fuel).
       - Sea-level EAS equals sea-level TAS (removes spurious *sqrt(rho_cr/rho_0) factor).
       - Sea-level V_D = V_C / 0.8 = 192.42 m/s per CS-25.335(b) (not Mach-limited at SL);
         cruise V_D = 162.99 m/s from M_D = M_C + 0.05 = 0.90 per CS-25.335(b)(2).
       - Point F (LC-13..LC-18, n = -1.0) uses V_C = 153.935 m/s (not flap speed V_F).
       - Point H (LC-25..LC-30, n = -1.0) uses the correctly matched V_S1,EAS per weight.
       - Flaps-down load cases are excluded per WP4.1 instructions (Assumption A17).
  3. Models per-load-case wing fuel mass (0 kg for W1 and W3; 23,294 kg for W2),
     quadratic induced drag via linear induced angle of attack Ai(y), Appendix C/D
     sign conventions, variable shear-centre support for WP4.2, and callable output
     functions S_fn(y), M_fn(y), T_fn(y), A_fn(y).
  4. Evaluates and ranks all load cases, verifies integration against scipy.integrate.quad,
     and saves publication-quality figures for the WP4 report.
"""

from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad

# ------------------------------------------------------------------------------ 0 CONSTANTS
G = 9.80665        # standard gravitational acceleration [m/s^2]
RHO_SL = 1.225     # ISA sea-level air density [kg/m^3]
RHO_CR = 0.4416    # cruise altitude air density [kg/m^3] (WP1/WP3 Excel, h ~ 9.45-9.50 km)
T_SL = 288.15      # ISA sea-level temperature [K]
T_CR = 226.43      # cruise altitude temperature [K] (WP1 Table B.1 / V-n Excel)
GAMMA_AIR = 1.4    # ratio of specific heats [-]
R_AIR = 287.0      # specific gas constant of dry air [J/(kg*K)]

_trapz = np.trapezoid

# Final Converged WP3 Aircraft Weight Configurations (WP3.pdf Table 5.1, pp. 33-34)
OEW = 46261.0              # W1: Operating Empty Weight [kg]
MTOW = 88515.0             # W2: Maximum Take-Off Weight = OEW + MPW + fuel [kg]
MPW = 18960.0              # Maximum Payload Weight [kg] (REQ-PAYLOAD-01)
MZFW = OEW + MPW           # W3: Maximum Zero-Fuel Weight = OEW + MPW = 65,221.0 [kg]
M_FUEL_W2 = MTOW - MZFW    # Wing fuel mass in W2 configuration = 23,294.0 [kg]


def _interp_cubic(y_stations, values):
    return interp1d(y_stations, values, kind="cubic", fill_value="extrapolate")


# ------------------------------------------------------------------------------ 1 DATA CLASSES
@dataclass
class Wing:
    """Final converged WP3 equivalent trapezoidal wing geometry (WP3.pdf Table 5.1)."""
    b: float = 46.18       # wingspan [m]
    S: float = 177.70      # flaps-retracted reference wing area [m^2]
    c_r: float = 5.95      # root chord [m]
    c_t: float = 1.74      # tip chord [m]
    le_sw: float = 32.66   # leading-edge sweep angle [deg]

    def chord(self, y):
        return self.c_r + (self.c_t - self.c_r) * np.asarray(y) / (self.b / 2)


@dataclass
class Masses:
    """Final WP3 Class II structural/propulsive masses and PW2037 geometry (WP3.pdf)."""
    m_wing_both: float = 0.1143 * MTOW           # Class II wing structure, BOTH wings [kg] (= 10,117.26 kg; Table 5.1)
    fuel_y_end_frac: float = 0.85                # wing tank spans y in [0, 0.85*b/2] (ASSUMPTION)
    m_engine_each: float = 1.3 * 3221.0          # PW2037 installed (dry + nacelle/pylon) [kg] (= 4,187.3 kg; Eq. 4.40)
    y_engine: float = 0.35 * (46.18 / 2.0)       # spanwise engine location = 35% semi-span [m] (= 8.0815 m; Sec. 2.1.4)
    engine_arm_fwd: float = 1.5                  # engine CG ahead of shear centre [m] (ASSUMPTION)
    thrust_each: float = 170.8e3                 # PW2037 max take-off thrust per engine [N] (Table 2.4 & Table 5.1)
    engine_below_axis: float = 1.5               # thrust line below shear centre [m] (ASSUMPTION)


@dataclass
class FlightEnvelope:
    """Parameters used to construct the WP4.3 CS-25 (V_EAS - n) manoeuvre envelopes."""
    S_retracted: float = 177.70     # clean-wing reference area [m^2] (WP3 Table 5.1)
    CL_max_clean: float = 1.544     # clean-wing maximum lift coefficient [-] (WP3 Table 5.1)
    M_cruise: float = 0.85          # design cruise Mach number [-] (WP3 Table 5.1)
    n_max: float = 2.5              # CS-25.337(b) positive limit load factor [-]
    n_min: float = -1.0             # CS-25.337(c) negative limit load factor [-]

    @property
    def a_sl(self) -> float:
        return float(np.sqrt(GAMMA_AIR * R_AIR * T_SL))

    @property
    def a_cr(self) -> float:
        return float(np.sqrt(GAMMA_AIR * R_AIR * T_CR))

    @property
    def V_C_eas(self) -> float:
        """Design cruising speed V_C in EAS [m/s] (altitude-independent per Appendix E.1)."""
        V_C_tas_cr = self.M_cruise * self.a_cr
        return float(V_C_tas_cr * np.sqrt(RHO_CR / RHO_SL))

    def V_D_eas(self, altitude: str, use_excel_sl_value: bool = False) -> float:
        """Design dive speed V_D in EAS [m/s] per CS-25.335(b).
        - Sea level: V_C/0.8 (not Mach-limited, M_D = 0.566).
        - Cruise:    M_D = M_C + 0.05 = 0.90 (compressibility-limited)."""
        if altitude == "Sea level":
            if use_excel_sl_value:
                return 173.6519609  # raw Excel L6 value (kept only for optional comparison)
            return float(self.V_C_eas / 0.8)
        elif altitude == "Cruise":
            M_D_cr = self.M_cruise + 0.05
            V_D_tas_cr = M_D_cr * self.a_cr
            return float(V_D_tas_cr * np.sqrt(RHO_CR / RHO_SL))
        raise ValueError(f"Unknown altitude: {altitude}")

    def V_S1_eas(self, mass: float) -> float:
        """Clean-wing 1g stall speed in EAS [m/s] (altitude-independent under Assumption A16)."""
        return float(np.sqrt(2.0 * mass * G / (RHO_SL * self.S_retracted * self.CL_max_clean)))

    def V_A_eas(self, mass: float) -> float:
        """Manoeuvring speed V_A = V_S1 * sqrt(n_max) in EAS [m/s] per CS-25.335(c)(1)."""
        return float(self.V_S1_eas(mass) * np.sqrt(self.n_max))


@dataclass
class LoadCase:
    """Single symmetric manoeuvre load case from WP4.3."""
    case_id: str           # e.g. 'LC-8'
    point: str             # V-n diagram corner: 'A', 'C', 'D', 'F', 'H'
    weight_label: str      # 'W1: OEW', 'W2: MTOW', 'W3: OEW+MPW'
    altitude: str          # 'Sea level' or 'Cruise'
    V_eas: float           # equivalent airspeed [m/s]
    n: float               # limit load factor [-]
    mass: float            # total aircraft mass [kg]
    fuel_mass: float       # wing fuel mass for this weight configuration [kg]
    thrust_scale: float = 1.0  # 1.0 = full thrust, 0.0 = idle thrust

    @property
    def q(self) -> float:
        """Dynamic pressure q = 0.5 * rho_0 * V_EAS^2 [Pa]."""
        return 0.5 * RHO_SL * self.V_eas ** 2

    @property
    def label(self) -> str:
        return (f"{self.case_id}: {self.weight_label}, Pt {self.point} "
                f"({self.altitude}, V_EAS={self.V_eas:.1f} m/s, n={self.n:+.1f})")


def build_wp43_load_cases(env: FlightEnvelope, use_excel_sl_vd: bool = False) -> List[LoadCase]:
    """Constructs all 30 clean-wing WP4.3 load cases (LC-1 to LC-30), preserving the
    row numbering of 'Wp3 V-n-diagrams.xlsx' (Sheet 'Load cases') with CS-25 & WP3 corrections."""
    weight_configs = [
        ("W1: OEW", OEW, 0.0),
        ("W2: MTOW", MTOW, M_FUEL_W2),
        ("W3: OEW+MPW", MZFW, 0.0),
    ]
    altitudes = ["Sea level", "Cruise"]
    cases: List[LoadCase] = []
    idx = 1

    # LC-1 .. LC-6 : Point A (V_A, n = +2.5)
    for alt in altitudes:
        for w_lab, mass, fuel in weight_configs:
            cases.append(LoadCase(f"LC-{idx}", "A", w_lab, alt, env.V_A_eas(mass), env.n_max, mass, fuel))
            idx += 1

    # LC-7 .. LC-12 : Point D (V_D, n = +2.5)
    for alt in altitudes:
        V_D = env.V_D_eas(alt, use_excel_sl_value=use_excel_sl_vd)
        for w_lab, mass, fuel in weight_configs:
            cases.append(LoadCase(f"LC-{idx}", "D", w_lab, alt, V_D, env.n_max, mass, fuel))
            idx += 1

    # LC-13 .. LC-18 : Point F (V_C, n = -1.0)  [corrected from Excel V_F mix-up]
    for alt in altitudes:
        for w_lab, mass, fuel in weight_configs:
            cases.append(LoadCase(f"LC-{idx}", "F", w_lab, alt, env.V_C_eas, env.n_min, mass, fuel))
            idx += 1

    # LC-19 .. LC-24 : Point C (V_C, n = +2.5)
    for alt in altitudes:
        for w_lab, mass, fuel in weight_configs:
            cases.append(LoadCase(f"LC-{idx}", "C", w_lab, alt, env.V_C_eas, env.n_max, mass, fuel))
            idx += 1

    # LC-25 .. LC-30 : Point H (V_S1, n = -1.0)  [corrected from Excel scrambled V_H]
    for alt in altitudes:
        for w_lab, mass, fuel in weight_configs:
            cases.append(LoadCase(f"LC-{idx}", "H", w_lab, alt, env.V_S1_eas(mass), env.n_min, mass, fuel))
            idx += 1

    return cases


# ------------------------------------------------------------------------------ 2 XFLR5 LOADER
def read_xflr_csv(path: str):
    """Reads one XFLR5 csv export. Returns (Main Wing table, total CL from header).
    Columns: 0 y-span, 1 Chord, 2 Ai [deg], 3 Cl, 4 PCd, 5 ICd, 6 CmGeom, 7 CmAirf@c/4, ..."""
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
    """Two XFLR5 inviscid 3D-panel runs (alpha = 0 and 10 deg) interpolated over span y."""

    def __init__(self, path0: str, path10: str):
        t0, self.CL0_file = read_xflr_csv(path0)
        t10, self.CL10_file = read_xflr_csv(path10)
        self.f = {}
        for tag, t in (("0", t0), ("10", t10)):
            y = t[:, 0]  # full span (-b/2 .. +b/2) -> root is smoothly interpolated
            self.f["Ai" + tag] = _interp_cubic(y, t[:, 2])   # induced angle of attack [deg]
            self.f["Cl" + tag] = _interp_cubic(y, t[:, 3])   # sectional lift coefficient [-]
            self.f["ICd" + tag] = _interp_cubic(y, t[:, 5])  # induced drag coefficient [-]
            self.f["Cm" + tag] = _interp_cubic(y, t[:, 7])   # quarter-chord pitching moment coeff [-]

        # Recover exact trapezoidal geometry simulated in XFLR5 from strip centres
        yh, ch = t0[t0[:, 0] > 0, 0], t0[t0[:, 0] > 0, 1]
        slope, self.c_r = np.polyfit(yh, ch, 1)
        edge = 0.0
        for yk in yh:
            edge = 2.0 * yk - edge
        self.half_span = edge
        self.c_t = self.c_r + slope * edge
        self.S = (self.c_r + self.c_t) * edge

    def wing(self, le_sw: float = 32.66) -> Wing:
        return Wing(b=2.0 * self.half_span, S=self.S, c_r=self.c_r, c_t=self.c_t, le_sw=le_sw)

    def at(self, y, k: float):
        """Returns (Cl, ICd, Cm) at span stations y for scaling factor k = (CL_d - CL0)/(CL10 - CL0).
        Cl, Ai, and Cm scale linearly with alpha/CL; induced drag is computed quadratically
        from Cl * (-sin(Ai)) so that ICd >= 0 holds even at negative load factors."""
        mix = lambda a: self.f[a + "0"](y) + k * (self.f[a + "10"](y) - self.f[a + "0"](y))
        Cl = mix("Cl")
        Ai = mix("Ai")
        ICd = -Cl * np.sin(np.radians(Ai))
        Cm = mix("Cm")
        return Cl, ICd, Cm


# ------------------------------------------------------------------------------ 3 HELPERS
def from_tip(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Cumulative integral from y to the free tip b/2 using the composite trapezoidal rule."""
    seg = 0.5 * (f[1:] + f[:-1]) * np.diff(y)
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def points_outboard(y: np.ndarray, points) -> np.ndarray:
    """Point loads (y_p, value) contribute via Heaviside step (1 - u_{y_p}(y)) for y < y_p."""
    return sum((np.where(y < yp, v, 0.0) for yp, v in points), np.zeros_like(y))


# ------------------------------------------------------------------------------ 4 MAIN CALCULATION
def internal_loads(
    case: LoadCase,
    wing: Wing,
    xflr: XFLRData,
    mass: Masses,
    x_sc_frac: Union[float, Callable[[np.ndarray], np.ndarray]] = 0.40,
    safety_factor: float = 1.0,
    npts: int = 2001,
    verify_quad: bool = False,
):
    """Computes spanwise internal load distributions S(y), M(y), T(y), A(y) for one LoadCase.

    Sign conventions (consistent with WP4 Reader Appendices C & D):
      - y in [0, b/2] from wing root to tip; z positive upward.
      - S(y) > 0 for upward shear carried toward root: dS/dy = -w(y).
      - M(y) < 0 for upward wing bending (top skin in compression): dM/dy = +S(y),
        so Euler-Bernoulli curvature v''(y) = -M(y) / (E * I_xx(y)) > 0 (upward deflection).
      - T(y) > 0 for nose-up torque: dT/dy = -t(y), matching d(theta)/dy = +T(y) / (G * J(y)).
      - A(y) < 0 for axial compression (tension positive).
      - safety_factor = 1.0 gives limit loads (WP4.2 deflections); 1.5 gives ultimate loads (WP5).
    """
    y = np.linspace(0.0, wing.b / 2, npts)
    c = wing.chord(y)
    sweep = np.radians(wing.le_sw)
    n = case.n
    q = case.q
    x_sc = x_sc_frac(y) if callable(x_sc_frac) else np.full_like(y, float(x_sc_frac))

    # 4a. Aerodynamic loads (XFLR5 linear scaling in CL + aerodynamic-to-body transformation)
    CL_d = n * case.mass * G / (q * wing.S)
    CL0 = 2.0 * _trapz(xflr.f["Cl0"](y) * c, y) / wing.S
    CL10 = 2.0 * _trapz(xflr.f["Cl10"](y) * c, y) / wing.S
    k = (CL_d - CL0) / (CL10 - CL0)
    alpha = np.radians(10.0 * k)

    Cl, ICd, Cm = xflr.at(y, k)
    L = Cl * q * c                     # lift per unit span [N/m]
    D = ICd * q * c                    # induced drag per unit span [N/m]
    Mp = Cm * q * c ** 2               # pitching moment about c/4 per unit span [N*m/m]
    N = np.cos(alpha) * L + np.sin(alpha) * D  # normal force per unit span [N/m]
    t_aero = N * (x_sc - 0.25) * c + Mp        # distributed aerodynamic torque about SC [N*m/m]

    # 4b. Distributed inertial loads: structure ~ c(y), wing fuel ~ c(y)^2 inside tank
    mw = (mass.m_wing_both / 2.0) * c / _trapz(c, y)
    if case.fuel_mass > 0.0:
        mf_shape = np.where(y <= mass.fuel_y_end_frac * wing.b / 2, c ** 2, 0.0)
        mf = (case.fuel_mass / 2.0) * mf_shape / _trapz(mf_shape, y)
    else:
        mf = np.zeros_like(y)
    w_inertia = -n * G * (mw + mf)     # inertial load per unit span [N/m]

    # 4c. Engine concentrated loads (Appendix A.3: thrust rotated inboard by sweep angle)
    thrust = mass.thrust_each * case.thrust_scale
    P_eng = -n * G * mass.m_engine_each                                   # vertical inertial load [N]
    T_eng = P_eng * mass.engine_arm_fwd + thrust * np.cos(sweep) * mass.engine_below_axis  # torque [N*m]
    A_eng = -thrust * np.sin(sweep)                                       # compressive axial force [N]

    # 4d. Integrate from the free wing tip (y = b/2) and apply safety factor
    w_tot = N + w_inertia
    S_cont = from_tip(y, w_tot)
    T_cont = from_tip(y, t_aero)

    S = safety_factor * (S_cont + points_outboard(y, [(mass.y_engine, P_eng)]))
    M = -safety_factor * (from_tip(y, S_cont) + np.where(y < mass.y_engine, P_eng * (mass.y_engine - y), 0.0))
    T = safety_factor * (T_cont + points_outboard(y, [(mass.y_engine, T_eng)]))
    A = safety_factor * points_outboard(y, [(mass.y_engine, A_eng)])

    # 4e. Callable functions for seamless WP4.2 integration (handles step at y_engine cleanly)
    S_cont_fn = interp1d(y, safety_factor * S_cont, kind="linear", fill_value="extrapolate")
    T_cont_fn = interp1d(y, safety_factor * T_cont, kind="linear", fill_value="extrapolate")
    M_fn = interp1d(y, M, kind="linear", fill_value="extrapolate")

    def S_fn(y_eval):
        ye = np.asarray(y_eval)
        return S_cont_fn(ye) + np.where(ye < mass.y_engine, safety_factor * P_eng, 0.0)

    def T_fn(y_eval):
        ye = np.asarray(y_eval)
        return T_cont_fn(ye) + np.where(ye < mass.y_engine, safety_factor * T_eng, 0.0)

    def A_fn(y_eval):
        ye = np.asarray(y_eval)
        return np.where(ye < mass.y_engine, safety_factor * A_eng, 0.0)

    # 4f. Equilibrium & numerical verification checks
    M_root_direct = -safety_factor * (_trapz(w_tot * y, y) + P_eng * mass.y_engine)
    checks = {
        "CL_d": CL_d,
        "alpha_d [deg]": np.degrees(alpha),
        "CL0  integrated / file": CL0 / xflr.CL0_file,
        "CL10 integrated / file": CL10 / xflr.CL10_file,
        "lift (both wings) / (n W)": 2.0 * _trapz(L, y) / (n * case.mass * G),
        "S(root) [kN]": S[0] / 1e3,
        "M(root) [kNm]": M[0] / 1e3,
        "M(root) direct sum [kNm]": M_root_direct / 1e3,
        "T(root) [kNm]": T[0] / 1e3,
        "T(peak) [kNm]": T[np.argmax(np.abs(T))] / 1e3,
        "A(root) [kN]": A[0] / 1e3,
    }

    if verify_quad:
        w_interp = interp1d(y, w_tot, kind="cubic", fill_value="extrapolate")
        t_interp = interp1d(y, t_aero, kind="cubic", fill_value="extrapolate")
        b_half = wing.b / 2.0
        y_tank = mass.fuel_y_end_frac * b_half
        S_quad = safety_factor * (quad(lambda eta: float(w_interp(eta)), 0.0, b_half, points=[y_tank], limit=200)[0] + P_eng)
        M_quad = -safety_factor * (quad(lambda eta: float(w_interp(eta)) * eta, 0.0, b_half, points=[y_tank], limit=200)[0] + P_eng * mass.y_engine)
        T_quad = safety_factor * (quad(lambda eta: float(t_interp(eta)), 0.0, b_half, limit=200)[0] + T_eng)
        checks["S(root) quad [kN]"] = S_quad / 1e3
        checks["M(root) quad [kNm]"] = M_quad / 1e3
        checks["T(root) quad [kNm]"] = T_quad / 1e3

    return dict(
        case=case, y=y, S=S, M=M, T=T, A=A,
        N=N, w_inertia=w_inertia, t_aero=t_aero,
        S_fn=S_fn, M_fn=M_fn, T_fn=T_fn, A_fn=A_fn,
        checks=checks,
    )


# ------------------------------------------------------------------------------ 5 PLOTTING
def plot_critical_diagrams(results_list, styles, y_engine: float, out_prefix: str = "WP4_1"):
    """Generates both a combined 4-panel diagram and individual figures for S, M, T, A,
    overlaying the critical positive and negative load cases on each plot (Deliverable 4.1.b)."""
    specs = [
        ("S", "Shear force $S(y)$ [kN]", f"{out_prefix}_shear.png"),
        ("M", "Bending moment $M(y)$ [kNm]", f"{out_prefix}_bending_moment.png"),
        ("T", "Torque $T(y)$ [kNm]", f"{out_prefix}_torque.png"),
        ("A", "Axial force $A(y)$ [kN]", f"{out_prefix}_axial.png"),
    ]

    # 1. Combined 4-panel overview figure (shared legend cleanly placed above panels)
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8.5, 11.2))
    for ax, (key, ylabel, _) in zip(axes, specs):
        for res, st in zip(results_list, styles):
            ax.plot(res["y"], res[key] / 1e3, **st)
        ax.axhline(0.0, color="black", lw=0.7)
        ax.axvline(y_engine, color="dimgray", ls=":", lw=1.1,
                   label=f"Engine station ($y_{{\\mathrm{{eng}}}} = {y_engine:.2f}$ m)" if key == "S" else None)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("WP4.1 Critical Wing Internal Load Diagrams (Limit Loads, FoS = 1.0)",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962),
               ncol=1, fontsize=8.8, frameon=True, framealpha=0.95)
    axes[-1].set_xlabel("Spanwise coordinate $y$ [m]", fontsize=10.5)
    axes[-1].set_xlim(0.0, results_list[0]["y"][-1])
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.865])
    fig.savefig(f"{out_prefix}_force_diagrams_combined.png", dpi=200)
    plt.close(fig)

    # 2. Individual high-resolution plots with unobstructed legend above axes
    for key, ylabel, fname in specs:
        fig_i, ax_i = plt.subplots(figsize=(7.6, 5.0))
        for res, st in zip(results_list, styles):
            ax_i.plot(res["y"], res[key] / 1e3, **st)
        ax_i.axhline(0.0, color="black", lw=0.7)
        ax_i.axvline(y_engine, color="dimgray", ls=":", lw=1.1,
                     label=f"Engine station ($y_{{\\mathrm{{eng}}}} = {y_engine:.2f}$ m)")
        ax_i.set_xlabel("Spanwise coordinate $y$ [m]", fontsize=10.5)
        ax_i.set_ylabel(ylabel, fontsize=10.5)
        ax_i.set_xlim(0.0, results_list[0]["y"][-1])
        ax_i.grid(True, linestyle="--", alpha=0.5)
        ax_i.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=1, fontsize=8.6, frameon=True, framealpha=0.95)
        fig_i.tight_layout()
        fig_i.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig_i)


# ------------------------------------------------------------------------------ 6 RUN & SCREENING
if __name__ == "__main__":
    xflr = XFLRData("MainWing_a=0.00_v=10.00ms.csv", "MainWing_a=10.00_v=10.00ms.csv")
    wp3_wing = Wing()
    wing = xflr.wing()  # exact XFLR5 wing geometry ensures 2*int(L dy) == n*W
    masses = Masses()
    env = FlightEnvelope()

    print("=== Geometry Check (WP3 vs XFLR5) ===")
    print(f"{'Param':6s} {'WP3':>10s} {'XFLR5':>10s}")
    for attr in ("b", "S", "c_r", "c_t"):
        print(f"{attr:6s} {getattr(wp3_wing, attr):10.3f} {getattr(wing, attr):10.3f}")

    # Build and evaluate all 30 WP4.3 load cases (limit loads, safety_factor=1.0)
    all_cases = build_wp43_load_cases(env, use_excel_sl_vd=False)
    all_results = [internal_loads(cs, wing, xflr, masses, x_sc_frac=0.40, safety_factor=1.0)
                   for cs in all_cases]

    print("\n=== Complete WP4.3 Manoeuvre Load Case Screening (Limit Loads, FoS = 1.0) ===")
    header = (f"{'ID':6s} {'Pt':>2s} {'Weight':11s} {'Alt':9s} {'n':>5s} {'V_EAS':>7s} "
              f"{'CL_d':>6s} {'alpha':>6s} {'S(0)[kN]':>9s} {'M(0)[kNm]':>10s} {'T(0)[kNm]':>10s} {'A(0)[kN]':>9s}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        cs, ch = r["case"], r["checks"]
        print(f"{cs.case_id:6s} {cs.point:>2s} {cs.weight_label:11s} {cs.altitude:9s} "
              f"{cs.n:5.1f} {cs.V_eas:7.2f} {ch['CL_d']:6.3f} {ch['alpha_d [deg]']:6.2f} "
              f"{ch['S(root) [kN]']:9.1f} {ch['M(root) [kNm]']:10.1f} "
              f"{ch['T(root) [kNm]']:10.1f} {ch['A(root) [kN]']:9.1f}")

    # Identify critical positive and negative load cases
    pos_res = [r for r in all_results if r["case"].n > 0]
    neg_res = [r for r in all_results if r["case"].n < 0]

    crit_pos_bend = max(pos_res, key=lambda r: abs(r["checks"]["M(root) [kNm]"]))
    crit_pos_torq = max(pos_res, key=lambda r: abs(r["checks"]["T(peak) [kNm]"]))
    crit_neg_all = max(neg_res, key=lambda r: abs(r["checks"]["M(root) [kNm]"]))

    # Re-run critical cases with scipy.integrate.quad verification enabled
    crit_pos_D = internal_loads(crit_pos_bend["case"], wing, xflr, masses, verify_quad=True)
    crit_pos_A = internal_loads(crit_pos_torq["case"], wing, xflr, masses, verify_quad=True)
    crit_neg_F = internal_loads(crit_neg_all["case"], wing, xflr, masses, verify_quad=True)

    # Sensitivity check: negative-g critical case at idle thrust (thrust_scale = 0.0)
    cs_neg_idle = LoadCase(**{**crit_neg_all["case"].__dict__, "case_id": "LC-14 (idle)", "thrust_scale": 0.0})
    crit_neg_F_idle = internal_loads(cs_neg_idle, wing, xflr, masses)

    print("\n=== Selected Critical Load Cases & Verification Checks ===")
    for tag, res in [
        ("Critical Positive (Shear & Bending)", crit_pos_D),
        ("Critical Positive (Torsion)", crit_pos_A),
        ("Critical Negative (Shear, Bending & Torsion)", crit_neg_F),
        ("Critical Negative Sensitivity (Idle Thrust)", crit_neg_F_idle),
    ]:
        print(f"\n[{tag}]  {res['case'].label}")
        for k_chk, v_chk in res["checks"].items():
            print(f"  {k_chk:28s} {v_chk:10.3f}")

    # Plot critical positive (Point D + Point A for torsion) and critical negative (Point F)
    plot_critical_diagrams(
        [crit_pos_D, crit_pos_A, crit_neg_F],
        styles=[
            dict(color="#0055A4", lw=2.0, ls="-",
                 label=(f"Positive critical (shear & bending): {crit_pos_D['case'].case_id}, W2 MTOW, "
                        f"Pt D ($V_D={crit_pos_D['case'].V_eas:.1f}$ m/s, $n={crit_pos_D['case'].n:+.1f}$)")),
            dict(color="#E67E22", lw=1.8, ls="--",
                 label=(f"Positive critical (torsion): {crit_pos_A['case'].case_id}, W2 MTOW, "
                        f"Pt A ($V_A={crit_pos_A['case'].V_eas:.1f}$ m/s, $n={crit_pos_A['case'].n:+.1f}$)")),
            dict(color="#C0392B", lw=2.0, ls="-",
                 label=(f"Negative critical (all loads): {crit_neg_F['case'].case_id}, W2 MTOW, "
                        f"Pt F ($V_C={crit_neg_F['case'].V_eas:.1f}$ m/s, $n={crit_neg_F['case'].n:+.1f}$)")),
        ],
        y_engine=masses.y_engine,
        out_prefix="WP4_1",
    )
    print("\nSaved figures: WP4_1_force_diagrams_combined.png, WP4_1_shear.png, "
          "WP4_1_bending_moment.png, WP4_1_torque.png, WP4_1_axial.png")
