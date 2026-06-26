# -*- coding: utf-8 -*-
"""Interactive 3-D geometry helpers for the Curtis study notebooks.

``plot_orbital_elements`` draws the geocentric-equatorial frame, the reference
(equatorial) plane and the tilted orbital plane, the orbit itself, and the four
classical angles -- Omega (RAAN), i (inclination), omega (argument of perigee)
and theta (true anomaly) -- as labelled, colour-filled wedges. This is the
picture behind Algorithms 4.1 / 4.2.  Rotate it with the mouse.

The ``focus`` argument lets you reveal the elements one at a time::

    plot_orbital_elements(focus="frame").show()        # just the I,J,K frame
    plot_orbital_elements(focus="raan").show()         # + Omega
    plot_orbital_elements(focus="inclination").show()  # + i
    plot_orbital_elements(focus="argp").show()         # + omega
    plot_orbital_elements(focus="ta").show()           # + theta
    plot_orbital_elements(focus="all").show()          # everything (default)

In a focused view the highlighted element is drawn boldly while the rest of the
scene dims to context, and the camera is identical across all views, so flipping
between them feels like layers switching on and off over a fixed stage.

Requires plotly (see requirements.txt).
"""

import numpy as np
import plotly.graph_objects as go

# --- palette (shared with the SVG / study notes) -------------------------
C_RAAN = "#2F7FD1"   # Omega
C_INC = "#15A06F"    # i
C_ARGP = "#E0651F"   # omega
C_TA = "#D23E6E"     # theta

C_AXIS = "#6B6A66"        # frame axes
C_AXIS_TXT = "#33332f"
C_ORBIT = "#55544f"       # orbit ellipse
C_NODE = "#5a5954"        # line of nodes
C_EQ_FILL = "#9b998f"     # equatorial (reference) plane
C_ORB_FILL = "#C98A5E"    # orbital plane (warm tint)
C_R = "#444039"           # radius vector


# --- rotations / frame transform -----------------------------------------
def _R3(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _R1(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _perifocal_to_eci(raan, inc, argp):
    """Q such that r_eci = Q @ r_perifocal (the 3-1-3 transform, Eq 4.44)."""
    return (_R3(argp) @ _R1(inc) @ _R3(raan)).T


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _slerp(a, b, n):
    """n unit vectors sweeping the short great-circle arc from a to b."""
    a, b = _unit(a), _unit(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    phi = np.arccos(dot)
    t = np.linspace(0, 1, n)
    if phi < 1e-9:
        return np.tile(a, (n, 1))
    return (np.sin((1 - t)[:, None] * phi) * a
            + np.sin(t[:, None] * phi) * b) / np.sin(phi)


# --- primitive trace builders --------------------------------------------
def _line(p0, p1, color, width=4, dash=None, opacity=1.0):
    return go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines", line=dict(color=color, width=width, dash=dash),
        opacity=opacity, showlegend=False, hoverinfo="skip")


def _cone(tip, direction, color, size=0.11, opacity=1.0):
    d = _unit(direction)
    return go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode="absolute", sizeref=size, anchor="tip", opacity=opacity,
        colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip")


def _text(p, txt, color, size=14, weight=None, opacity=1.0):
    t = txt if weight is None else f"<b>{txt}</b>"
    return go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="text",
        text=[t], textfont=dict(color=color, size=size),
        opacity=opacity, showlegend=False, hoverinfo="skip")


def _marker(p, color, size=5, opacity=1.0):
    return go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
        marker=dict(size=size, color=color), opacity=opacity,
        showlegend=False, hoverinfo="skip")


_FLAT = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0)


def _disk(normal, radius, color, opacity, n=90):
    """A translucent flat disk through the origin with the given normal."""
    normal = _unit(normal)
    seed = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(normal, seed))
    v = np.cross(normal, u)
    ang = np.linspace(0, 2 * np.pi, n)
    ring = (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v) * radius
    pts = np.vstack([[0, 0, 0], ring])
    i = np.zeros(n - 1, dtype=int)
    j = np.arange(1, n)
    k = np.arange(2, n + 1)
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=i, j=j, k=k,
        color=color, opacity=opacity, lighting=_FLAT,
        hoverinfo="skip", showlegend=False)


def _wedge(vertex, a_dir, b_dir, radius, color, opacity, n=56):
    """A translucent filled circular sector (the body of an angle)."""
    sl = _slerp(a_dir, b_dir, n) * radius + vertex
    pts = np.vstack([vertex, sl])
    i = np.zeros(n - 1, dtype=int)
    j = np.arange(1, n)
    k = np.arange(2, n + 1)
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=i, j=j, k=k,
        color=color, opacity=opacity, lighting=_FLAT,
        hoverinfo="skip", showlegend=False)


def _legend_proxy(color, name, mode="lines"):
    return go.Scatter3d(
        x=[None], y=[None], z=[None], mode=mode,
        line=dict(color=color, width=10), marker=dict(color=color, size=11),
        name=name, showlegend=True, hoverinfo="skip")


def _angle(fig, vertex, a_dir, b_dir, radius, color, label,
           label_at=1.22, fill_opacity=0.33, arc_w=8):
    """Draw one angle: filled wedge + bright arc + Greek label."""
    a_dir, b_dir = _unit(a_dir), _unit(b_dir)
    fig.add_trace(_wedge(vertex, a_dir, b_dir, radius, color, fill_opacity))
    sl = _slerp(a_dir, b_dir, 64) * radius + vertex
    fig.add_trace(go.Scatter3d(
        x=sl[:, 0], y=sl[:, 1], z=sl[:, 2], mode="lines",
        line=dict(color=color, width=arc_w), showlegend=False, hoverinfo="skip"))
    mid = _slerp(a_dir, b_dir, 3)[1]
    fig.add_trace(_text(vertex + mid * radius * label_at, label, color,
                        size=22, weight="b"))


# --- per-focus stage settings --------------------------------------------
#  eq/orb : plane opacities   orbit : orbit-line opacity (0 hides it)
#  nodes/node/peri/sat : whether to draw those supporting pieces
#  hero : which angle is highlighted ("all" -> draw all four)
_CFG = {
    "frame":       dict(eq=0.17, orb=0.00, orbit=0.00, nodes=False,
                        node=False, peri=False, sat=False, hero=None),
    "raan":        dict(eq=0.17, orb=0.06, orbit=0.22, nodes=True,
                        node=True, peri=False, sat=False, hero="raan"),
    "inclination": dict(eq=0.13, orb=0.17, orbit=0.22, nodes=True,
                        node=True, peri=False, sat=False, hero="inc"),
    "argp":        dict(eq=0.06, orb=0.17, orbit=0.85, nodes=True,
                        node=True, peri=True, sat=False, hero="argp"),
    "ta":          dict(eq=0.05, orb=0.17, orbit=0.85, nodes=False,
                        node=False, peri=True, sat=True, hero="ta"),
    "all":         dict(eq=0.11, orb=0.13, orbit=0.85, nodes=True,
                        node=True, peri=True, sat=True, hero="all"),
}

_TITLE = {
    "frame": ("The geocentric-equatorial (I, J, K) frame",
              "I \u2192 vernal equinox \u00b7 K \u2192 rotation axis "
              "\u00b7 the equatorial reference plane"),
    "raan": ("\u03A9 \u2014 right ascension of the ascending node",
             "in the equatorial plane, from the vernal equinox (I) "
             "round to the ascending node"),
    "inclination": ("i \u2014 inclination",
                    "the tilt between the equatorial and orbital planes, "
                    "measured at the ascending node"),
    "argp": ("\u03C9 \u2014 argument of perigee",
             "in the orbital plane, from the ascending node round to perigee"),
    "ta": ("\u03B8 \u2014 true anomaly",
           "in the orbital plane, from perigee round to the satellite"),
    "all": ("Classical orbital elements in the geocentric-equatorial frame",
            "drag to rotate"),
}


def plot_orbital_elements(raan_deg=40.0, inc_deg=40.0, argp_deg=60.0,
                          ta_deg=45.0, e=0.30, a=1.0, focus="all"):
    """Build an interactive 3-D figure of the frame, planes, orbit and angles.

    ``focus`` is one of ``"frame"``, ``"raan"``, ``"inclination"``, ``"argp"``,
    ``"ta"`` or ``"all"`` (default) and reveals the elements one at a time.
    Returns a plotly Figure; call ``.show()``.
    """
    focus = focus.lower()
    if focus not in _CFG:
        raise ValueError(f"focus must be one of {tuple(_CFG)}; got {focus!r}")
    cfg = _CFG[focus]
    hero = cfg["hero"]
    everything = hero == "all"

    d = np.pi / 180.0
    raan, inc, argp, ta = raan_deg * d, inc_deg * d, argp_deg * d, ta_deg * d
    Q = _perifocal_to_eci(raan, inc, argp)
    p = a * (1 - e ** 2)

    def at(theta):
        r = p / (1 + e * np.cos(theta))
        return Q @ np.array([r * np.cos(theta), r * np.sin(theta), 0.0])

    R = 1.40
    fig = go.Figure()

    # key directions / points ------------------------------------------------
    Ihat = np.array([1.0, 0.0, 0.0])
    Wn = Q @ np.array([0.0, 0.0, 1.0])
    node_dir = np.array([np.cos(raan), np.sin(raan), 0.0])
    peri_dir = _unit(Q @ np.array([1.0, 0.0, 0.0]))
    node_pt = at(-argp)
    peri_pt = at(0.0)
    sat_pt = at(ta)
    sat_dir = _unit(sat_pt)
    O = np.zeros(3)

    # --- the two planes ------------------------------------------------------
    if cfg["eq"] > 0:
        fig.add_trace(_disk([0, 0, 1], R, C_EQ_FILL, cfg["eq"]))
    if cfg["orb"] > 0:
        fig.add_trace(_disk(Wn, R * 0.96, C_ORB_FILL, cfg["orb"]))

    # --- I, J, K axes --------------------------------------------------------
    ax_op = 1.0 if focus in ("frame", "raan", "all") else 0.5
    for axis, lbl, off, show_lbl in [
        (np.array([1, 0, 0]), "I  (vernal equinox)", 1.30, True),
        (np.array([0, 1, 0]), "J", 1.12, focus in ("frame", "all")),
        (np.array([0, 0, 1]), "K  (rotation axis)", 1.13, True),
    ]:
        tip = axis * R * 0.92
        fig.add_trace(_line(O, tip, C_AXIS, width=3, opacity=ax_op))
        fig.add_trace(_cone(tip, axis, C_AXIS, opacity=ax_op))
        if show_lbl:
            fig.add_trace(_text(axis * R * off, lbl, C_AXIS_TXT, 13,
                                opacity=ax_op))

    # --- line of nodes -------------------------------------------------------
    if cfg["nodes"]:
        fig.add_trace(_line(-node_dir * R, node_dir * R, C_NODE, width=2,
                            dash="dash"))
        fig.add_trace(_text(-node_dir * (R * 0.98) + np.array([0, 0, -0.10]),
                            "line of nodes", C_NODE, 11))

    # --- the orbit -----------------------------------------------------------
    if cfg["orbit"] > 0:
        thetas = np.linspace(0, 2 * np.pi, 260)
        pts = np.array([at(t) for t in thetas])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line=dict(color=C_ORBIT, width=6), opacity=cfg["orbit"],
            showlegend=False, hoverinfo="skip"))

    # --- apse line + radius vector ------------------------------------------
    if cfg["peri"]:
        fig.add_trace(_line(O, peri_pt, C_NODE, width=2))
    if cfg["sat"]:
        fig.add_trace(_line(O, sat_pt, C_R, width=5))
        fig.add_trace(_cone(sat_pt, sat_pt, C_R, size=0.10))
        fig.add_trace(_text(sat_pt * 0.48 + np.array([-0.05, -0.10, 0.12]),
                            "r", C_R, 16, weight="b"))

    # --- Earth + sparse point labels ----------------------------------------
    fig.add_trace(_marker(O, "#2E6FB8", size=9))
    fig.add_trace(_text(O + np.array([-0.16, -0.05, -0.16]), "O", C_AXIS_TXT, 15,
                        weight="b"))
    pt_labels = []
    if cfg["node"]:
        pt_labels.append((node_pt, "ascending node", C_AXIS_TXT,
                          np.array([-0.06, 0.04, -0.20])))
    if cfg["peri"]:
        pt_labels.append((peri_pt, "perigee", C_ARGP,
                          peri_dir * 0.18 + np.array([0, 0, -0.12])))
    if cfg["sat"]:
        pt_labels.append((sat_pt, "satellite", C_TA,
                          np.array([0.02, 0.02, 0.20])))
    for pt, txt, col, off in pt_labels:
        fig.add_trace(_marker(pt, col, size=6))
        fig.add_trace(_text(pt + off, txt, col, 13))

    # --- the angle(s) --------------------------------------------------------
    fo = 0.40 if not everything else 0.33     # heroes a touch bolder
    aw = 9 if not everything else 8
    o1 = np.cross(Wn, node_dir)
    o1 = _unit(o1 if o1[2] >= 0 else -o1)
    eq_perp = np.array([-np.sin(raan), np.cos(raan), 0.0])
    if np.dot(eq_perp, o1) < 0:
        eq_perp = -eq_perp
    inc_vertex = node_dir * 0.95

    def draw(name):
        if name == "raan":
            _angle(fig, O, Ihat, node_dir, 0.55, C_RAAN, "\u03A9",
                   label_at=1.18, fill_opacity=fo, arc_w=aw)
        elif name == "inc":
            _angle(fig, inc_vertex, eq_perp, o1, 0.30, C_INC, "i",
                   label_at=1.34, fill_opacity=fo, arc_w=aw)
        elif name == "argp":
            _angle(fig, O, node_dir, peri_dir, 0.62, C_ARGP, "\u03C9",
                   label_at=1.20, fill_opacity=fo, arc_w=aw)
        elif name == "ta":
            _angle(fig, O, peri_dir, sat_dir, 0.40, C_TA, "\u03B8",
                   label_at=1.30, fill_opacity=fo, arc_w=aw)

    if everything:
        for nm in ("raan", "argp", "ta", "inc"):
            draw(nm)
    elif hero is not None:
        draw(hero)

    # --- legend (adapts to what's on stage) ---------------------------------
    if everything:
        for col, name in [
            (C_RAAN, "\u03A9 \u2014 right ascension of ascending node"),
            (C_INC, "i \u2014 inclination"),
            (C_ARGP, "\u03C9 \u2014 argument of perigee"),
            (C_TA, "\u03B8 \u2014 true anomaly"),
        ]:
            fig.add_trace(_legend_proxy(col, name))
        fig.add_trace(_legend_proxy(C_ORBIT, "orbit"))
        fig.add_trace(_legend_proxy(C_EQ_FILL, "equatorial (reference) plane",
                                    mode="markers"))
        fig.add_trace(_legend_proxy(C_ORB_FILL, "orbital plane", mode="markers"))
    else:
        if cfg["eq"] >= 0.10:
            fig.add_trace(_legend_proxy(C_EQ_FILL,
                                        "equatorial (reference) plane",
                                        mode="markers"))
        if cfg["orb"] >= 0.10:
            fig.add_trace(_legend_proxy(C_ORB_FILL, "orbital plane",
                                        mode="markers"))
        if cfg["orbit"] >= 0.5:
            fig.add_trace(_legend_proxy(C_ORBIT, "orbit"))

    show_legend = (focus != "frame")
    head, sub = _TITLE[focus]
    title = f"{head}  \u00b7  {sub}" if not everything else \
        f"{head}  \u00b7  {sub}"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        showlegend=show_legend,
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.72)", bordercolor="#d9d8d2",
            borderwidth=1, font=dict(size=12), itemsizing="constant"),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="data",
            camera=dict(eye=dict(x=0.96, y=1.05, z=0.56),
                        center=dict(x=0, y=0, z=0.06),
                        up=dict(x=0, y=0, z=1)),
            dragmode="orbit", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=46, b=0), height=620,
        paper_bgcolor="rgba(0,0,0,0)")
    return fig


if __name__ == "__main__":
    for f in ("frame", "raan", "inclination", "argp", "ta", "all"):
        plot_orbital_elements(focus=f).show()